"""Hierarchical LeWM — two-level CEM-MPC planner.

Entry point: plan()
Utility:     cem()

Both operate on a trained HierarchicalLeWM from hierarchical_lewm.py.
The rollout helpers (_rollout_high, _rollout_low) live on the model because
they directly use its weights; planning logic that is independent of model
parameters lives here.
"""

import logging
import time

import torch

from hierarchical_lewm import HierarchicalLeWM

py_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CEM
# ──────────────────────────────────────────────────────────────────────────────


def cem(
    cost_fn,
    mu: torch.Tensor,
    std: torch.Tensor,
    n_samples: int = 512,
    n_iters: int = 5,
    elite_frac: float = 0.1,
) -> torch.Tensor:
    """Minimal diagonal-Gaussian Cross-Entropy Method.

    Parameters
    ----------
    cost_fn    : callable (n_samples, *shape) -> (n_samples,) — lower is better
    mu         : (*shape,) initial mean
    std        : (*shape,) initial std
    n_samples  : number of candidates sampled per iteration
    n_iters    : number of CEM iterations
    elite_frac : fraction of candidates kept as elites

    Returns
    -------
    (*shape,) optimised mean
    """
    n_elites = max(1, int(n_samples * elite_frac))
    for i in range(n_iters):
        t_iter = time.perf_counter()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        candidates = mu.unsqueeze(0) + std.unsqueeze(0) * eps   # (S, *shape)
        costs = cost_fn(candidates)                              # (S,)
        elite_idx = costs.argsort()[:n_elites]
        elites = candidates[elite_idx]
        mu = elites.mean(0)
        std = elites.std(0).clamp(min=0.1)
        py_log.debug(
            "  cem iter %d/%d  best_cost=%.4f  std_mean=%.3f  %.1f ms",
            i + 1, n_iters, costs[elite_idx[0]].item(), std.mean().item(),
            (time.perf_counter() - t_iter) * 1e3,
        )
    return mu


# ──────────────────────────────────────────────────────────────────────────────
# Two-level CEM-MPC
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def plan(
    model: HierarchicalLeWM,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    H_high: int = 3,
    h_low: int = 10,
    outer_samples: int = 512,
    inner_samples: int = 256,
    outer_iters: int = 5,
    inner_iters: int = 5,
    outer_std: float = 5.0,
    inner_std: float = 1.0,
    warmstart: dict | None = None,
    stats: dict | None = None,
    step: int | None = None,
) -> torch.Tensor:
    """Two-level CEM-MPC. Returns the first primitive action to execute.

    Outer CEM
    ---------
    Optimises H_high latent macro-actions in R^{H_high × d_L} by minimising
    the L1 distance between the final P^(2) rollout state and z_goal.

    Inner CEM
    ---------
    Given the first subgoal from the winning macro-action sequence, optimises
    h_low primitive actions by minimising the L1 distance between the final
    P^(1) rollout state and that subgoal.

    Call this every K steps and re-plan with the updated observation (MPC loop).

    Parameters
    ----------
    model          : trained HierarchicalLeWM
    z_init         : (D,) current latent state
    z_goal         : (D,) goal latent state
    H_high         : number of high-level macro-action steps
    h_low          : number of low-level primitive steps per subgoal
    outer_samples  : CEM sample count for the outer loop
    inner_samples  : CEM sample count for the inner loop
    outer_iters    : CEM iterations for the outer loop
    inner_iters    : CEM iterations for the inner loop
    outer_std      : initial CEM std for the outer (macro-action) loop. 5.0 matches
                     A_ψ's empirical output spread; the original 1.0 under-explored.
    inner_std      : initial CEM std for the inner (primitive-action) loop. Must
                     roughly match the dataset action scale (StandardScaler-normalised
                     actions have std~1.0); too small starves the search of exploration.
    warmstart      : mutable dict optionally carrying ``"mu_mac": (H_high, d_L)`` from
                     the previous call. When present, the outer CEM is initialised from
                     that tensor with halved std (near-solution warm-start). Updated
                     in-place after each call so callers only need to pass the same dict
                     every re-plan step.
    stats          : optional mutable dict for accumulating totals across calls.
                     Keys updated in-place: ``n_calls``, ``total_ms``, ``outer_ms``,
                     ``inner_ms``.  Pass the same dict on every MPC step then read it
                     at the end for a full planning session summary.

    Returns
    -------
    (action_dim,) — first primitive action to execute
    """
    device = z_init.device
    d_L = model.latent_action_dim
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = device_type == "cuda"
    t0 = time.perf_counter()

    step_tag = f"step={step}  " if step is not None else ""
    dist_init_goal = (z_goal - z_init).abs().sum().item()

    # Progress: Δ vs previous call and % of initial distance covered.
    if stats is not None:
        init_dist0 = stats.setdefault("init_dist0", dist_init_goal)
        delta = stats.get("prev_dist", dist_init_goal) - dist_init_goal  # +ve = closer
        pct   = 100.0 * (1.0 - dist_init_goal / init_dist0) if init_dist0 > 0 else 0.0
        prog_str = f"  Δ={delta:+.4f}  done={pct:.1f}%"
        stats["prev_dist"] = dist_init_goal
    else:
        prog_str = ""

    py_log.debug("%sz_init→z_goal L1=%.4f%s", step_tag, dist_init_goal, prog_str)

    # Progressive weights: later subgoals matter more; normalised to sum = 1.
    w_high = torch.linspace(1.0 / H_high, 1.0, H_high, device=device)
    w_high = w_high / w_high.sum()                                        # (H,)

    # ── Outer CEM: optimise macro-action sequence ─────────────────────────────
    warm = warmstart is not None and "mu_mac" in warmstart
    if warm:
        mu_mac = warmstart["mu_mac"].to(device)
        std_mac = torch.full((H_high, d_L), outer_std / 2, device=device)
    else:
        mu_mac = torch.zeros(H_high, d_L, device=device)
        std_mac = torch.full((H_high, d_L), outer_std, device=device)

    py_log.debug(
        "outer CEM — S=%d  iters=%d  H=%d  warm=%s",
        outer_samples, outer_iters, H_high, warm,
    )

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, H_high, d_L)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            subgoals = model._rollout_high(z_init, candidates)             # (S, H_high, D)
        dists = (subgoals - z_goal.unsqueeze(0).unsqueeze(1)).abs().sum(-1)  # (S, H)
        return (dists * w_high.unsqueeze(0)).sum(-1)                       # (S,)

    t_outer = time.perf_counter()
    best_mac = cem(outer_cost, mu_mac, std_mac, outer_samples, outer_iters)
    outer_ms = (time.perf_counter() - t_outer) * 1e3
    outer_cost_val = outer_cost(best_mac.unsqueeze(0)).item()
    py_log.info("%souter CEM — best_cost=%.4f  dist=%.4f%s  %.1f ms",
                step_tag, outer_cost_val, dist_init_goal, prog_str, outer_ms)

    if warmstart is not None:
        warmstart["mu_mac"] = best_mac.cpu()

    # ── Derive first subgoal ──────────────────────────────────────────────────
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        z_sg = model._rollout_high(z_init, best_mac.unsqueeze(0))[:, 0].squeeze(0)  # (D,)
    dist_init_sg = (z_sg - z_init).abs().sum().item()
    dist_sg_goal = (z_goal - z_sg).abs().sum().item()
    py_log.debug("%ssubgoal — z_init→z_sg L1=%.4f  z_sg→z_goal L1=%.4f",
                 step_tag, dist_init_sg, dist_sg_goal)

    # ── Inner CEM: optimise primitive actions to reach z_sg ──────────────────
    mu_act = torch.zeros(h_low, model.action_dim, device=device)
    std_act = torch.full((h_low, model.action_dim), inner_std, device=device)

    py_log.debug(
        "inner CEM — S=%d  iters=%d  h=%d",
        inner_samples, inner_iters, h_low,
    )

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, h_low, action_dim)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            z_final = model._rollout_low(z_init, candidates)               # (S, D)
        return (z_final - z_sg.unsqueeze(0)).abs().sum(-1)                 # (S,)

    t_inner = time.perf_counter()
    best_act = cem(inner_cost, mu_act, std_act, inner_samples, inner_iters)
    inner_ms = (time.perf_counter() - t_inner) * 1e3
    inner_cost_val = inner_cost(best_act.unsqueeze(0)).item()
    total_ms = (time.perf_counter() - t0) * 1e3
    py_log.info(
        "%sinner CEM — best_cost=%.4f  %.1f ms  |  total=%.1f ms",
        step_tag, inner_cost_val, inner_ms, total_ms,
    )

    if stats is not None:
        stats["n_calls"]  = stats.get("n_calls",  0)   + 1
        stats["total_ms"] = stats.get("total_ms", 0.0) + total_ms
        stats["outer_ms"] = stats.get("outer_ms", 0.0) + outer_ms
        stats["inner_ms"] = stats.get("inner_ms", 0.0) + inner_ms

    return best_act[0]   # first primitive action: (action_dim,)
