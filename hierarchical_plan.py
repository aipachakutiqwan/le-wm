"""Hierarchical LeWM — two-level CEM-MPC planner.

Entry points
------------
plan()          Single-environment two-level CEM-MPC (backward-compatible).
plan_batched()  Batched version: E environments planned in one shot by
                flattening to a single E×S rollout call per CEM iteration.
                Eliminates the Python loop over environments in
                HierarchicalPolicy.get_action and keeps the GPU saturated.

Utilities
---------
cem()                   Minimal diagonal-Gaussian CEM (single env).
compile_for_planning()  Apply torch.compile to the autoregressive rollout
                        helpers so the fixed-length inner loops are fused.

Both operate on a trained HierarchicalLeWM from hierarchical_lewm.py.
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


# ──────────────────────────────────────────────────────────────────────────────
# Batched two-level CEM-MPC  (E environments in parallel)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def plan_batched(
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
    stats: dict | None = None,
    step: int | None = None,
) -> torch.Tensor:
    """Two-level CEM-MPC for E environments in a single batched pass.

    All E CEMs share one flattened rollout call per iteration:
      outer: (E × outer_samples, H_high, d_L) through _rollout_high
      inner: (E × inner_samples, h_low, action_dim) through _rollout_low

    This eliminates the Python for-loop over environments and fully saturates
    the GPU when E is large (e.g., the 50-episode eval batch).

    Parameters
    ----------
    model         : trained HierarchicalLeWM
    z_init        : (E, D) current latent states
    z_goal        : (E, D) goal latent states
    H_high        : macro-action steps for the outer CEM
    h_low         : primitive steps for the inner CEM
    outer_samples : CEM candidates per environment per outer iteration
    inner_samples : CEM candidates per environment per inner iteration
    outer_iters   : outer CEM iterations
    inner_iters   : inner CEM iterations
    outer_std     : initial outer CEM std
    inner_std     : initial inner CEM std
    stats         : optional mutable dict; same keys as plan() (n_calls, total_ms, …)
    step          : MPC step index for logging

    Returns
    -------
    (E, action_dim) — first primitive action to execute for each environment
    """
    E, D = z_init.shape
    device = z_init.device
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    amp_enabled = device_type == "cuda"
    d_L = model.latent_action_dim
    t0 = time.perf_counter()

    step_tag = f"step={step}  " if step is not None else ""

    # Progressive subgoal weights: later waypoints weighted higher.
    w_high = torch.linspace(1.0 / H_high, 1.0, H_high, device=device)
    w_high = w_high / w_high.sum()                                   # (H_high,)

    # ── Outer CEM ─────────────────────────────────────────────────────────────
    mu_mac  = torch.zeros(E, H_high, d_L, device=device)
    std_mac = torch.full((E, H_high, d_L), outer_std, device=device)
    n_elites_o = max(1, int(outer_samples * 0.1))

    # Expand z_init once for the outer loop: (E, outer_samples, D) → (E*S, D)
    z_init_o = z_init.unsqueeze(1).expand(-1, outer_samples, -1).reshape(E * outer_samples, D)

    t_outer = time.perf_counter()
    for _ in range(outer_iters):
        # (E, S, H_high, d_L)
        cands = mu_mac.unsqueeze(1) + std_mac.unsqueeze(1) * torch.randn(
            E, outer_samples, H_high, d_L, device=device
        )
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            subgoals = model._rollout_high(
                z_init_o, cands.reshape(E * outer_samples, H_high, d_L)
            )                                                         # (E*S, H_high, D)

        subgoals = subgoals.reshape(E, outer_samples, H_high, -1)
        # weighted L1 to z_goal across subgoal steps
        dists = (subgoals - z_goal.unsqueeze(1).unsqueeze(2)).abs().sum(-1)  # (E, S, H_high)
        costs = (dists * w_high).sum(-1)                              # (E, S)

        elite_idx = costs.argsort(dim=1)[:, :n_elites_o]             # (E, K)
        elites = cands.gather(
            1,
            elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_high, d_L),
        )                                                             # (E, K, H_high, d_L)
        mu_mac  = elites.mean(dim=1)
        std_mac = elites.std(dim=1).clamp(min=0.1)

    outer_ms = (time.perf_counter() - t_outer) * 1e3

    # ── First subgoal per environment ──────────────────────────────────────────
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        z_sg = model._rollout_high(z_init, mu_mac)[:, 0]             # (E, D)

    # ── Inner CEM ─────────────────────────────────────────────────────────────
    action_dim = model.action_dim
    mu_act  = torch.zeros(E, h_low, action_dim, device=device)
    std_act = torch.full((E, h_low, action_dim), inner_std, device=device)
    n_elites_i = max(1, int(inner_samples * 0.1))

    z_init_i = z_init.unsqueeze(1).expand(-1, inner_samples, -1).reshape(E * inner_samples, D)

    t_inner = time.perf_counter()
    for _ in range(inner_iters):
        # (E, S, h_low, action_dim)
        cands = mu_act.unsqueeze(1) + std_act.unsqueeze(1) * torch.randn(
            E, inner_samples, h_low, action_dim, device=device
        )
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            z_final = model._rollout_low(
                z_init_i, cands.reshape(E * inner_samples, h_low, action_dim)
            )                                                         # (E*S, D)

        costs = (z_final.reshape(E, inner_samples, D) - z_sg.unsqueeze(1)).abs().sum(-1)  # (E, S)

        elite_idx = costs.argsort(dim=1)[:, :n_elites_i]
        elites = cands.gather(
            1,
            elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_low, action_dim),
        )
        mu_act  = elites.mean(dim=1)
        std_act = elites.std(dim=1).clamp(min=0.1)

    inner_ms = (time.perf_counter() - t_inner) * 1e3
    total_ms = (time.perf_counter() - t0) * 1e3

    py_log.info(
        "%sbatched E=%d  outer=%.1f ms  inner=%.1f ms  total=%.1f ms",
        step_tag, E, outer_ms, inner_ms, total_ms,
    )

    if stats is not None:
        stats["n_calls"]  = stats.get("n_calls",  0)   + 1
        stats["total_ms"] = stats.get("total_ms", 0.0) + total_ms
        stats["outer_ms"] = stats.get("outer_ms", 0.0) + outer_ms
        stats["inner_ms"] = stats.get("inner_ms", 0.0) + inner_ms

    return mu_act[:, 0]   # (E, action_dim)


# ──────────────────────────────────────────────────────────────────────────────
# torch.compile helper
# ──────────────────────────────────────────────────────────────────────────────


def compile_for_planning(model: HierarchicalLeWM, mode: str = "reduce-overhead") -> HierarchicalLeWM:
    """torch.compile the autoregressive rollout helpers on a HierarchicalLeWM.

    Call once after loading a checkpoint, before the planning loop.  The
    "reduce-overhead" mode trades a one-time ~2 min warm-up for significantly
    lower per-call overhead on the repeated fixed-shape rollout loops.

    Parameters
    ----------
    model : HierarchicalLeWM returned by torch.load(checkpoint)
    mode  : torch.compile mode — "reduce-overhead" (default) suits planning
            because the rollout shapes are fixed and calls are frequent;
            "default" or "max-autotune" are alternatives for longer runs.

    Returns
    -------
    The same model with _rollout_high and _rollout_low replaced by compiled
    versions (instance-level attribute, class method unaffected).
    """
    model._rollout_high = torch.compile(model._rollout_high, mode=mode)
    model._rollout_low  = torch.compile(model._rollout_low,  mode=mode)
    py_log.info("Planning rollouts compiled with torch.compile(mode=%r)", mode)
    return model
