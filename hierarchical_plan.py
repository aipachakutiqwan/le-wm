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

log = logging.getLogger(__name__)


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
    name: str = "cem",
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
    name       : label used in log messages

    Returns
    -------
    (*shape,) optimised mean
    """
    n_elites = max(1, int(n_samples * elite_frac))
    t_cem = time.perf_counter()
    for i in range(n_iters):
        t_iter = time.perf_counter()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        candidates = mu.unsqueeze(0) + std.unsqueeze(0) * eps   # (S, *shape)
        costs = cost_fn(candidates)                              # (S,)
        elite_idx = costs.argsort()[:n_elites]
        elites = candidates[elite_idx]
        mu = elites.mean(0)
        std = elites.std(0).clamp(min=1e-4)
        log.debug(
            "[%s] iter %d/%d  best_cost=%.4f  mean_cost=%.4f  std_mean=%.4f  iter_ms=%.1f",
            name, i + 1, n_iters,
            costs[elite_idx[0]].item(),
            costs.mean().item(),
            std.mean().item(),
            (time.perf_counter() - t_iter) * 1e3,
        )
    log.debug("[%s] total_ms=%.1f", name, (time.perf_counter() - t_cem) * 1e3)
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

    Returns
    -------
    (action_dim,) — first effective action to execute (action_dim = frameskip * base_dim)
    """
    device = z_init.device
    d_L = model.latent_action_dim

    log.info(
        "plan: H_high=%d h_low=%d outer=%d×%d inner=%d×%d d_L=%d",
        H_high, h_low, outer_samples, outer_iters, inner_samples, inner_iters, d_L,
    )
    t_plan = time.perf_counter()

    # ── Outer CEM: optimise macro-action sequence ─────────────────────────────
    mu_mac = torch.zeros(H_high, d_L, device=device)
    std_mac = torch.ones(H_high, d_L, device=device)

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, H_high, d_L)
        subgoals = model._rollout_high(z_init, candidates)          # (S, H_high, D)
        # Linearly increasing weights: later subgoals penalised more for being far from goal.
        # This encourages progressive approach (z_1 < z_2 < z_last in distance to goal)
        # while still prioritising the final subgoal landing near z_goal.
        H = subgoals.shape[1]
        w = torch.linspace(1.0 / H, 1.0, H, device=device)         # (H,) e.g. [0.33, 0.67, 1.0]
        w = w / w.sum()                                              # normalise to sum=1
        dists = (subgoals - z_goal.unsqueeze(0).unsqueeze(1)).abs().sum(-1)  # (S, H)
        return (dists * w.unsqueeze(0)).sum(-1)                      # (S,)

    t_outer = time.perf_counter()
    best_mac = cem(outer_cost, mu_mac, std_mac, outer_samples, outer_iters, name="outer")
    # best_mac: (H_high, d_L)

    outer_final_cost = outer_cost(best_mac.unsqueeze(0))[0].item()
    log.info("outer CEM done — best_cost=%.4f  ms=%.1f", outer_final_cost, (time.perf_counter() - t_outer) * 1e3)

    # ── Derive first subgoal ──────────────────────────────────────────────────
    z_sg = model._rollout_high(z_init, best_mac.unsqueeze(0))[:, 0].squeeze(0)  # (D,)
    sg_dist = (z_sg - z_goal).abs().sum().item()
    log.info("subgoal derived — dist_to_goal=%.4f", sg_dist)

    # ── Inner CEM: optimise primitive actions to reach z_sg ──────────────────
    mu_act = torch.zeros(h_low, model.action_dim, device=device)
    std_act = torch.full((h_low, model.action_dim), 0.5, device=device)

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, h_low, action_dim)
        z_final = model._rollout_low(z_init, candidates)       # (S, D)
        return (z_final - z_sg.unsqueeze(0)).abs().sum(-1)     # (S,)

    t_inner = time.perf_counter()
    best_act = cem(inner_cost, mu_act, std_act, inner_samples, inner_iters, name="inner")
    # best_act: (h_low, action_dim)

    inner_final_cost = inner_cost(best_act.unsqueeze(0))[0].item()
    log.info(
        "inner CEM done — best_cost=%.4f  action_norm=%.4f  ms=%.1f",
        inner_final_cost, best_act[0].norm().item(), (time.perf_counter() - t_inner) * 1e3,
    )
    log.info("plan total_ms=%.1f", (time.perf_counter() - t_plan) * 1e3)

    return best_act[0]   # (action_dim,) — effective action covering frameskip primitive steps
