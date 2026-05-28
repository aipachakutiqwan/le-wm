"""Hierarchical LeWM — two-level CEM-MPC planner.

Entry point: plan()
Utility:     cem()

Both operate on a trained HierarchicalLeWM from hierarchical_lewm.py.
The rollout helpers (_rollout_high, _rollout_low) live on the model because
they directly use its weights; planning logic that is independent of model
parameters lives here.
"""

import logging

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
    for _ in range(n_iters):
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        candidates = mu.unsqueeze(0) + std.unsqueeze(0) * eps   # (S, *shape)
        costs = cost_fn(candidates)                              # (S,)
        elite_idx = costs.argsort()[:n_elites]
        elites = candidates[elite_idx]
        mu = elites.mean(0)
        std = elites.std(0).clamp(min=0.1)
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
    (action_dim,) — first primitive action to execute
    """
    device = z_init.device
    d_L = model.latent_action_dim

    # ── Outer CEM: optimise macro-action sequence ─────────────────────────────
    mu_mac = torch.zeros(H_high, d_L, device=device)
    std_mac = torch.full((H_high, d_L), 5.0, device=device)

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, H_high, d_L)
        subgoals = model._rollout_high(z_init, candidates)     # (S, H_high, D)
        z_last = subgoals[:, -1]                               # (S, D)
        return (z_last - z_goal.unsqueeze(0)).abs().sum(-1)    # (S,)

    best_mac = cem(outer_cost, mu_mac, std_mac, outer_samples, outer_iters)
    # best_mac: (H_high, d_L)

    outer_best_cost = outer_cost(best_mac.unsqueeze(0)).item()
    py_log.debug("  outer CEM done — best cost: %.4f", outer_best_cost)

    # ── Derive first subgoal ──────────────────────────────────────────────────
    z_sg = model._rollout_high(z_init, best_mac.unsqueeze(0))[:, 0].squeeze(0)  # (D,)

    # ── Inner CEM: optimise primitive actions to reach z_sg ──────────────────
    mu_act = torch.zeros(h_low, model.action_dim, device=device)
    std_act = torch.full((h_low, model.action_dim), 1.0, device=device)

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, h_low, action_dim)
        z_final = model._rollout_low(z_init, candidates)       # (S, D)
        return (z_final - z_sg.unsqueeze(0)).abs().sum(-1)     # (S,)

    best_act = cem(inner_cost, mu_act, std_act, inner_samples, inner_iters)
    # best_act: (h_low, action_dim)

    inner_best_cost = inner_cost(best_act.unsqueeze(0)).item()
    py_log.debug("  inner CEM done — best cost: %.4f", inner_best_cost)

    return best_act[0]   # first primitive action: (action_dim,)
