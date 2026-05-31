"""Hierarchical LeWM — two-level CEM-MPC planner.

Entry points : plan()         — single-environment (D,) latents
               plan_batched() — E environments in parallel on one device
Utilities    : cem(), cem_batched()

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
    eps = torch.empty(n_samples, *mu.shape, device=mu.device)
    for _ in range(n_iters):
        eps.normal_()
        candidates = mu.unsqueeze(0) + std.unsqueeze(0) * eps   # (S, *shape)
        costs = cost_fn(candidates)                              # (S,)
        elite_idx = torch.topk(costs, n_elites, largest=False).indices
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
    t0 = time.perf_counter()

    # ── Outer CEM: optimise macro-action sequence ─────────────────────────────
    mu_mac = torch.zeros(H_high, d_L, device=device)
    std_mac = torch.ones(H_high, d_L, device=device)

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, H_high, d_L)
        subgoals = model._rollout_high(z_init, candidates)     # (S, H_high, D)
        # Linear weights across subgoals: later waypoints penalised more,
        # but all contribute — encourages progressive approach to the goal.
        w = torch.linspace(1.0 / H_high, 1.0, H_high, device=device)
        w = w / w.sum()
        dists = (subgoals - z_goal.unsqueeze(0).unsqueeze(1)).abs().sum(-1)  # (S, H)
        return (dists * w.unsqueeze(0)).sum(-1)                # (S,)

    t_outer = time.perf_counter()
    best_mac = cem(outer_cost, mu_mac, std_mac, outer_samples, outer_iters)
    # best_mac: (H_high, d_L)

    # Single rollout: reuse subgoals for both the debug cost and z_sg.
    best_subgoals = model._rollout_high(z_init, best_mac.unsqueeze(0))  # (1, H_high, D)
    outer_cost_val = (best_subgoals[0, -1] - z_goal).abs().sum().item()
    py_log.info("outer CEM — best_cost=%.4f  ms=%.1f", outer_cost_val, (time.perf_counter() - t_outer) * 1e3)

    # ── Derive first subgoal ──────────────────────────────────────────────────
    z_sg = best_subgoals[0, 0]  # (D,)

    # ── Inner CEM: optimise primitive actions to reach z_sg ──────────────────
    mu_act = torch.zeros(h_low, model.action_dim, device=device)
    std_act = torch.full((h_low, model.action_dim), 0.5, device=device)

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (S, h_low, action_dim)
        z_final = model._rollout_low(z_init, candidates)       # (S, D)
        return (z_final - z_sg.unsqueeze(0)).abs().sum(-1)     # (S,)

    t_inner = time.perf_counter()
    best_act = cem(inner_cost, mu_act, std_act, inner_samples, inner_iters)
    # best_act: (h_low, action_dim)

    if py_log.isEnabledFor(logging.DEBUG):
        inner_cost_val = inner_cost(best_act.unsqueeze(0)).item()
        py_log.debug("inner CEM — best_cost=%.4f", inner_cost_val)
    total_s = time.perf_counter() - t0
    py_log.info("inner CEM — ms=%.1f  total=%.1fs (%.2f min)",
                (time.perf_counter() - t_inner) * 1e3, total_s, total_s / 60)

    return best_act[0]   # first primitive action: (action_dim,)


# ──────────────────────────────────────────────────────────────────────────────
# Batched CEM  (E independent problems solved simultaneously)
# ──────────────────────────────────────────────────────────────────────────────


def cem_batched(
    cost_fn,
    mu: torch.Tensor,
    std: torch.Tensor,
    n_samples: int = 512,
    n_iters: int = 5,
    elite_frac: float = 0.1,
) -> torch.Tensor:
    """CEM for E independent problems solved simultaneously.

    Parameters
    ----------
    cost_fn    : callable (E, S, *shape) -> (E, S) — lower is better
    mu         : (E, *shape) initial means
    std        : (E, *shape) initial stds
    n_samples  : candidates per iteration per environment
    n_iters    : CEM iterations
    elite_frac : fraction kept as elites

    Returns
    -------
    (E, *shape) optimised means
    """
    E = mu.shape[0]
    n_elites = max(1, int(n_samples * elite_frac))
    e_idx = torch.arange(E, device=mu.device)
    eps = torch.empty(E, n_samples, *mu.shape[1:], device=mu.device)
    for _ in range(n_iters):
        eps.normal_()
        candidates = mu.unsqueeze(1) + std.unsqueeze(1) * eps        # (E, S, *shape)
        costs = cost_fn(candidates)                                   # (E, S)
        elite_idx = torch.topk(costs, n_elites, dim=1, largest=False).indices  # (E, n_elites)
        elites = candidates[e_idx.unsqueeze(1), elite_idx]            # (E, n_elites, *shape)
        mu = elites.mean(dim=1)
        std = elites.std(dim=1).clamp(min=0.1)
    return mu


# ──────────────────────────────────────────────────────────────────────────────
# Batched two-level CEM-MPC
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
    warmstart: dict | None = None,
) -> torch.Tensor:
    """Two-level CEM-MPC for E environments solved in parallel on one device.

    Equivalent to calling plan() E times but fuses all rollouts into single
    batched forward passes, giving a large GPU utilisation improvement.

    Parameters
    ----------
    model     : trained HierarchicalLeWM
    z_init    : (E, D) current latent states
    z_goal    : (E, D) goal latent states
    warmstart : mutable dict optionally carrying ``"mu_mac": (E, H_high, d_L)``
                from the previous call.  When present the outer CEM is
                initialised from that tensor (std halved).  Updated in-place
                with the new best_mac after each call so callers only need to
                pass the same dict every time.
    (remaining args identical to plan())

    Returns
    -------
    (E, action_dim) — first primitive action for each environment
    """
    E, D = z_init.shape
    device = z_init.device
    d_L = model.latent_action_dim
    t0 = time.perf_counter()

    # ── Outer CEM ──────────────────────────────────────────────────────────────
    # Warm-start: reuse the previous best macro-action sequence as the initial
    # mean.  The std is halved because the solution should be nearby.
    if warmstart is not None and "mu_mac" in warmstart:
        mu_mac = warmstart["mu_mac"].to(device)
        std_mac = torch.full((E, H_high, d_L), 0.5, device=device)
    else:
        mu_mac = torch.zeros(E, H_high, d_L, device=device)
        std_mac = torch.ones(E, H_high, d_L, device=device)

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (E, S, H_high, d_L)
        S = candidates.shape[1]
        mac_flat = candidates.reshape(E * S, H_high, d_L)
        z_flat = z_init.unsqueeze(1).expand(E, S, D).reshape(E * S, D)
        subgoals = model._rollout_high(z_flat, mac_flat)              # (E*S, H_high, D)
        subgoals = subgoals.reshape(E, S, H_high, D)                 # (E, S, H, D)
        w = torch.linspace(1.0 / H_high, 1.0, H_high, device=device)
        w = w / w.sum()
        dists = (subgoals - z_goal.unsqueeze(1).unsqueeze(2)).abs().sum(-1)  # (E, S, H)
        return (dists * w.unsqueeze(0).unsqueeze(0)).sum(-1)          # (E, S)

    t_outer = time.perf_counter()
    best_mac = cem_batched(outer_cost, mu_mac, std_mac, outer_samples, outer_iters)

    # Persist best_mac for the next call's warm-start (CPU to avoid GPU memory leak).
    if warmstart is not None:
        warmstart["mu_mac"] = best_mac.cpu()

    # Single rollout: reuse subgoals for both the debug cost and z_sg.
    best_subgoals = model._rollout_high(z_init, best_mac)             # (E, H_high, D)
    outer_cost_val = (best_subgoals[:, -1] - z_goal).abs().sum(-1).mean().item()
    py_log.info("outer CEM — mean_best_cost=%.4f  E=%d  ms=%.1f",
                outer_cost_val, E, (time.perf_counter() - t_outer) * 1e3)

    # ── Derive first subgoal per environment ───────────────────────────────────
    z_sg = best_subgoals[:, 0]                                        # (E, D)

    # ── Inner CEM ──────────────────────────────────────────────────────────────
    mu_act = torch.zeros(E, h_low, model.action_dim, device=device)
    std_act = torch.full((E, h_low, model.action_dim), 0.5, device=device)

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        # candidates: (E, S, h_low, action_dim)
        S = candidates.shape[1]
        act_flat = candidates.reshape(E * S, h_low, model.action_dim)
        z_flat = z_init.unsqueeze(1).expand(E, S, D).reshape(E * S, D)
        z_final = model._rollout_low(z_flat, act_flat).reshape(E, S, D)  # (E, S, D)
        return (z_final - z_sg.unsqueeze(1)).abs().sum(-1)               # (E, S)

    t_inner = time.perf_counter()
    best_act = cem_batched(inner_cost, mu_act, std_act, inner_samples, inner_iters)
    if py_log.isEnabledFor(logging.DEBUG):
        py_log.debug("inner CEM — mean_best_cost=%.4f",
                     inner_cost(best_act.unsqueeze(1)).mean().item())
    total_s = time.perf_counter() - t0
    py_log.info("inner CEM — ms=%.1f  total=%.1fs (%.2f min)",
                (time.perf_counter() - t_inner) * 1e3, total_s, total_s / 60)

    return best_act[:, 0]   # (E, action_dim)
