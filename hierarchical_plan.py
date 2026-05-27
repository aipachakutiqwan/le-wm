"""Hierarchical LeWM — two-level CEM-MPC planner.

Entry point: plan()
Utility:     cem()

Both operate on a trained HierarchicalLeWM from hierarchical_lewm.py.
The rollout helpers (_rollout_high, _rollout_low) live on the model because
they directly use its weights; planning logic that is independent of model
parameters lives here.
"""

import torch

from hierarchical_lewm import HierarchicalLeWM


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
    var_ema: float = 0.0,
) -> torch.Tensor:
    """Diagonal-Gaussian Cross-Entropy Method with optional variance EMA.

    Parameters
    ----------
    cost_fn    : callable (n_samples, *shape) -> (n_samples,) — lower is better
    mu         : (*shape,) initial mean
    std        : (*shape,) initial std
    n_samples  : number of candidates sampled per iteration
    n_iters    : number of CEM iterations
    elite_frac : fraction of candidates kept as elites
    var_ema    : exponential moving average coefficient for variance update.
                 0.0 = no EMA (recompute from elites each iteration).
                 >0 = var = var_ema * var_prev + (1-var_ema) * elite_var.
                 Prevents premature variance collapse; paper (HWM) uses 0.9 (outer)
                 and 0.8 (inner) for Push-T, 0.65/0.25 for Franka.

    Returns
    -------
    (*shape,) optimised mean
    """
    n_elites = max(1, int(n_samples * elite_frac))
    var = std ** 2
    for _ in range(n_iters):
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        candidates = mu.unsqueeze(0) + std.unsqueeze(0) * eps      # (S, *shape)
        costs = cost_fn(candidates)                                  # (S,)
        elite_idx = costs.argsort()[:n_elites]
        elites = candidates[elite_idx]
        mu = elites.mean(0)
        elite_var = elites.var(0, unbiased=False).clamp(min=1e-8)
        var = var_ema * var + (1.0 - var_ema) * elite_var           # EMA update
        std = var.sqrt().clamp(min=1e-4)
    return mu


# ──────────────────────────────────────────────────────────────────────────────
# MPPI
# ──────────────────────────────────────────────────────────────────────────────


def mppi(
    cost_fn,
    mu: torch.Tensor,
    sigma: float,
    n_samples: int = 2000,
    n_iters: int = 5,
    lam: float = 1.0,
) -> torch.Tensor:
    """Model Predictive Path Integral optimiser (paper algorithm for navigation).

    Unlike CEM (hard elite threshold), MPPI soft-weights ALL samples via
    exp(−cost/λ).  Near-miss trajectories (e.g. almost through a doorway)
    still contribute positively, giving smoother gradient signal near bottlenecks.

    Parameters
    ----------
    cost_fn   : callable (n_samples, *shape) -> (n_samples,) — lower is better
    mu        : (*shape,) initial mean
    sigma     : fixed noise std — exploration scale (paper maze: 10 for high-level)
    n_samples : trajectories per iteration (paper maze: 2000–4000)
    n_iters   : refinement iterations
    lam       : temperature — lower = greedier; calibrate to cost scale
                (paper maze: 0.0025 for their latent space; start with ~1.0 here)

    Returns
    -------
    (*shape,) optimised mean
    """
    for _ in range(n_iters):
        noise = torch.randn(n_samples, *mu.shape, device=mu.device) * sigma
        candidates = mu.unsqueeze(0) + noise                         # (S, *shape)
        costs = cost_fn(candidates)                                   # (S,)
        beta = costs.min()
        # Normalise by std so λ is scale-invariant: e^{-1/λ} ≈ 0.37 at 1σ above best.
        scale = costs.std().clamp(min=1e-6)
        weights = torch.exp(-(costs - beta) / (scale * lam))         # (S,)
        weights = weights / (weights.sum() + 1e-8)
        mu = (weights.view(-1, *([1] * mu.dim())) * candidates).sum(0)
    return mu


# ──────────────────────────────────────────────────────────────────────────────
# Two-level CEM-MPC / MPPI
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def plan(
    model: HierarchicalLeWM,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    H_high: int = 3,
    h_low: int = 5,
    outer_samples: int = 2000,
    inner_samples: int = 1000,
    outer_iters: int = 10,
    inner_iters: int = 10,
    outer_var_ema: float = 0.9,
    inner_var_ema: float = 0.8,
    use_mppi: bool = True,
    mppi_sigma_outer: float = 1.0,
    mppi_sigma_inner: float = 0.5,
    mppi_lam: float = 1.0,
    inner_goal_alpha: float = 0.0,
    prev_mac: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-level MPPI/CEM-MPC matching the HWM paper (arXiv 2604.03208).

    Outer  — E_2 = ‖z_g − P^(2)(l̂_{1:H}; z_1)‖_1  (paper eq., final subgoal)
    Inner  — E_1 = ‖z̃_1 − P^(1)(â_{1:h}; z_1)‖_1  (paper eq.)

    Parameters
    ----------
    model            : trained HierarchicalLeWM
    z_init           : (D,) current latent state
    z_goal           : (D,) goal latent state
    H_high           : high-level horizon
    h_low            : low-level horizon (primitive steps per segment)
    outer_samples    : sample count for outer loop
    inner_samples    : sample count for inner loop
    outer_iters      : refinement iterations for outer loop
    inner_iters      : refinement iterations for inner loop
    outer_var_ema    : CEM variance EMA for outer loop (ignored when use_mppi=True)
    inner_var_ema    : CEM variance EMA for inner loop (ignored when use_mppi=True)
    use_mppi         : use MPPI (paper algorithm for navigation) instead of CEM
    mppi_sigma_outer : MPPI noise std for outer loop (paper maze high-level: 10)
    mppi_sigma_inner : MPPI noise std for inner loop (paper maze low-level: 5)
    mppi_lam         : MPPI temperature — calibrate to cost scale (paper: 0.0025)
    prev_mac         : (H_high, d_L) warm-start from previous plan's best macro-action.
                       Shift by one: warm_mac[k] = prev_mac[k+1], warm_mac[-1] = 0.
                       Pass the second return value of the previous plan() call.

    Returns
    -------
    best_act[0]  : (action_dim,) first effective action to execute
    best_mac     : (H_high, d_L) optimal macro-action for warm-starting next call
    """
    device = z_init.device
    d_L = model.latent_action_dim

    # ── Outer loop: optimise macro-action sequence ────────────────────────────
    # Warm start: shift previous macro-action by one step so the plan is consistent
    # across replanning calls (common MPC technique — not in paper but standard practice).
    if prev_mac is not None:
        mu_mac = torch.cat([prev_mac[1:], torch.zeros(1, d_L, device=device)], dim=0)
    else:
        mu_mac = torch.zeros(H_high, d_L, device=device)

    def outer_cost(candidates: torch.Tensor) -> torch.Tensor:
        subgoals = model._rollout_high(z_init, candidates)     # (S, H_high, D)
        z_last = subgoals[:, -1]                               # (S, D)
        return (z_last - z_goal.unsqueeze(0)).abs().sum(-1)    # (S,)

    if use_mppi:
        best_mac = mppi(outer_cost, mu_mac, mppi_sigma_outer,
                        outer_samples, outer_iters, mppi_lam)
    else:
        std_mac = torch.ones(H_high, d_L, device=device)
        best_mac = cem(outer_cost, mu_mac, std_mac, outer_samples, outer_iters,
                       var_ema=outer_var_ema)

    # ── Derive first subgoal ──────────────────────────────────────────────────
    z_sg = model._rollout_high(z_init, best_mac.unsqueeze(0))[:, 0].squeeze(0)  # (D,)

    # ── Inner loop: optimise primitive actions to reach z_sg ─────────────────
    mu_act = torch.zeros(h_low, model.action_dim, device=device)

    def inner_cost(candidates: torch.Tensor) -> torch.Tensor:
        z_final = model._rollout_low(z_init, candidates)                           # (S, D)
        cost = (z_final - z_sg.unsqueeze(0)).abs().sum(-1)                         # (S,)
        if inner_goal_alpha > 0.0:
            cost = cost + inner_goal_alpha * (z_final - z_goal.unsqueeze(0)).abs().sum(-1)
        return cost

    if use_mppi:
        best_act = mppi(inner_cost, mu_act, mppi_sigma_inner,
                        inner_samples, inner_iters, mppi_lam)
    else:
        std_act = torch.full((h_low, model.action_dim), 0.5, device=device)
        best_act = cem(inner_cost, mu_act, std_act, inner_samples, inner_iters,
                       var_ema=inner_var_ema)

    return best_act[0], best_mac
