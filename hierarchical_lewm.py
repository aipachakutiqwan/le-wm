"""Hierarchical LeWM — models and stage-2 training.

New components (jepa.py and module.py are not modified):
  ActionEncoder       A_ψ  : action chunk  →  latent macro-action
  HighLevelPredictor  P^(2): AR transformer over waypoint latents
  HierarchicalLeWM        : wrapper with forward_low / forward_high + rollout helpers
  sample_waypoints        : HWM-style waypoint sampler
  train_hierarchical_lewm : stage-2 training driver

See hierarchical_plan.py for the two-level CEM-MPC planner.
"""

import torch
import torch.nn.functional as F
from torch import nn

from jepa import JEPA
from module import ARPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint sampler
# ──────────────────────────────────────────────────────────────────────────────


def sample_waypoints(T: int, N: int = 3, device=None) -> torch.Tensor:
    """N random interior indices plus the two fixed endpoints [0, T-1].

    Returns a sorted 1-D tensor of shape (N+2,).
    Falls back to a full arange when N >= T-1 (very short trajectories).
    """
    if N >= T - 1:
        return torch.arange(T, device=device)
    interior = torch.randperm(T - 2, device=device)[:N] + 1  # never 0 or T-1
    endpoints = torch.tensor([0, T - 1], device=device)
    return torch.cat([endpoints, interior]).sort().values


# ──────────────────────────────────────────────────────────────────────────────
# Action Encoder  A_ψ
# ──────────────────────────────────────────────────────────────────────────────


class ActionEncoder(nn.Module):
    """Compress a variable-length chunk of primitive actions into one latent macro-action.

    Architecture
    ------------
    1. Linear input projection  (action_dim → hidden_dim)
    2. Prepend a learnable [CLS] token
    3. Bidirectional TransformerEncoder with an optional padding mask
    4. Extract [CLS] output, apply LayerNorm
    5. Linear projection  (hidden_dim → latent_action_dim)

    Parameters
    ----------
    action_dim        : primitive action dimension (effective, after frameskip)
    hidden_dim        : internal transformer width
    latent_action_dim : d_L — output dimension of the macro-action
    depth             : number of TransformerEncoderLayers
    heads             : attention heads
    dropout           : dropout applied inside the encoder layers

    Inputs
    ------
    actions      : (B, L, action_dim)  — chunk, L may vary across calls
    padding_mask : (B, L) bool         — True = padding (ignored in attention)

    Returns
    -------
    (B, latent_action_dim)
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        latent_action_dim: int = 4,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.input_proj = nn.Linear(action_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm, consistent with LeWM ConditionalBlock style
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_action_dim)

    def forward(
        self,
        actions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = actions.size(0)
        x = self.input_proj(actions.float())           # (B, L, hidden_dim)

        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, hidden_dim)
        x = torch.cat([cls, x], dim=1)                 # (B, 1+L, hidden_dim)

        if padding_mask is not None:
            # CLS is never a padding position
            cls_mask = padding_mask.new_zeros(B, 1)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)   # (B, 1+L)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x[:, 0])     # CLS output → (B, hidden_dim)
        return self.proj(x)        # (B, latent_action_dim)


# ──────────────────────────────────────────────────────────────────────────────
# High-level Predictor  P^(2)
# ──────────────────────────────────────────────────────────────────────────────


class HighLevelPredictor(nn.Module):
    """Autoregressive transformer over waypoint latents conditioned on latent macro-actions.

    Mirrors the LeWM low-level predictor (ARPredictor + AdaLN-zero ConditionalBlocks)
    but receives latent macro-actions (d_L-dim) instead of primitive action embeddings.
    A linear layer mac_proj lifts d_L → embed_dim so the existing ARPredictor's internal
    cond_proj can then route it into the AdaLN blocks unchanged.

    Parameters
    ----------
    embed_dim         : latent space dimension (shared with the LeWM encoder)
    latent_action_dim : d_L — dimension of incoming macro-actions
    hidden_dim        : transformer width (defaults to embed_dim)
    num_frames        : positional-embedding buffer size; must be >= max waypoints
    depth / heads / mlp_dim / dim_head / dropout : match the low-level predictor defaults

    Inputs
    ------
    z   : (B, K, embed_dim)          — K waypoint latents (teacher-forced)
    mac : (B, K, latent_action_dim)  — macro-action conditioning per position

    Returns
    -------
    (B, K, embed_dim)  — predicted next-waypoint latent at every position
                         (causal: position k predicts waypoint k+1)
    """

    def __init__(
        self,
        embed_dim: int,
        latent_action_dim: int = 4,
        hidden_dim: int | None = None,
        num_frames: int = 8,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        # lift d_L → embed_dim so ARPredictor's cond_proj sees the expected width
        self.mac_proj = nn.Linear(latent_action_dim, embed_dim)

        self.predictor = ARPredictor(
            num_frames=num_frames,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor, mac: torch.Tensor) -> torch.Tensor:
        c = self.mac_proj(mac)        # (B, K, embed_dim)
        return self.predictor(z, c)   # (B, K, embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalLeWM
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalLeWM(nn.Module):
    """LeWM extended with HWM-style two-level hierarchical planning.

    Training
    --------
    Stage 1 — train the inner JEPA normally (see train.py / train_hierarchical_lewm).
    Stage 2 — freeze E and P^(1); jointly train A_ψ and P^(2) on the teacher-forcing
               waypoint loss L_tf = (1/N) Σ_k ‖ẑ_{t_{k+1}} − z_{t_{k+1}}‖_1.

    Planning
    --------
    See hierarchical_plan.plan() for the two-level CEM-MPC entry point.
    The rollout helpers below (_rollout_high, _rollout_low) are called from there.

    Parameters
    ----------
    jepa              : JEPA trained at stage 1
    embed_dim         : projected latent dimension (output of jepa.projector)
    action_dim        : effective primitive action dimension (after frameskip)
    latent_action_dim : d_L — HWM default 4 (sweep this first when tuning)
    n_waypoints       : N interior waypoints per trajectory (HWM default 3)
    history_size      : low-level context window size (must match stage-1 training)
    """

    def __init__(
        self,
        jepa: JEPA,
        embed_dim: int,
        action_dim: int,
        latent_action_dim: int = 4,
        n_waypoints: int = 3,
        history_size: int = 3,
        # high-level predictor knobs (mirror low-level defaults from train.py)
        high_depth: int = 6,
        high_heads: int = 16,
        high_mlp_dim: int = 2048,
        high_hidden_dim: int | None = None,
        high_num_frames: int = 8,
        # action encoder knobs
        action_enc_hidden: int = 128,
        action_enc_depth: int = 2,
        action_enc_heads: int = 4,
    ):
        super().__init__()
        self.jepa = jepa
        self.latent_action_dim = latent_action_dim
        self.n_waypoints = n_waypoints
        self.action_dim = action_dim
        self.history_size = history_size

        self.action_encoder_high = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_enc_hidden,
            latent_action_dim=latent_action_dim,
            depth=action_enc_depth,
            heads=action_enc_heads,
        )

        self.high_predictor = HighLevelPredictor(
            embed_dim=embed_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=high_hidden_dim or embed_dim,
            num_frames=high_num_frames,
            depth=high_depth,
            heads=high_heads,
            mlp_dim=high_mlp_dim,
        )

    # ── Stage-1 forward ──────────────────────────────────────────────────────

    def forward_low(self, obs: dict) -> dict:
        """Unchanged LeWM encode — use with the existing stage-1 loss in train.py."""
        return self.jepa.encode(obs)

    # ── Stage-2 forward ──────────────────────────────────────────────────────

    def forward_high(
        self,
        obs: dict,
        waypoint_idx: torch.Tensor,
        freeze_encoder: bool = True,
    ) -> dict:
        """Teacher-forcing loss on waypoint latents (stage-2 objective).

        L_tf = (1/N) Σ_k ‖ẑ_{t_{k+1}} − z_{t_{k+1}}‖_1

        where ẑ_{t_{k+1}} = P^(2)((l_{t_i}, z_{t_i})_{i≤k}) and
              l_k = A_ψ(a_{t_k : t_{k+1}}).

        Parameters
        ----------
        obs            : batch dict with 'pixels' (B,T,C,H,W) and 'action' (B,T,A)
        waypoint_idx   : (W,) sorted frame indices; W = N_interior + 2
        freeze_encoder : block gradients through E and P^(1) (recommended)

        Returns
        -------
        dict with 'loss' (scalar), 'high_pred_emb', 'high_target_emb'
        """
        pixels = obs["pixels"]                              # (B, T, C, H, W)
        actions = torch.nan_to_num(obs["action"], 0.0)     # (B, T, action_dim)

        grad_ctx = torch.no_grad() if freeze_encoder else torch.enable_grad()
        with grad_ctx:
            emb = self.jepa.encode({"pixels": pixels})["emb"]  # (B, T, embed_dim)

        W = len(waypoint_idx)
        wp_emb = emb[:, waypoint_idx]      # (B, W, embed_dim)

        # A_ψ: encode each inter-waypoint action chunk → one macro-action
        n_seg = W - 1
        macro_list = []
        for k in range(n_seg):
            s = int(waypoint_idx[k])
            e = int(waypoint_idx[k + 1])
            chunk = actions[:, s:e]                              # (B, chunk_len, A)
            macro_list.append(self.action_encoder_high(chunk))  # (B, d_L)

        macro_actions = torch.stack(macro_list, dim=1)   # (B, n_seg, d_L)

        # P^(2): causal AR prediction — position k predicts waypoint k+1
        pred_emb = self.high_predictor(wp_emb[:, :-1], macro_actions)  # (B, n_seg, D)
        target_emb = wp_emb[:, 1:].detach()                             # (B, n_seg, D)

        loss = F.l1_loss(pred_emb, target_emb)

        return {
            "loss": loss,
            "high_pred_emb": pred_emb,
            "high_target_emb": target_emb,
        }

    # ── Rollout helpers (called by hierarchical_plan.plan) ────────────────────

    @torch.no_grad()
    def _rollout_high(
        self,
        z_init: torch.Tensor,         # (D,) or (n, D)
        macro_actions: torch.Tensor,  # (n, H, d_L)
    ) -> torch.Tensor:                # (n, H, D)
        """Autoregressively roll out P^(2) and return all H subgoal latents."""
        n, H = macro_actions.shape[:2]
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0).expand(n, -1)

        z_seq = z_init.unsqueeze(1)   # (n, 1, D)
        subgoals = []
        for k in range(H):
            mac_so_far = macro_actions[:, : k + 1]                    # (n, k+1, d_L)
            pred = self.high_predictor(z_seq, mac_so_far)[:, -1:]     # (n, 1, D)
            subgoals.append(pred)
            z_seq = torch.cat([z_seq, pred], dim=1)                   # (n, k+2, D)

        return torch.cat(subgoals, dim=1)   # (n, H, D)

    @torch.no_grad()
    def _rollout_low(
        self,
        z_init: torch.Tensor,   # (D,) or (n, D)
        actions: torch.Tensor,  # (n, h, action_dim)
    ) -> torch.Tensor:          # (n, D)
        """Autoregressively roll out P^(1) for h primitive steps."""
        n, h = actions.shape[:2]
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0).expand(n, -1)

        z = z_init.unsqueeze(1)   # (n, 1, D)
        HS = self.history_size
        for t in range(h):
            act_hist = actions[:, max(0, t + 1 - HS) : t + 1]   # (n, hs, A)
            act_emb = self.jepa.action_encoder(act_hist)          # (n, hs, A_emb)
            z_hist = z[:, -HS:]                                   # (n, hs, D)
            pred = self.jepa.predict(z_hist, act_emb)[:, -1:]    # (n, 1, D)
            z = torch.cat([z, pred], dim=1)                       # (n, t+2, D)

        return z[:, -1]   # (n, D)


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 training driver
# ──────────────────────────────────────────────────────────────────────────────


def train_hierarchical_lewm(
    model: HierarchicalLeWM,
    dataloader,
    n_waypoints: int = 3,
    lr: float = 1e-4,
    n_epochs: int = 10,
    device: str = "cuda",
    freeze_encoder: bool = True,
) -> HierarchicalLeWM:
    """Jointly optimise A_ψ and P^(2) on L_tf (stage 2).

    Stage 1 (training jepa) is handled by the existing train.py.
    Load the stage-1 checkpoint into a JEPA instance, wrap it in
    HierarchicalLeWM, then call this function.

    Parameters
    ----------
    model          : HierarchicalLeWM with a stage-1-trained inner jepa
    dataloader     : yields batches with 'pixels' and 'action' keys
    n_waypoints    : N interior waypoints per trajectory (HWM default 3)
    lr             : AdamW learning rate for stage-2 parameters
    n_epochs       : number of stage-2 epochs
    device         : target device string
    freeze_encoder : if True, no gradients flow through E or P^(1)
    """
    model = model.to(device)

    # only the two new modules are optimised in stage 2
    stage2_params = (
        list(model.action_encoder_high.parameters())
        + list(model.high_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(stage2_params, lr=lr)

    if freeze_encoder:
        for p in model.jepa.parameters():
            p.requires_grad_(False)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            T = batch["pixels"].shape[1]
            wp_idx = sample_waypoints(T, N=n_waypoints, device=device)

            out = model.forward_high(batch, wp_idx, freeze_encoder=freeze_encoder)

            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            epoch_loss += out["loss"].item()

        print(
            f"epoch {epoch + 1}/{n_epochs}  stage-2 L_tf: "
            f"{epoch_loss / len(dataloader):.5f}"
        )

    return model
