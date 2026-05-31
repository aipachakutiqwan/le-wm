"""Hierarchical LeWM — models and stage-2 training.

New components (jepa.py and module.py are not modified):
  ActionEncoder      A_ψ  : action chunk  →  latent macro-action
  HighLevelPredictor P^(2): AR transformer over waypoint latents
  HierarchicalLeWM       : wrapper with forward_high + rollout helpers
  sample_waypoints       : HWM-style waypoint sampler
  HierarchicalLeWMModule : Lightning module for stage-2 training (1 or N GPUs)

See hierarchical_plan.py for the two-level CEM-MPC planner.
"""

import logging
import time

import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from jepa import JEPA
from module import ARPredictor, SIGReg

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint sampler
# ──────────────────────────────────────────────────────────────────────────────


def sample_waypoints(T: int, N: int = 3, device=None) -> torch.Tensor:
    """N+2 evenly-spaced waypoint indices across [0, T-1] (PLDM-style fixed stride).

    Returns a sorted 1-D tensor of shape (N+2,) — endpoints always included.
    Falls back to a full arange when N >= T-1.
    """
    if N >= T - 1:
        return torch.arange(T, device=device)
    return torch.linspace(0, T - 1, N + 2, device=device).round().long()


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
    Stage 1 — train the inner JEPA normally (see train.py).
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
        lambda_sigreg: float = 0.0,
        lambda_mac_reg: float = 0.0,
        # high-level predictor knobs (mirror low-level defaults from train.py)
        high_depth: int = 6,
        high_heads: int = 16,
        high_mlp_dim: int = 2048,
        high_hidden_dim: int | None = None,
        high_num_frames: int = 8,
        high_dropout: float = 0.0,
        # action encoder knobs
        action_enc_hidden: int = 128,
        action_enc_depth: int = 2,
        action_enc_heads: int = 4,
        action_enc_dropout: float = 0.0,
    ):
        super().__init__()
        self.jepa = jepa
        self.latent_action_dim = latent_action_dim
        self.n_waypoints = n_waypoints
        self.action_dim = action_dim
        self.history_size = history_size
        self.lambda_sigreg = lambda_sigreg
        self.lambda_mac_reg = lambda_mac_reg
        if lambda_sigreg > 0.0:
            self.sigreg = SIGReg()

        self.action_encoder_high = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_enc_hidden,
            latent_action_dim=latent_action_dim,
            depth=action_enc_depth,
            heads=action_enc_heads,
            dropout=action_enc_dropout,
        )

        self.high_predictor = HighLevelPredictor(
            embed_dim=embed_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=high_hidden_dim or embed_dim,
            num_frames=high_num_frames,
            depth=high_depth,
            heads=high_heads,
            mlp_dim=high_mlp_dim,
            dropout=high_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning calls model.train() at the start of every epoch.
        # JEPA must stay in eval mode so its dropout and BN running stats
        # are not corrupted during stage-2 training.
        self.jepa.eval()
        return self

    # ── Stage-2 forward ──────────────────────────────────────────────────────

    def forward_high(
        self,
        obs: dict,
        waypoint_idx: torch.Tensor,
        freeze_encoder: bool = True,
        ss_prob: float = 0.0,
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
        ss_prob        : scheduled-sampling probability — at each AR step, replace the
                         ground-truth input with the model's own previous prediction with
                         this probability.  0.0 = pure teacher forcing; ramp toward
                         ss_max_prob over training to close the train/eval gap.

        Returns
        -------
        dict with 'loss' (scalar), 'loss_tf', 'loss_reg', 'high_pred_emb', 'high_target_emb'
        """
        actions = torch.nan_to_num(obs["action"], 0.0)     # (B, T, action_dim)
        B, _, A = actions.shape

        if "emb" in obs:
            emb = obs["emb"]                                    # (B, T, embed_dim) — pre-cached
        else:
            grad_ctx = torch.no_grad() if freeze_encoder else torch.enable_grad()
            with grad_ctx:
                emb = self.jepa.encode({"pixels": obs["pixels"]})["emb"]  # (B, T, embed_dim)

        W = len(waypoint_idx)
        wp_emb = emb[:, waypoint_idx]      # (B, W, embed_dim)

        # A_ψ: encode all inter-waypoint action chunks in one batched forward pass.
        # Chunks may differ in length by ±1 step (linspace rounding), so we pad to
        # max_len and pass a mask; ActionEncoder ignores masked positions via
        # src_key_padding_mask.
        n_seg = W - 1
        seg_slices = [(int(waypoint_idx[k]), int(waypoint_idx[k + 1])) for k in range(n_seg)]
        chunk_lens = [e - s for s, e in seg_slices]
        max_len = max(chunk_lens)

        padded = actions.new_zeros(B, n_seg, max_len, A)       # (B, n_seg, max_len, A)
        mask = torch.ones(B, n_seg, max_len, dtype=torch.bool, device=actions.device)  # True = pad

        for k, ((s, e), clen) in enumerate(zip(seg_slices, chunk_lens)):
            padded[:, k, :clen] = actions[:, s:e]
            mask[:, k, :clen] = False

        macro_flat = self.action_encoder_high(
            padded.view(B * n_seg, max_len, A),
            padding_mask=mask.view(B * n_seg, max_len),
        )                                                        # (B*n_seg, d_L)
        macro_actions = macro_flat.view(B, n_seg, -1)           # (B, n_seg, d_L)

        # P^(2): causal AR prediction — position k predicts waypoint k+1.
        # With ss_prob > 0 (scheduled sampling), each step feeds back the model's
        # own prediction instead of ground truth with probability ss_prob.
        if ss_prob <= 0.0:
            pred_emb = self.high_predictor(wp_emb[:, :-1], macro_actions)  # (B, n_seg, D)
        else:
            preds = []
            z_seq = wp_emb[:, :1]                                    # (B, 1, D) — ground truth z_0
            for k in range(n_seg):
                pred_k = self.high_predictor(
                    z_seq, macro_actions[:, : k + 1]
                )[:, -1:]                                             # (B, 1, D)
                preds.append(pred_k)
                if k < n_seg - 1:
                    # Per-element Bernoulli: each sample independently decides whether
                    # to feed back the model's own prediction or the ground-truth waypoint.
                    use_pred = (torch.rand(B, 1, 1, device=wp_emb.device) < ss_prob)
                    next_z = torch.where(use_pred, pred_k.detach(), wp_emb[:, k + 1 : k + 2])
                    z_seq = torch.cat([z_seq, next_z], dim=1)
            pred_emb = torch.cat(preds, dim=1)                       # (B, n_seg, D)

        target_emb = wp_emb[:, 1:].detach()                         # (B, n_seg, D)

        loss_tf = F.l1_loss(pred_emb, target_emb)

        if self.lambda_sigreg > 0.0:
            loss_reg = self.sigreg(pred_emb.permute(1, 0, 2))       # SIGReg expects (T, B, D)
        else:
            loss_reg = pred_emb.new_zeros(1).squeeze()

        # Keep A_ψ outputs near N(0, I) so CEM prior (also N(0,1)) is well-matched.
        if self.lambda_mac_reg > 0.0:
            mac_reg = (macro_actions.pow(2).mean()
                       + (macro_actions.std(dim=(0, 1)) - 1).pow(2).mean())
        else:
            mac_reg = macro_actions.new_zeros(1).squeeze()

        loss = (loss_tf
                + self.lambda_sigreg * loss_reg
                + self.lambda_mac_reg * mac_reg)

        return {
            "loss": loss,
            "loss_tf": loss_tf,
            "loss_reg": loss_reg,
            "loss_mac_reg": mac_reg,
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
# Lightning Module — stage-2 training (1 GPU or multi-GPU via DDP)
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalLeWMModule(pl.LightningModule):
    """Stage-2 Lightning module: trains A_ψ and P^(2) while keeping E and P^(1) frozen.

    Works transparently on 1 GPU or N GPUs (DDP) — set ``devices: auto`` in the
    Trainer config.  The inner HierarchicalLeWM is accessible via ``self.model``
    after training.

    Parameters
    ----------
    model          : HierarchicalLeWM with a stage-1-trained inner jepa
    n_waypoints    : N interior waypoints per trajectory
    lr             : AdamW learning rate for stage-2 parameters
    freeze_encoder : keep E and P^(1) frozen (recommended)
    """

    def __init__(
        self,
        model: HierarchicalLeWM,
        n_waypoints: int,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        freeze_encoder: bool = True,
        compile_model: bool = True,
        ss_max_prob: float = 0.0,
        ss_ramp_epochs: int = 30,
    ):
        super().__init__()
        self.model = model
        self.n_waypoints = n_waypoints
        self.lr = lr
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.compile_model = compile_model
        self.ss_max_prob = ss_max_prob
        self.ss_ramp_epochs = ss_ramp_epochs

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return
        if self.freeze_encoder:
            for p in self.model.jepa.parameters():
                p.requires_grad_(False)
        if self.compile_model:
            self.model.action_encoder_high = torch.compile(self.model.action_encoder_high)
            self.model.high_predictor = torch.compile(self.model.high_predictor)

    def training_step(self, batch, batch_idx):
        t0 = time.perf_counter()
        obs = batch["emb"] if "emb" in batch else batch["pixels"]
        T = obs.shape[1]
        wp_idx = sample_waypoints(T, N=self.n_waypoints, device=self.device)
        ss_prob = 0.0
        if self.ss_max_prob > 0.0 and self.ss_ramp_epochs > 0:
            ss_prob = min(self.ss_max_prob, self.ss_max_prob * self.current_epoch / self.ss_ramp_epochs)
        out = self.model.forward_high(batch, wp_idx, freeze_encoder=self.freeze_encoder, ss_prob=ss_prob)
        self.log("train/loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_tf", out["loss_tf"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_reg", out["loss_reg"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_mac_reg", out["loss_mac_reg"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ss_prob", ss_prob, on_step=False, on_epoch=True, sync_dist=False)
        log.debug("step %d — forward_ms=%.1f  loss=%.5f",
                  batch_idx, (time.perf_counter() - t0) * 1e3, out["loss"].item())
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        obs = batch["emb"] if "emb" in batch else batch["pixels"]
        T = obs.shape[1]
        wp_idx = sample_waypoints(T, N=self.n_waypoints, device=self.device)
        # Always pure teacher forcing for validation (matches AR rollout at eval with ss_prob=0).
        out = self.model.forward_high(batch, wp_idx, freeze_encoder=True, ss_prob=0.0)
        self.log("val/loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss_tf", out["loss_tf"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_reg", out["loss_reg"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_mac_reg", out["loss_mac_reg"], on_step=False, on_epoch=True, sync_dist=True)
        return out["loss"]

    def configure_optimizers(self):
        stage2_params = (
            list(self.model.action_encoder_high.parameters())
            + list(self.model.high_predictor.parameters())
        )
        # Linear scaling rule: effective batch = batch_size × world_size × accumulate_grad_batches.
        accum = self.trainer.accumulate_grad_batches
        effective_lr = self.lr * self.trainer.world_size * accum
        log.info(
            "world_size=%d  accum=%d  base_lr=%.2e  effective_lr=%.2e",
            self.trainer.world_size, accum, self.lr, effective_lr,
        )
        optimizer = torch.optim.AdamW(stage2_params, lr=effective_lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=effective_lr,
            total_steps=total_steps,
            pct_start=0.05,         # 5% warmup
            div_factor=25.0,        # initial_lr = max_lr / 25
            final_div_factor=1e4,   # final_lr = initial_lr / 1e4
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
