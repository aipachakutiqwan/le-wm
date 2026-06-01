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
from waypoint_sampler import sample_waypoints_fixed_stride


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
    dropout           : dropout inside A_ψ and P^(2) transformers (0 = off)
    """

    def __init__(
        self,
        jepa: JEPA,
        embed_dim: int,
        action_dim: int,
        latent_action_dim: int = 4,
        n_waypoints: int = 3,
        history_size: int = 3,
        lambda_var: float = 0.0,
        lambda_kl: float = 0.0,
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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.jepa = jepa
        self.latent_action_dim = latent_action_dim
        self.n_waypoints = n_waypoints
        self.action_dim = action_dim
        self.history_size = history_size
        self.lambda_var = lambda_var
        self.lambda_kl = lambda_kl

        self.action_encoder_high = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_enc_hidden,
            latent_action_dim=latent_action_dim,
            depth=action_enc_depth,
            heads=action_enc_heads,
            dropout=dropout,
        )

        self.high_predictor = HighLevelPredictor(
            embed_dim=embed_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=high_hidden_dim or embed_dim,
            num_frames=high_num_frames,
            depth=high_depth,
            heads=high_heads,
            mlp_dim=high_mlp_dim,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep JEPA in eval so its dropout/BN stats are unaffected by stage-2.
        self.jepa.eval()
        return self

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
        teacher_forcing_prob: float = 1.0,
    ) -> dict:
        """Stage-2 objective on waypoint latents.

        L = (1/N) Σ_k ‖ẑ_{t_{k+1}} − z_{t_{k+1}}‖²

        where ẑ_{t_{k+1}} = P^(2)((l_{t_i}, z_{t_i})_{i≤k}) and
              l_k = A_ψ(a_{t_k : t_{k+1}}).

        Parameters
        ----------
        obs                  : batch dict with 'pixels' (B,T,C,H,W) and 'action' (B,T,A)
        waypoint_idx         : (W,) sorted frame indices; W = N_interior + 2
        freeze_encoder       : block gradients through E and P^(1) (recommended)
        teacher_forcing_prob : 1.0 → one-step teacher forcing (each prediction conditions
                               on the TRUE previous waypoints; fast parallel pass). < 1.0 →
                               scheduled-sampling rollout: the model's own prediction is fed
                               back as the next-step input with prob (1 - p), so it is trained
                               on its autoregressive distribution and penalised for compounding
                               error. Gradients flow through the fed-back predictions (BPTT).

        Returns
        -------
        dict with 'loss' (scalar), 'high_pred_emb', 'high_target_emb'
        """
        actions = torch.nan_to_num(obs["action"], 0.0)     # (B, T, action_dim)

        if "emb" in obs:
            # Fast path: pre-computed embeddings from the cache (no ViT forward pass).
            emb = obs["emb"]                                # (B, T, embed_dim)
        else:
            pixels = obs["pixels"]                          # (B, T, C, H, W)
            grad_ctx = torch.no_grad() if freeze_encoder else torch.enable_grad()
            with grad_ctx:
                emb = self.jepa.encode({"pixels": pixels})["emb"]

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
        if teacher_forcing_prob >= 1.0:
            # Fast parallel pass: every position conditions on the TRUE history.
            pred_emb = self.high_predictor(wp_emb[:, :-1], macro_actions)  # (B, n_seg, D)
        else:
            # Scheduled-sampling rollout: build the history step-by-step, feeding back the
            # model's own prediction (with prob 1 - teacher_forcing_prob) instead of truth.
            preds = []
            hist = wp_emb[:, :1]                              # (B, 1, D) — true z_0
            for k in range(n_seg):
                pred_next = self.high_predictor(hist, macro_actions[:, :k + 1])[:, -1:]
                preds.append(pred_next)                       # predict z_{k+1}
                if k < n_seg - 1:
                    use_true = (
                        torch.rand(pixels.shape[0], 1, 1, device=hist.device)
                        < teacher_forcing_prob
                    )
                    next_in = torch.where(use_true, wp_emb[:, k + 1:k + 2], pred_next)
                    hist = torch.cat([hist, next_in], dim=1)
            pred_emb = torch.cat(preds, dim=1)                # (B, n_seg, D)

        target_emb = wp_emb[:, 1:].detach()                             # (B, n_seg, D)

        loss_pred = F.mse_loss(pred_emb, target_emb)

        if self.lambda_var > 0.0:
            # Variance penalty on macro-actions — prevents A_ψ from collapsing to a
            # constant embedding regardless of the input action chunk.  The hinge form
            # relu(γ − std) only penalises dimensions whose batch std falls below γ=1;
            # once std > 1 the gradient is zero, so the loss does not fight natural
            # spread in the data.  Technique adapted from the VICReg variance term
            # (Bardes et al., 2022 — https://arxiv.org/abs/2105.04906, Eq. 2).
            flat = macro_actions.reshape(-1, self.latent_action_dim)
            loss_var = F.relu(1.0 - flat.std(dim=0)).mean()
        else:
            loss_var = pred_emb.new_zeros(1).squeeze()

        if self.lambda_kl > 0.0:
            # KL-to-N(0,I) moment-matching on A_ψ outputs: push mean->0 and std->1
            # per dim so the macro-action distribution matches the planner's CEM
            # sampling prior. Stronger than the one-sided variance hinge above
            # (also constrains the mean; can be used in place of lambda_var).
            loss_kl = (
                macro_actions.pow(2).mean()
                + (macro_actions.std(dim=(0, 1)) - 1).pow(2).mean()
            )
        else:
            loss_kl = pred_emb.new_zeros(1).squeeze()

        loss = loss_pred + self.lambda_var * loss_var + self.lambda_kl * loss_kl

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_var": loss_var,
            "loss_kl": loss_kl,
            "high_pred_emb": pred_emb,
            "high_target_emb": target_emb,
            "macro_actions": macro_actions,
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
    val_dataloader=None,
    n_waypoints: int = 3,
    lr: float = 1e-4,
    n_epochs: int = 10,
    device: str = "cuda",
    freeze_encoder: bool = True,
    log_every_n_steps: int = 10,
    wandb_run=None,
    ckpt_callback=None,
    rollout_loss: bool = False,
    ss_start: float = 1.0,
    ss_end: float = 0.25,
    weight_decay: float = 0.01,
    select_by: str = "tf",
    ar_every: int = 5,
    use_amp: bool = True,
    compile_model: bool = False,
) -> HierarchicalLeWM:
    """Jointly optimise A_ψ and P^(2) on L_tf (stage 2).

    Stage 1 (training jepa) is handled by the existing train.py.
    Load the stage-1 checkpoint into a JEPA instance, wrap it in
    HierarchicalLeWM, then call this function.

    Parameters
    ----------
    model               : HierarchicalLeWM with a stage-1-trained inner jepa
    dataloader          : yields batches with 'pixels' and 'action' keys
    val_dataloader      : optional held-out loader; if given, a no-grad pass runs
                          each epoch and logs stage2/val_* metrics
    n_waypoints         : N interior waypoints per trajectory (HWM default 3)
    lr                  : AdamW learning rate for stage-2 parameters
    n_epochs            : number of stage-2 epochs
    device              : target device string
    freeze_encoder      : if True, no gradients flow through E or P^(1)
    log_every_n_steps   : W&B step-level logging frequency (epoch summary always logged)
    rollout_loss        : enable multi-step scheduled-sampling rollout for P^(2). When
                          False (default) training is pure one-step teacher forcing.
    ss_start, ss_end    : teacher-forcing probability annealed linearly from ss_start
                          (epoch 0) to ss_end (final epoch). 1.0 = teacher forcing, 0.0 =
                          full free-running. Only used when rollout_loss=True.
    weight_decay        : AdamW weight decay on the stage-2 params (A_ψ, P^(2)).
    select_by           : checkpoint-selection metric — "tf" (teacher-forced
                          stage2/val_loss, default; computed every epoch) or "ar"
                          (free-running rollout MSE, stage2/val_loss_ar_pred; matches how
                          the planner uses P²). NOTE: with select_by="ar", AR must be
                          computed on every epoch — set ar_every=1 — else selection only
                          sees the throttled AR epochs.
    ar_every            : run the (expensive sequential) free-running AR val pass only
                          every `ar_every` epochs (and always on the final epoch); other
                          epochs compute the cheap teacher-forced val loss only. AR metrics
                          are absent from W&B on the skipped epochs.
    ckpt_callback       : optional ModelObjectCallBack; if set, its save_epoch() is
                          called each epoch to pickle the model (same naming/location
                          convention as stage-1)
    """
    model = model.to(device)

    if compile_model:
        model.action_encoder_high = torch.compile(model.action_encoder_high)
        model.high_predictor = torch.compile(model.high_predictor)

    device_type = device.split(":")[0]   # "cuda", "cpu", or "mps"
    amp_enabled = use_amp and device_type == "cuda"

    # only the two new modules are optimised in stage 2
    stage2_params = (
        list(model.action_encoder_high.parameters())
        + list(model.high_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(stage2_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if freeze_encoder:
        for p in model.jepa.parameters():
            p.requires_grad_(False)

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = None
    best_ckpt_path = None
    best_metrics = None
    # Checkpoint-selection metric: "ar" = free-running rollout MSE (matches how the planner
    # uses P²); "tf" = teacher-forced val loss (legacy). AR is the recommended default.
    if select_by not in ("ar", "tf"):
        raise ValueError(f"select_by must be 'ar' or 'tf', got {select_by!r}")
    sel_key = "stage2/val_loss_ar_pred" if select_by == "ar" else "stage2/val_loss"
    for epoch in range(n_epochs):
        # Scheduled sampling: anneal teacher-forcing prob ss_start -> ss_end over epochs.
        if rollout_loss:
            frac = epoch / max(1, n_epochs - 1)
            tf_prob = ss_start + (ss_end - ss_start) * frac
        else:
            tf_prob = 1.0

        epoch_loss = epoch_pred = epoch_var = epoch_kl = 0.0
        epoch_mac_absmean = epoch_mac_std = 0.0
        for batch in dataloader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            T = batch["pixels"].shape[1]
            wp_idx = sample_waypoints_fixed_stride(T, N=n_waypoints, device=device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
                out = model.forward_high(
                    batch, wp_idx, freeze_encoder=freeze_encoder,
                    teacher_forcing_prob=tf_prob,
                )

            optimizer.zero_grad()
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(stage2_params, float("inf"))
            optimizer.step()

            loss_val = out["loss"].item()
            pred_val = out["loss_pred"].item()
            var_val  = out["loss_var"].item()
            kl_val   = out["loss_kl"].item()
            # A_ψ macro-action distribution vs the planner's N(0,1) prior — what the KL
            # term actually steers. |mean| → 0 and std → 1 means the moment-matching worked.
            with torch.no_grad():
                mac = out["macro_actions"]                       # (B, n_seg, d_L)
                mac_absmean = mac.mean(dim=(0, 1)).abs().mean().item()
                mac_std = mac.std(dim=(0, 1)).mean().item()
            epoch_loss += loss_val
            epoch_pred += pred_val
            epoch_var  += var_val
            epoch_kl   += kl_val
            epoch_mac_absmean += mac_absmean
            epoch_mac_std     += mac_std
            global_step += 1

            if wandb_run is not None and global_step % log_every_n_steps == 0:
                wandb_run.log({
                    "stage2/loss":          loss_val,
                    "stage2/loss_pred":     pred_val,
                    "stage2/loss_var":      var_val,
                    "stage2/loss_kl":       kl_val,
                    "stage2/macro_absmean": mac_absmean,
                    "stage2/macro_std":     mac_std,
                    "stage2/grad_norm":     grad_norm.item(),
                }, step=global_step)

        n = len(dataloader)
        epoch_metrics = {
            "stage2/epoch_loss":         epoch_loss / n,
            "stage2/epoch_loss_pred":    epoch_pred / n,
            "stage2/epoch_loss_var":     epoch_var  / n,
            "stage2/epoch_loss_kl":      epoch_kl   / n,
            "stage2/epoch_macro_absmean": epoch_mac_absmean / n,
            "stage2/epoch_macro_std":     epoch_mac_std / n,
            "stage2/epoch":              epoch + 1,
            "stage2/tf_prob":            tf_prob,
            "stage2/lr":                 scheduler.get_last_lr()[0],
        }
        scheduler.step()

        val_str = ""
        if val_dataloader is not None and len(val_dataloader) > 0:
            # The free-running AR pass is a sequential rollout (expensive); run it only
            # every ar_every epochs and on the final epoch. The teacher-forced val loss
            # (the default selection metric) is computed every epoch.
            run_ar = ((epoch + 1) % max(1, ar_every) == 0) or (epoch + 1 == n_epochs)
            val_metrics = _validate_hierarchical(
                model, val_dataloader, n_waypoints, freeze_encoder, device,
                run_ar=run_ar, device_type=device_type, amp_enabled=amp_enabled,
            )
            epoch_metrics.update(val_metrics)
            ar_str = (
                f"val_ar: {val_metrics['stage2/val_loss_ar_pred']:.5f}  "
                if "stage2/val_loss_ar_pred" in val_metrics else ""
            )
            val_str = (
                f"  | val_loss: {val_metrics['stage2/val_loss']:.5f}  "
                f"{ar_str}"
                f"val_kl: {val_metrics['stage2/val_loss_kl']:.5f}  "
                f"val_var: {val_metrics['stage2/val_loss_var']:.5f}"
            )

            # Track the best model by the selection metric (`select_by`); keep one stable
            # checkpoint on disk and defer the (single) W&B artifact upload to end of run.
            # If selecting by AR on an epoch where AR was skipped, sel_key is absent, so
            # that epoch simply isn't a candidate.
            if sel_key in val_metrics and val_metrics[sel_key] < best_val_loss and ckpt_callback is not None:
                best_val_loss = val_metrics[sel_key]
                best_epoch = epoch + 1
                best_metrics = dict(val_metrics)
                best_ckpt_path = ckpt_callback.save_best(model)
                val_str += f"  (new best {select_by})"

        print(
            f"epoch {epoch + 1}/{n_epochs}  "
            f"loss: {epoch_loss/n:.5f}  pred: {epoch_pred/n:.5f}  "
            f"kl: {epoch_kl/n:.5f}  var: {epoch_var/n:.5f}  "
            f"macro(|mean|/std): {epoch_mac_absmean/n:.3f}/{epoch_mac_std/n:.3f}"
            f"{val_str}"
        )
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=global_step)

        if ckpt_callback is not None:
            ckpt_callback.save_epoch(model, epoch + 1)

    if wandb_run is not None and best_ckpt_path is not None:
        import wandb
        artifact = wandb.Artifact(
            name=ckpt_callback.filename,
            type="model",
            metadata={
                "select_by": select_by,
                "select_metric": best_val_loss,
                "val_loss": best_metrics["stage2/val_loss"],
                "val_loss_ar_pred": best_metrics.get("stage2/val_loss_ar_pred"),
                "epoch": best_epoch,
            },
        )
        artifact.add_file(str(best_ckpt_path))
        wandb_run.log_artifact(artifact, aliases=["best"])
        wandb_run.summary["stage2/best_select_by"] = select_by
        wandb_run.summary["stage2/best_select_metric"] = best_val_loss
        wandb_run.summary["stage2/best_val_loss"] = best_metrics["stage2/val_loss"]
        if "stage2/val_loss_ar_pred" in best_metrics:
            wandb_run.summary["stage2/best_val_loss_ar_pred"] = best_metrics["stage2/val_loss_ar_pred"]
        wandb_run.summary["stage2/best_epoch"] = best_epoch
        print(f"registered best model (epoch {best_epoch}, {select_by}={best_val_loss:.5f}) to W&B")

    if best_metrics is not None:
        ar_best = best_metrics.get("stage2/val_loss_ar_pred")
        ar_best_str = f"val_ar: {ar_best:.5f}  " if ar_best is not None else ""
        print(
            f"best model ({select_by}) — epoch {best_epoch}/{n_epochs}  "
            f"val_loss: {best_metrics['stage2/val_loss']:.5f}  "
            f"{ar_best_str}"
            f"val_kl: {best_metrics['stage2/val_loss_kl']:.5f}  "
            f"ckpt: {best_ckpt_path}"
        )

    return model


@torch.no_grad()
def _validate_hierarchical(
    model: HierarchicalLeWM,
    val_dataloader,
    n_waypoints: int,
    freeze_encoder: bool,
    device: str,
    run_ar: bool = True,
    device_type: str = "cuda",
    amp_enabled: bool = False,
) -> dict:
    """Held-out teacher-forced loss, plus an optional free-running (AR) loss.

    On the same (deterministic fixed-stride) waypoints:
      - teacher-forced (tf_prob=1.0): every prediction conditions on the TRUE previous
        waypoints — the classic `stage2/val_loss`, comparable to train loss at tf_prob=1.
        Always computed; cheap (one parallel pass).
      - free-running (tf_prob=0.0): the model feeds back its OWN predictions every step,
        measuring multi-step autoregressive fidelity — the regime the planner uses. This
        is a sequential rollout (n_seg forward passes), so it is only run when `run_ar`
        is True (the caller throttles it via ar_every). When skipped, the
        `stage2/val_loss_ar*` keys are absent from the returned dict.

    eval() disables dropout in A_ψ / P^(2) (the inner JEPA stays in eval via train()).
    """
    was_training = model.training
    model.eval()
    val_loss = val_pred = val_var = val_kl = 0.0
    ar_loss = ar_pred = 0.0
    for batch in val_dataloader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        T = batch["pixels"].shape[1]
        wp_idx = sample_waypoints_fixed_stride(T, N=n_waypoints, device=device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            out = model.forward_high(
                batch, wp_idx, freeze_encoder=freeze_encoder, teacher_forcing_prob=1.0
            )
        val_loss += out["loss"].item()
        val_pred += out["loss_pred"].item()
        val_var  += out["loss_var"].item()
        val_kl   += out["loss_kl"].item()

        if run_ar:
            # Free-running rollout: tf_prob=0.0 deterministically feeds back predictions.
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
                out_ar = model.forward_high(
                    batch, wp_idx, freeze_encoder=freeze_encoder, teacher_forcing_prob=0.0
                )
            ar_loss += out_ar["loss"].item()
            ar_pred += out_ar["loss_pred"].item()

    if was_training:
        model.train()

    nv = len(val_dataloader)
    metrics = {
        "stage2/val_loss":      val_loss / nv,
        "stage2/val_loss_pred": val_pred / nv,
        "stage2/val_loss_var":  val_var  / nv,
        "stage2/val_loss_kl":   val_kl   / nv,
    }
    if run_ar:
        metrics["stage2/val_loss_ar"]      = ar_loss / nv
        metrics["stage2/val_loss_ar_pred"] = ar_pred / nv
    return metrics
