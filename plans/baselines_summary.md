# Experiment Settings — LeWM (2603.19312) and HWM (2604.03208)

The uploaded code implements **LeWM (LeWorldModel)** — `jepa.py`, `module.py`, and the `lejepa_forward` in `train.py` match paper #1's two-term objective (`pred_loss + λ·sigreg_loss`), AdaLN-zero predictor (`ConditionalBlock`), and post-encoder/predictor BN-MLP projectors. Paper #2 (HWM) treats a LeWM-style model as its *low-level* world model and trains an additional *high-level* world model on top for hierarchical planning. This doc summarizes train/test settings for both, with explicit pointers to the uploaded code.

---

## Paper 1 — LeWM: Stable End-to-End JEPA from Pixels

### Method (what the code implements)

Two trainable components — encoder $\text{enc}_\theta$ and predictor $\text{pred}_\phi$ — jointly optimized end-to-end:

$$\mathcal{L}_{\text{LeWM}} = \underbrace{\|\hat z_{t+1} - z_{t+1}\|_2^2}_{L_{\text{pred}}} + \lambda \cdot \text{SIGReg}(Z)$$

- **No** stop-gradient, EMA, or pretrained encoder.
- SIGReg ≜ mean Epps–Pulley statistic over $M$ random univariate projections of latents; encourages an isotropic-Gaussian target distribution. → `module.SIGReg`.
- Projectors after the encoder CLS token *and* after the predictor output are 1-layer MLPs with **BatchNorm** (necessary because the final ViT LayerNorm otherwise prevents SIGReg from being optimized). → `module.MLP`, used as `projector` and `pred_proj` in `jepa.JEPA`.

### Training Setup

| Item | Value | Code reference |
|---|---|---|
| Encoder | ViT-Tiny, patch size 14, hidden 192, 12 layers, 3 heads (~5M params) | `train.py: spt.backbone.utils.vit_hf(cfg.encoder_scale, patch_size=cfg.patch_size, ...)` |
| Predictor | ViT-S (6 layers, 16 heads, dropout 0.1) with AdaLN-zero action conditioning (~10M params total model) | `module.ARPredictor` + `ConditionalBlock` |
| Action encoder | Conv1d(1×1) + 2-layer MLP-SiLU | `module.Embedder` |
| Frame size | 224 × 224 | `cfg.img_size` |
| Batch size | 128 | `cfg.loader` |
| Sub-trajectory | 4 frames × 4 action blocks (frameskip 5 ⇒ 20 raw env steps) | `cfg.data.dataset.frameskip`, `cfg.wm.history_size` |
| History length $N$ | **3** for PushT / OGBench-Cube, **1** for TwoRoom | `cfg.wm.history_size` |
| λ (SIGReg weight) | 0.1 (robust over [0.01, 0.2]) | `cfg.loss.sigreg.weight` |
| # projections $M$ | 1024 (insensitive) | `cfg.loss.sigreg.kwargs.num_proj` |
| Integration knots | 17 (insensitive) | `cfg.loss.sigreg.kwargs.knots` |
| Epochs per env | 10 | `cfg.trainer.max_epochs` |
| Hardware | Single NVIDIA L40S GPU | — |

### Environments & Datasets

| Env | Modality | Episodes | Avg length | Behavior policy |
|---|---|---|---|---|
| **TwoRoom** | 2D nav | 10,000 | ~92 steps | Noisy heuristic |
| **PushT** | 2D manipulation | 20,000 | ~196 steps | Expert (DINO-WM dataset) |
| **OGBench-Cube** | 3D manipulation | 10,000 | 200 steps | OGBench heuristic, single-cube |
| **Reacher** (DMC) | 2D motion | 10,000 | 200 steps | SAC policy |

### Test / Planning Setup

- **Planner**: Cross-Entropy Method (CEM) in latent space. → `cfg.solver` instantiated in `eval.py`.
- **Cost**: terminal $\|\hat z_H - z_g\|_2^2$. → `JEPA.criterion`, `JEPA.get_cost` (note: code uses MSE on last predicted vs. goal embedding).
- **CEM**: 300 samples / iteration, top-30 elites, 30 iterations (PushT) or 10 (others), init σ = I.
- **MPC**: horizon $H = 5$ planning steps (= 25 env steps with frameskip 5); receding-horizon with full sequence executed before replanning ("action_block" in `cfg.plan_config`).
- **Goal sampling**: from offline trajectories — goal_offset_steps timesteps after the chosen start. → `eval.py: get_episodes_length`, `goal_offset_steps`.
- **Eval budget**: 50 steps with goal 25 ahead for **all four environments** (PushT, OGBench, Reacher, **and TwoRoom**). The printed paper Appendix F.1 says 150/100 for TwoRoom, but first author lucas-maes confirmed on GitHub (Apr 12) that this is a typo — actual eval used 50/25, which matches the `config/eval/tworoom.yaml` defaults. → `cfg.eval.eval_budget`, `cfg.eval.goal_offset_steps`.

### Baselines

| Baseline | Family | Loss / training | Notes |
|---|---|---|---|
| **DINO-WM** | JEPA + frozen pretrained encoder | $\frac{1}{BT}\sum\|\hat z_{t+1}-z_{t+1}\|_2^2$ with DINOv2 as frozen encoder | Same arch/HP as Zhou et al. 2024. Variant **DINO-WM+prop** also receives proprioception. |
| **PLDM** | End-to-end JEPA (VICReg-style) | 7-term loss: prediction + VICReg variance + VICReg covariance + temporal-sim + temporal-var + temporal-cov + IDM. Coefficients $(α,β,γ,ζ,ν,μ) = (18, 12, 0.2, 0.7, 0, 0)$ via grid search on PushT, then fixed across envs. | Closest setup to LeWM. |
| **GCBC** | Imitation | $\mathbb{E}\|\pi_\theta(s_t,g)-a_t\|_2^2$ over DINOv2 patches | Goal-conditioned BC. |
| **GCIQL** | Offline goal-cond. RL | IQL with expectile regression on $Q_\psi(s,a,g)$ and $V_\theta(s,g)$, AWR policy extraction | DINOv2 patch features. |
| **GCIVL** | Offline goal-cond. RL | IVL: drops Q, value learned directly via bootstrapped expectile | DINOv2 patch features. |
| **Random** | Reference | — | — |

### Key Results (success rate, 50-traj eval)

| Method | TwoRoom | Reacher | PushT | OGB-Cube |
|---|---|---|---|---|
| **LeWM** | 87 | **86** | **96** | 74 |
| DINO-WM | 100 | 79 | 74 | **86** |
| DINO-WM + prop | 100 | — | 92 | — |
| PLDM | 97 | 78 | 78 | 65 |
| GCBC | 100 | — | 75 | 84 |
| GCIQL | 100 | — | 20 | 64 |
| GCIVL | 100 | — | 33 | 56 |

LeWM planning is **~48× faster** than DINO-WM (0.98 s vs. 47 s per full plan) because the [CLS] token gives ~200× fewer tokens than DINOv2's patch grid.

### Ablations (PushT)

| Knob | Range | Finding |
|---|---|---|
| Embedding dim | 8 → 384 | Saturates near 184 |
| # SIGReg projections | 64 → 1024 | Flat |
| # integration knots | 4 → 32 | Flat |
| λ | 0.01 → 0.5 | ≥80% SR over [0.01, 0.2]; crashes at 0.5 |
| Predictor size | Tiny / S / Base | ViT-S best (96%) |
| Encoder arch | ViT vs ResNet-18 | 96 vs 94 — backbone-agnostic |
| Predictor dropout | 0 / 0.1 / 0.2 / 0.5 | 0.1 best (matches `cfg.predictor.dropout`) |
| Decoder reconstruction loss | with / without | Slightly *worse* with decoder loss |

---

## Paper 2 — HWM: Hierarchical Planning with Latent World Models

HWM is a **plug-in** planning abstraction that wraps an existing latent world model — for the code here, replace "low-level WM" with the LeWM that `train.py` produces, then train an additional **high-level WM** with macro-actions.

### Method

Two world models share a latent space:
- **Low-level** $P^{(1)}(z_{t+1}\,|\,z_t, a_t)$: trained per Eq. (2-4) — teacher-forcing $\ell_1$ + multi-step rollout $\ell_1$.
- **High-level** $P^{(2)}(z_{t_{k+1}}\,|\,z_{t_k}, l_{t_k})$: predicts between waypoint states; conditioned on **latent macro-actions** $l_{t_k} = A_\psi(a_{t_k:t_{k+1}})$ from a learned action encoder (transformer CLS → MLP). Trained with $\ell_1$ teacher-forcing only.

Loss for both is $\ell_1$ instead of LeWM's $\ell_2$.

Planning is two-stage MPC:
- High level picks $\hat l_{1:H}$ minimizing $\|z_g - P^{(2)}(\hat l_{1:H}; z_1)\|_1$.
- The first predicted latent $\tilde z_1$ becomes a **subgoal**; low level optimizes primitive actions $\|\tilde z_1 - P^{(1)}(\hat a_{1:h}; z_1)\|_1$.

### Training Setup (per backbone)

| Backbone | Domain | Low-level | High-level | Waypoints $N$ |
|---|---|---|---|---|
| **VJEPA2-AC** (Franka) | DROID 96h + RoboSet 30h, 256×256 | ViT-g/16 frozen encoder + ~300M ViT predictor, $T=16$, 200 epochs, BS 256, FPS 3–10 (DROID) / 1–5 (RoboSet) | Same ViT arch, $T=3$, 120 epochs, BS 768, segments 0.33–4 s | 3 |
| **DINO-WM** (Push-T) | DINO-WM offline dataset, 18,500 traj | 25M ViT, $T=4$, DINOv2 frozen, 100 epochs, BS 256 | Scaled 25M → 75M (10 layers, dim 768, MLP 3072, 12 heads), $T=5$, 500 epochs, BS 128, segments 25–70 steps | 5 |
| **PLDM** (Diverse Maze) | 25 train mazes × 2000 ep × 100 steps = 5M transitions, 98×98 RGB | Conv encoder 33k + predictor 20k, $T=15$, VICReg, lr 0.018, 3 epochs | High-capacity conv predictor, frozen encoder reused, $T=6$, 5 epochs, stride 10 | 6 |

All losses combine: `γ_tf · L_tf + γ_roll · L_roll` with weights given in the paper's Tables 6–8.

### Test / Planning Setup

**Franka (real robot)** — 7-DoF Panda + Robotiq gripper:
- High-level: CEM, 3000 samples × 15 iters, latent action dim 4, prediction horizon 2, σ-EMA 0.65.
- Low-level: CEM, 800 samples × 15 iters, horizon 2, σ-EMA 0.25.
- Replan every $k=1$ step.

**Push-T** — extended horizons ($d \in \{25, 50, 75\}$ vs. DINO-WM's $d=25$):
- High-level: 900–1500 samples × 20–40 iters, horizon 2–5.
- Low-level: 300–1200 samples × 20–30 iters, horizon 5, $k=5$.

**Diverse Maze** — 10×10 grid, 25 train / 20 held-out layouts, distances $D\in\{[5,8], [9,12], [13,16]\}$:
- Uses **MPPI** (not CEM). High-level: 2000–4000 samples, horizon 25–47. Low-level: 500–1000 samples, horizon 15, $k=4$, λ=0.0025, σ=10/5.

### Baselines

| Group | Baselines | Notes |
|---|---|---|
| **Flat WM (apples-to-apples)** | VJEPA2-AC, DINO-WM, PLDM (same arch & data, no hierarchy) | Isolates the effect of hierarchy. |
| **Vision-Language-Action (Franka only)** | Octo (DROID-finetuned, image goals), π₀-FAST-DROID, π₀.5-DROID | Pretrained on ~77× more robot data than HWM's WM. Text-goal prompting tuned. |
| **Goal-conditioned RL** | GCIQL, HIQL (subgoal generator + reacher policy sharing value fn), HILP (distance-equivariant rep + direction-conditioned policy) | Hyperparams tuned per (env, horizon); from OGBench / official repos. |

### Key Results

**Franka** (zero-shot from a single goal image):

| Method | P&P Cup (subgoals) | P&P Cup | P&P Box | Drawer |
|---|---|---|---|---|
| Octo | 20 | 10 | 0 | 43 |
| π₀-FAST-DROID | — | 52 | 18 | — |
| π₀.5-DROID | — | 68 | 36 | — |
| VJEPA2-AC (flat) | 80 | **0** | **0** | 30 |
| **VJEPA2-AC + HWM** | **80** | **70** | **60** | **70** |

**Push-T** (success rate %):

| | $d=25$ | $d=50$ | $d=75$ |
|---|---|---|---|
| GCIQL / HIQL / HILP | 40/55/25 | 25/30/13 | 7.5/20/0 |
| DINO-WM (flat) | 84 | 55 | 17 |
| **DINO-WM + HWM** | **89** | **78** | **61** |

**Diverse Maze**:

| | $D\in[5,8]$ | $[9,12]$ | $[13,16]$ |
|---|---|---|---|
| GCIQL / HIQL / HILP | 85/88/48 | 40/73/20 | 33/48/10 |
| PLDM (flat) | 100 | 63 | 44 |
| **PLDM + HWM** | **100** | **95** | **83** |

### Ablations Worth Noting

- **Latent vs. delta-pose macro-actions**: latent macro-actions improve cosine sim to expert behavior (0.88 vs 0.80) and reduce $\ell_1$ — the high-level model needs the full action sequence's structure, not just net displacement.
- **Latent action dim** on Franka: $d=4$ is the sweet spot — higher dims yield subgoals the low-level planner can't reach with greedy behavior.
- **Long-horizon prediction**: low-level WM beats high-level WM for ≤1 s, high-level wins for ≥1.5 s — validates hierarchy.

---

## How the Code Maps to Each Paper

| Code object | LeWM role | HWM role |
|---|---|---|
| `JEPA.encode` | Encoder $E$ + projector | Same — shared latent space across both levels |
| `JEPA.predict` | Predictor $\text{pred}_\phi$ | This *is* the low-level $P^{(1)}$ |
| `JEPA.rollout` | Autoregressive latent rollout for planning | Same; HWM would add a second JEPA-like module for $P^{(2)}$ |
| `module.Embedder` | Action encoder (per-step) | LeWM's per-step action embed; for HWM you'd add a transformer-based `Aψ` producing macro-actions from action sub-sequences |
| `module.SIGReg` | Anti-collapse loss | Not used in HWM (which relies on the backbone's own anti-collapse) |
| `module.ConditionalBlock` | AdaLN-zero action conditioning | Same in both papers' predictor designs |
| `lejepa_forward` (`pred_loss + λ·sigreg_loss`) | Full LeWM training loss | Would be the low-level loss; high-level uses $\ell_1$ teacher-forcing on waypoint pairs |
| `eval.py` CEM solver | Single-level latent MPC | Becomes the *low-level* planner; HWM adds an outer CEM/MPPI loop over latent macro-actions |
| `cfg.plan_config.horizon`, `action_block` | Planning horizon $H$, executed chunk | Low-level horizon $h$; HWM introduces separate `H` and macro-action sequence length |

### To Extend This Code Toward HWM

1. Train a LeWM as-is — this gives the encoder $E$ and low-level predictor $P^{(1)}$.
2. Add an action encoder $A_\psi$ (transformer with CLS → MLP) that consumes variable-length action chunks.
3. Add a second predictor $P^{(2)}$ (same `ARPredictor` class is fine) conditioned on macro-actions, trained with $\ell_1$ teacher-forcing on waypoint latent pairs from the *frozen* encoder $E$.
4. Wrap `eval.py`'s solver in an outer CEM/MPPI loop that produces subgoals, then re-uses the inner solver against each subgoal.
