# Glossary: Overloaded Terms

Terms that appear in both papers (and in `jepa_notes.md`) but mean different things, or have a shared surface meaning with important differences in practice.

---

## latent / latent space

**HWM (Franka):** The output of the frozen V-JEPA 2 encoder — a spatial feature map of shape `[256 tokens × 1408D]` per frame. Spatially explicit: each token corresponds to a 16×16 patch of the image. Planning and subgoal transfer happen in this space. The latent space is defined entirely by the pretrained backbone and is not changed during HWM training.

**HWM (Maze / PLDM):** The output of a convolutional encoder trained jointly with the low-level predictor — a spatial feature map of shape `[32, 35, 35]`. Trained end-to-end with VICReg anti-collapse. The high-level model operates on these latents as well (shared space).

**LeWM:** The CLS token of a ViT-Tiny encoder, projected through MLP+BatchNorm1d — a single 192-dimensional vector per frame. Globally pooled, no spatial locality. SIGReg enforces that the aggregate distribution across all frames and batch elements approximates N(0,I).

**Cross-talk warning:** "Working in latent space" means spatially rich feature maps in HWM (Franka) but a compact global vector in LeWM. These spaces have very different structure and dimensionality.

---

## target / target encoder

**In JEPA pretraining (jepa_notes.md / V-JEPA 2):** The *target encoder* is an EMA copy of the context encoder. Its outputs are the regression targets. "Target" = what the predictor is trained to predict. The target encoder is a separate network — its weights do not receive gradient, and it is updated slowly via EMA.

**HWM (during world model training):** There is no target encoder. The single frozen V-JEPA 2 backbone encodes both the current frame (predictor input) and the next frame (regression target). "Target" = the next-frame latent `z_{t+1} = E(s_{t+1})` produced by the same frozen encoder. The word "target" in the paper refers to the learning target, not to a separate target-encoder network.

**LeWM:** Also no target encoder. The online encoder `enc_θ` produces both `z_t` (predictor input) and `z_{t+1}` (regression target) in the same forward pass. Gradients flow through both. SIGReg replaces the collapse-prevention role that the EMA target encoder played in I-JEPA / V-JEPA.

**Cross-talk warning:** "Target encoder" in jepa_notes.md is a specific module (the EMA copy). In HWM and LeWM, "target" means the regression label — no separate target encoder network exists.

---

## context

**In JEPA pretraining (jepa_notes.md / I-JEPA / V-JEPA 2):** The *context* is the visible (unmasked) portion of the input given to the context encoder. The target is the masked portion the predictor must recover. Context and target are complementary disjoint regions.

**HWM:** "Context" is not a prominent term. The world model sees the full trajectory `{(z_k, p_k, a_k)}_{k∈[T]}` — all frames, not a masked subset.

**LeWM:** The *context* is the history of N=3 frames (`ctx_emb = emb[:, :3]`) fed to the predictor. The predictor uses these to predict the next frame's embedding (`tgt_emb = emb[:, 1:]`). No masking — it is a causal autoregressive model, not a masked prediction model.

**Cross-talk warning:** "Context" in pretraining JEPA means masked vs. unmasked; in world-modeling JEPA (HWM, LeWM) it means the history window fed to the predictor.

---

## predictor

**In JEPA pretraining (jepa_notes.md):** The predictor takes context encoder outputs + mask tokens (with positional embeddings of target locations) and predicts target representations. It is a relatively small transformer (~50M params in I-JEPA). Its key property: it receives location information via positional embeddings on mask tokens.

**HWM — low-level P^(1):** A ~300M ViT that takes `{z_k, p_k, a_k}` and predicts `z_{k+1}`. Block-causal attention. This is the world model's dynamics component.

**HWM — high-level P^(2):** Same ViT architecture as P^(1), conditioned on latent macro-actions `l_t` instead of primitive actions. Predicts next waypoint latent `ẑ_{t_{k+1}}`.

**LeWM:** ARPredictor — ViT-S with 6 layers, 16 heads, causal masking, AdaLN-zero conditioning on action embeddings. Predicts `ẑ_{t+1}` from a window of N=3 context embeddings.

**Cross-talk warning:** "Predictor" in pretraining JEPA is primarily about spatial prediction from context; in world-modeling JEPA it is a temporal dynamics model conditioned on actions.

---

## world model

**HWM:** The paper uses "world model" (WM) to mean the full encoder + predictor system that approximates transition dynamics `p(z_{t+1} | z_t, a_t)`. HWM specifically has two world models: P^(1) and P^(2). Often "low-level world model" and "high-level world model" refer to the predictor component only (shared encoder).

**LeWM:** "World model" = the full `JEPA` object: encoder + predictor + projectors. In the `le-wm` codebase, the `JEPA` class is the world model.

**HWM_PLDM codebase:** "World model" roughly corresponds to `JEPA` for single-level, `HJEPA` for two-level.

---

## level

**HWM only.** Level 1 (L1) = the low-level world model P^(1); operates on individual timesteps, conditioned on primitive actions. Level 2 (L2) = the high-level world model P^(2); operates on waypoints, conditioned on macro-actions.

In the `HWM_PLDM` codebase, `level1` and `level2` are attributes of `HJEPA` — both are `JEPA` instances. The YAML config keys `hjepa.level1` and `hjepa.level2` configure them independently. `train_l1: false` + `freeze_l1: true` = L2-only training with frozen L1.

**LeWM:** No levels. Single-level only.

---

## plan / planning

**HWM:** Planning = CEM or MPPI optimization over action sequences at inference time.
- High-level plan: a sequence of `H` latent macro-actions `l̂_{1:H}`, optimized to minimize `||z_g - P^(2)(l̂_{1:H}; z_1)||_1`.
- Low-level plan: a sequence of `h` primitive actions `â_{1:h}`, optimized to reach the first subgoal `z̃_1`.

**LeWM:** Planning = CEM optimization over a sequence of `H=5` primitive actions (each is `frameskip×A_dim`-dimensional), optimizing terminal MSE to goal latent.

**Cross-talk warning:** Both use CEM, but plan over fundamentally different action spaces (macro vs. primitive) and at different temporal scales.

---

## macro-action

**HWM:** A compact latent vector that summarizes a variable-length sequence of primitive actions between two waypoints.
- Franka: 4D, produced by a transformer action encoder A_ψ with CLS token projection (per paper).
- Maze (code): sum of primitive actions in a chunk, then normalized — a fixed operation, not a learned encoder.
- Push-T: produced by a transformer action encoder (per paper appendix).

**LeWM:** Not used. Actions are always primitive (frameskip=5 raw actions concatenated per step).

---

## waypoint

**HWM:** A state selected from a trajectory to serve as an anchor for high-level world model training. Each training sample consists of N waypoints with variable-length action chunks between them. High-level model predicts from one waypoint to the next.
- Franka: N=3 waypoints, random middle waypoint, variable segment length (up to 4s).
- Maze (code): N=6 waypoints, fixed stride of 10 steps — equally spaced, no randomness.

**LeWM:** Not used. No temporal abstraction.

---

## encoder

**HWM:** The encoder `E` maps observations to latents. It is **frozen** during all HWM training. In the Franka setting it is the pretrained V-JEPA 2 ViT-g/16; in Push-T it is DINOv2 ViT-S/14; in Maze it is a convolutional encoder trained jointly with the PLDM backbone (but frozen when training L2).

**LeWM:** The encoder `enc_θ` is a ViT-Tiny trained **jointly** with the predictor end-to-end. It produces a CLS token representation (192D after projector). The encoder's parameters are updated by both `L_pred` and `SIGReg` gradients simultaneously.

---

## collapse / representation collapse

Both papers use this term to mean the failure mode where the encoder maps all inputs to a nearly identical representation, trivially minimizing the prediction loss with zero information in the latent space.

**HWM's approach to avoiding collapse:** Use a frozen pretrained encoder. The representations are fixed; they cannot collapse.

**LeWM's approach to avoiding collapse:** SIGReg regularizer forces the embedding distribution to match N(0,I). Proven to prevent collapse: SIGReg(Z) → 0 iff P_Z → N(0,I) (Cramér-Wold).

**PLDM's approach (used as HWM low-level backbone for maze):** VICReg — variance + invariance + covariance regularization, 6 loss terms.

---

## teacher-forcing

Both papers: a training regime where the model receives **ground-truth** observations (encoded from real data) at each timestep, rather than its own predictions from the previous step. Produces stable gradients but doesn't train the model to handle its own errors.

**HWM:** Teacher-forcing loss `L_tf` is the primary loss for both levels. The rollout loss `L_roll` (feeding own predictions back) is added for the low-level model to address the resulting gap at inference time. High-level model: `γ_roll = 0` (teacher-forcing only, no rollout loss).

**LeWM:** Teacher-forcing only (`L_pred`). No rollout loss. At planning time, the model is used autoregressively (rolling out its own predictions), which is a training-inference mismatch — partially mitigated by the moderate dropout in the predictor.

---

## rollout loss

**HWM only.** A loss computed by feeding the model's own predictions back as input for multiple steps, then comparing the final prediction to ground truth. Addresses the teacher-forcing gap.

Formula (Appendix B.1): `L_roll(θ) = Σ_{j=2}^{T} || P^(1)_θ(a_{1:j}, z_1) - z_{j+1} ||_1`

- Applied to low-level model only (Franka: `γ_roll = 1.0`; Push-T: `γ_roll = 0.0`; Maze: `γ_roll = 1.0`).
- Not applied to high-level model (`γ_roll = 0.0` everywhere).

**LeWM:** Not used. Only teacher-forcing prediction loss.

---

## subgoal

**HWM:** An intermediate predicted latent state produced by the high-level planner, used as the target for the low-level planner.
- `z̃_i = P^(2)(l*_{1:i}; z_1)` — the i-th state in the optimized high-level plan.
- `z̃_1` is the first subgoal; low-level CEM optimizes primitive actions to reach it.
- Subgoals live in the same latent space as observations, enabling direct L1 distance matching without a decoder.

**LeWM:** Not used. Planning directly targets the final goal latent.

---

## CEM (Cross-Entropy Method)

Both papers use CEM for trajectory optimization at planning time. Same algorithm; different search spaces.

**HWM (Franka high-level):** Searches over 4D latent macro-action sequences of length H=2. 3000 samples, 22 elites, 15 iterations.

**HWM (Franka low-level):** Searches over 7D primitive action sequences of length h=2. 800 samples, 12 elites, 5 iterations.

**LeWM:** Searches over `frameskip × A_dim`-dimensional primitive action sequences of length H=5. 300 samples, 30 elites, 30 iterations (PushT) / 10 iterations (others).

The CEM algorithm itself is identical in both papers (sample from Gaussian, evaluate candidates, refit distribution to top-k).

---

## SIGReg

**LeWM only.** Sketched Isotropic Gaussian Regularizer. A statistical regularizer that pushes the distribution of latent embeddings toward N(0,I) by:
1. Projecting embeddings onto M random unit-norm directions.
2. Computing the Epps-Pulley normality test statistic for each 1D projection.
3. Minimizing the average statistic.

Comes from companion paper "Le-JEPA" (Balestriero & LeCun 2025). Not used in HWM.

---

## MPPI (Model Predictive Path Integral)

**HWM (Maze backend only).** A sampling-based optimizer that importance-weights action samples by `exp(-cost/λ)` and updates a nominal trajectory by a weighted average. Produces smoother updates than CEM (no hard elite selection). Used instead of CEM for the maze environment.

**LeWM:** Not used (uses CEM only).
