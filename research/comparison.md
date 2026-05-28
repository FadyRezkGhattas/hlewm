# HWM vs LeWM: Side-by-Side Comparison

---

## Table: Key Design Axes

| Axis | HWM | LeWM |
|---|---|---|
| **Core objective** | Hierarchical MPC planning over learned latent WMs; not a new representation learning method | Stable end-to-end JEPA from raw pixels; representation + dynamics learning |
| **What's new** | High-level WM + action encoder layered on top of existing backbone WMs | SIGReg anti-collapse regularizer enabling end-to-end training without EMA/stop-grad |
| **Encoder** | **Frozen** pretrained backbone: ViT-g/16 (V-JEPA 2, Franka), DINOv2 ViT-S/14 (Push-T), convolutional net trained from scratch (Maze) | **Trained jointly** end-to-end: ViT-Tiny (~5M params), patch size 14, 12 layers |
| **Encoder output** | Spatial feature map: 256 tokens × 1408D (Franka); patch embeddings (Push-T) | Single CLS token → 192D after MLP+BatchNorm1d projector |
| **Predictor** | ~300M ViT (Franka/Push-T); convolutional (Maze). Block-causal attention: full attention within timestep, causal across timesteps | ViT-S (~10M params), 6 layers, 16 heads, causal masking. Action via AdaLN-zero at each layer |
| **Action conditioning** | Actions mapped to hidden dim via learned affine → concatenated with spatial tokens (Franka); summed into macro-actions for high-level model | AdaLN-zero: action embedding → shift/scale/gate per transformer block (zero-init, grows during training) |
| **Target construction** | `z_{t+1} = E(s_{t+1})` from the same frozen encoder. No stop-gradient needed (encoder is frozen by design) | `z_{t+1} = enc_θ(o_{t+1})` from the same online encoder. No stop-gradient — full end-to-end backprop |
| **Stop-gradient** | None in the world model. Encoder is frozen, so there is no target encoder issue | None — end-to-end backprop through both encoder and predictor |
| **EMA** | None in HWM itself. V-JEPA 2 backbone was trained with EMA (standard JEPA), but it is frozen when HWM trains | None — explicitly removed as a contribution |
| **Collapse prevention** | Frozen encoder. Encoder representations are fixed; predictor only needs to learn dynamics on top | SIGReg: projects embeddings onto M=1024 random directions; minimizes Epps-Pulley normality statistic toward N(0,I). Provable guarantee via Cramér-Wold theorem |
| **Hierarchy** | **Two levels.** Low-level P^(1): short-horizon, primitive actions. High-level P^(2): long-horizon, macro-actions (compressed action sequences). Shared latent space enables direct subgoal transfer | **Single level.** No hierarchy. Authors list hierarchical extension as future work |
| **Macro-actions / temporal abstraction** | Learned action encoder compresses primitive action sequences between waypoints → fixed-dim latent macro-action (4D Franka, 8D Maze). Plans at two temporal resolutions | No temporal abstraction. Actions are raw (concatenated frameskip=5 primitives) |
| **Planning algorithm** | CEM (Franka, Push-T) or MPPI (Maze). Two-level: high-level plan → subgoals → low-level plan to reach subgoal | CEM (all environments). Flat: optimize single action sequence to reach goal |
| **Goal specification** | Goal observation encoded by shared encoder → goal latent z_g. L1 distance to z_g as cost | Goal observation encoded by same encoder → goal latent z_g. MSE distance to z_g as cost |
| **MPC replanning** | Every step (k=1): observe new frame, re-encode, re-plan high and low levels | Every 5 steps (receding_horizon=5): execute full planned horizon, then replan |
| **Planning cost** | L1(ẑ_final, z_g) for both levels | MSE(ẑ_H, z_g) terminal only |
| **Training loss** | L1 teacher-forcing + L1 rollout (low-level); L1 teacher-forcing only (high-level). **Code (maze) uses MSE, not L1** | MSE next-embedding prediction + λ·SIGReg. λ=0.09 default |
| **Loss hyperparameters** | Low-level: γ_tf, γ_roll (loss weights), VICReg coefficients (6 terms for maze), IDM. High-level: γ_tf only | Two terms: 1 hyperparameter (λ). M (projections) and knots are insensitive |
| **Number of trainable params** | Low-level: ~300M (Franka/Push-T); Maze: ~20K params (lightweight conv). High-level: same arch + small action encoder | 15M total (5M encoder + 10M predictor + projectors) |
| **Training data** | ~130h labeled real-robot manipulation (Franka); 20K Push-T expert episodes; 5M transitions from 25 maze layouts | 10–20K offline episodes per environment (simulated only) |
| **Hardware** | Multi-GPU (parallel CEM rollouts); V-JEPA 2 backbone required | Single GPU, few hours |
| **Environments** | Real robot (Franka arm, 7-DoF), Push-T simulation, MuJoCo PointMaze | PushT, OGBench-Cube, TwoRoom, DMControl Reacher (all simulated) |
| **Evaluation metric** | Binary task success rate (goal reached) | Binary task success rate |
| **Headline numbers** | Pick-and-place: 70% (vs 0% flat). Push-T d=75: 61% (vs 17%). Maze D∈[9,12]: 95% (vs 63%) | PushT: 86% (vs PLDM 78%, DINO-WM 79%). 48× faster than DINO-WM |
| **Task type** | Non-greedy long-horizon manipulation and navigation | Short-horizon goal-reaching (25–100 steps) |
| **Zero-shot generalization** | Yes — train on unlabeled data, deploy on new environments and objects | Yes — offline training on fixed dataset, eval on held-out trajectories |
| **Reward required** | No | No |
| **Key failure mode** | Subgoal reachability: high-level plan may propose latent states dynamically unreachable by low-level | Low intrinsic dimensionality: SIGReg forces high-dimensional Gaussian on low-d environments (Two-Room: 20% SR) |

---

## Prose: Conceptual Differences

### 1. What problem each paper is actually solving

HWM and LeWM address different problems in the same family. HWM asks: **given that you already have a working latent world model, how do you plan over long horizons without failure from error accumulation?** It does not learn new representations — it adds a planning layer. LeWM asks: **how do you train a JEPA world model stably from raw pixels, end-to-end, without the heuristic stabilizers (EMA, stop-grad, frozen encoders) that the field has accumulated?** It does not add a planning capability — it removes training fragility.

These are orthogonal contributions. LeWM produces a better-trained latent WM; HWM makes better use of any latent WM for planning. The authors of LeWM explicitly list "hierarchical world modeling" as future work, and the authors of HWM explicitly say their framework is backbone-agnostic — suggesting LeWM as a backbone for HWM is a natural combination neither paper explores.

### 2. Collapse prevention: structural vs. principled

The JEPA literature has two main camps for collapse prevention. The **structural camp** (V-JEPA 2, DINO-WM, HWM) uses a frozen pretrained encoder: the representation cannot collapse because it was already fixed. This works but limits what the model can learn — the dynamics predictor can only learn dynamics in the pretrained encoder's latent space. The **principled camp** (PLDM, LeWM) learns encoder and predictor jointly, using statistical regularizers to prevent collapse. PLDM uses VICReg (6-term loss, training instability). LeWM uses SIGReg (2-term loss, provable guarantee).

The tradeoff: structural approaches inherit rich representations from large-scale pretraining but cannot adapt them to the downstream task. Principled approaches can learn task-relevant representations from scratch but require enough data diversity for the regularizer to work.

### 3. Latent space geometry

In HWM (Franka), the latent space is V-JEPA 2's spatial feature map: 256 spatial tokens per frame, each 1408-dimensional. This is a spatially explicit representation — different tokens correspond to different image regions. Planning and subgoal transfer happen in this high-dimensional, spatially structured space.

In LeWM, the latent space is the CLS token: a single 192-dimensional global representation of each frame. It is structurally isotropic (SIGReg enforces Gaussian marginals), spatially compressed, and semantically global. Planning happens in this compact space, which is why LeWM plans 48× faster than DINO-WM.

These are fundamentally different notions of "latent." HWM's latent preserves spatial locality; LeWM's does not. This has implications for tasks requiring fine-grained spatial reasoning.

### 4. Hierarchy and temporal abstraction

HWM introduces temporal abstraction explicitly: the high-level model learns dynamics at a longer timescale, and macro-actions compress sequences of primitive actions. The key insight is that fewer sequential prediction steps means less error accumulation — a single high-level step is more accurate than 10–16 sequential low-level steps at the same total horizon.

LeWM makes no temporal abstraction. Its CEM planning horizon is H=5 steps at frameskip=5, corresponding to 25 environment steps. For longer horizons, it would accumulate prediction errors at the low level, exactly the failure mode HWM is designed to address.

### 5. Real-world vs. simulation gap

HWM is demonstrated on a real robot (Franka arm, Droid dataset, real camera input). Its approach is designed to handle the distribution shifts, noise, and latency of real robot data. The frozen encoder provides stable representations even when pixel statistics shift slightly.

LeWM is evaluated entirely in simulation with clean, controlled data. Its end-to-end training requires sufficient data diversity for SIGReg to work — something that may be harder to guarantee with real robot data. The Two-Room failure illustrates what happens when data is insufficiently diverse.

### 6. Modularity claims

Both papers claim modularity, but in different senses. HWM claims modular in the planning sense: the same hierarchical planning framework applies to three different backbone world models without modification. LeWM claims modular in the architecture sense: the encoder can be swapped (ViT, ResNet both work), and the regularizer is decoupled from the architecture.

In practice, HWM's modularity is real at the planning algorithm level but less real at the training level — the high-level WM must be retrained for each new backbone, and the shared latent space assumption ties them together. LeWM's modularity is real at the encoder level but the loss function is hardcoded in `lejepa_forward` and not easily swappable.
