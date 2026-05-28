# From I-JEPA to V-JEPA-AC: An Operational Guide

*A plain-language walkthrough of the JEPA family of models — focusing on what actually happens to the data at each step, not what the equations say.*

---

## 1. I-JEPA: The Foundation

### Core Idea

I-JEPA (Image-based Joint-Embedding Predictive Architecture) is a self-supervised learning method from Meta AI. Instead of reconstructing pixels like masked autoencoders, I-JEPA learns by predicting the *representations* of masked image regions in an abstract latent space. This means the model focuses on semantic, high-level features rather than pixel-level details.

This aligns with Yann LeCun's argument that generative models waste capacity modeling irrelevant pixel-level details (like the exact position of every blade of grass), and that operating in latent space is a more efficient path to learning world models.

### Architecture: Three Components

**Context Encoder:** A Vision Transformer that processes only the visible (unmasked) patches. The unseen patches are dropped entirely — no mask tokens, no placeholders, just gone. Each visible patch carries a positional embedding corresponding to its original location in the image grid, so spatial information is preserved despite the sparsity.

**Target Encoder:** Processes the full image. Its weights are an exponential moving average (EMA) of the context encoder weights. After encoding, you extract representations at the target patch positions.

**Predictor:** Takes the context encoder's output tokens and a set of learnable mask tokens. Those mask tokens carry the positional embeddings of the target patch positions — this is how the predictor knows *where* it needs to predict. It then attends over the encoded context to produce predictions at those target locations.

### Tensor Shapes Through the Pipeline

The context encoder receives: `[B, num_context_tokens, D]`

Where `num_context_tokens` is consistent across the batch because block dimensions (height, width in patches) are shared across all samples — only the positions vary per sample. This avoids any need for padding or attention masks.

The predictor receives context tokens concatenated with mask tokens. Since I-JEPA applies the predictor M times (once per target block), the batch dimension is expanded:

`[B * M, context_tokens + target_tokens_per_mask, D]`

The `repeat_interleave` operation duplicates the context representations M times so each target block gets its own copy. Each target block is predicted independently — this prevents attention leakage where one target block's mask tokens could help predict another.

### Why Token Order Doesn't Matter

Self-attention is permutation equivariant. The output for each token depends on its content and positional embedding, not its position in the sequence. You could shuffle context and target tokens arbitrarily, and as long as each token has the correct PE attached, the output would be identical (up to the corresponding permutation of outputs).

### EMA Sensitivity

The target encoder's EMA schedule typically follows a cosine ramp from ~0.996 toward 1.0 during training. Too aggressive (low momentum) and targets shift too fast, causing instability or collapse. Too conservative and targets become stale and uninformative. The interaction between EMA schedule, learning rate, and batch size can be quite delicate — this is a known pain point in joint-embedding methods generally (BYOL, DINO, etc.).

---

## 2. V-JEPA: Extending to Video

V-JEPA extends I-JEPA to video by working with spatiotemporal volumes instead of 2D images.

### Input Format

`[B, D, H, W, C]` where B is batch size, D is depth (number of frames), and [H, W, C] is a standard frame.

During pretraining, 16 frames are sampled from each video with a temporal stride of 4. Temporal stride of 4 means picking every 4th frame: frame 0, 4, 8, 12, ..., 60. That's 16 frames spanning 64 frames total (~2 seconds at 30fps). The stride is the step size between selected frames, not the number skipped (you skip 3 in between).

### Patch Embedding via 3D Convolution

The first layer is a 3D convolution with d filters of size 2×16×16, temporal stride 2, spatial stride 16. Each filter processes a spatiotemporal patch of 2 frames × 16×16 pixels.

Output shape calculation (assuming 224×224 input):

- Temporal: 16 input frames ÷ stride 2 = **8** temporal positions
- Height: 224 pixels ÷ stride 16 = **14** spatial positions
- Width: 224 pixels ÷ stride 16 = **14** spatial positions
- Channels: **d** (embedding dimension — each filter produces one output channel)

So: `[B, 16, 224, 224, 3]` → `[B, 8, 14, 14, d]`

The input channels (C=3 for RGB) are absorbed in the convolution dot product. Each filter has actual shape `[C_in, 2, 16, 16]` — it sums across all input channels. Papers typically omit C_in from kernel dimensions since it's implicitly determined by the input.

This is the video equivalent of ViT's patch embedding — a single conv layer that simultaneously tokenizes and embeds.

### Positional Embeddings

V-JEPA uses absolute 3D sin-cos positional embeddings to encode spatiotemporal position.

### Masking Strategy

Like I-JEPA but extended to 3D: target blocks are spatiotemporal tubes spanning both space and time. V-JEPA uses aggressive masking — roughly 90% of the volume is masked (target) and only ~10% is kept as context. This makes sense for video because of massive temporal redundancy: you can give the model very little context and still expect meaningful prediction. The high masking ratio forces learning of spatiotemporal representations rather than relying on easy spatial shortcuts.

---

## 3. V-JEPA 2: Scaling Up

V-JEPA 2 is primarily about scaling V-JEPA with some simplifications:

- Larger models (up to 1B parameters, ViT-g)
- More training data (1M+ hours of internet video)
- Simplified hyperparameters (fixed EMA and weight decay instead of ramp-up schedules)
- 3D Rotary Position Embeddings (RoPE) instead of absolute sin-cos, which helps stabilize training at scale. The feature dimension is partitioned into three segments (temporal, height, width) with 1D rotations applied separately to each.
- Progressive resolution training: train on shorter, lower-resolution clips first, then increase resolution/duration only during the final cooldown phase — achieving 8× speedup for high-resolution training.

---

## 4. V-JEPA-AC: Adding Actions for Planning

V-JEPA-AC takes the pretrained V-JEPA 2 encoder and builds an action-conditioned world model on top of it. The encoder is frozen — only a new predictor is trained.

### Inputs

**Video:** 16 frames sampled at 4fps from 4-second clips (256×256 resolution) from the Droid dataset (robot manipulation videos). Unlike V-JEPA's 30fps with temporal stride, this is direct 4fps sampling — robotics doesn't need high temporal resolution.

**End-effector state s_k:** A 7D vector per frame describing the robot's hand pose relative to its base:

- Dimensions 1–3: Cartesian position (x, y, z)
- Dimensions 4–6: Orientation as extrinsic Euler angles (roll, pitch, yaw)
- Dimension 7: Gripper state (open/closed)

**Actions a_k:** Defined as state deltas — a_k = s_{k+1} − s_k. Each action is a 7D vector representing the change in end-effector state between consecutive frames. 16 frames yield 16 states and 15 actions (fence-post: 16 posts, 15 gaps).

### The Core Question the Model Learns to Answer

"I'm seeing z_k (visual observation), I'm at s_k (proprioceptive state), I take action a_k (intended displacement) — what will I see next?"

During training, a_k is ground truth from the dataset — what the teleoperator actually did. The model learns world dynamics, not a policy. It never outputs "do this action." It only answers "if you do this action, here's what happens."

### Loss Function

The V-JEPA 2 encoder is frozen. Each frame is encoded independently: z_k = E(x_k) ∈ ℝ^{16×16×1408}.

**Teacher-forcing loss:** The predictor takes the interleaved sequence (a_k, s_k, z_k) and at each timestep k predicts z_{k+1}. The loss is L1 distance between predicted and actual next-frame encoding, averaged over all 15 timesteps. The model always sees ground-truth encoded frames at each step.

**Rollout loss:** At inference, the model must feed its own predictions back as input (no ground truth available), so errors compound. To combat this, a rollout loss feeds the predictor's output back for one additional step (T=2), comparing the final prediction to ground truth. This teaches the model to handle its own imperfect predictions.

**Total loss:** Simply the sum of both — no weighting coefficient.

### Architecture

The predictor is a ~300M parameter transformer (24 layers, 16 heads, 1024 hidden dimension, GELU activations).

**Input processing:** Actions (7D), states (7D), and flattened feature maps (256 tokens × 1408D) each go through separate learned affine transformations to map into the predictor's 1024 hidden dimension. Outputs go through an affine to map back to 1408.

**Positional embeddings:** 3D-RoPE for video patch tokens (spatial + temporal). For action and state tokens, only the temporal component of RoPE is applied — they have no spatial extent.

**Block-causal attention:** Within a timestep, everything can attend to everything (action, state, all spatial patches see each other). Across timesteps, attention is causal — timestep k can see all previous timesteps but not future ones. This is what makes autoregressive prediction possible without information leakage.

### Planning via Cross-Entropy Method (CEM)

At inference, the model doesn't output actions — it searches for them. Given a goal image encoded as z_g, the system finds action sequences that minimize L1 distance between the predictor's imagined future state and z_g.

**How CEM works, concretely:**

Say planning horizon is T=4 steps, each action is 7D (28 numbers total).

**Round 1:** Initialize 4 Gaussians (one per timestep), each 7D, mean=0, variance=1. Sample 200 candidate action sequences. Roll all 200 through the predictor. Compute L1 to goal for each. Take top 20.

**Round 2:** Compute mean and variance of those 20 winners at each timestep. The mean shifts toward what worked (~[0.3, -0.1, 0.5, ...]) and variance tightens. Sample 200 new candidates from these updated Gaussians — now clustered around what worked in round 1.

**Rounds 3–5:** Same process. Distributions get tighter and tighter.

**Final output:** Return the final mean as the chosen action sequence.

The Gaussians aren't learned parameters — they're a search mechanism that progressively narrows from "try everything" to "try variations of what's working." It's essentially evolution: generate, select the fittest, breed from them, repeat.

**Receding horizon control:** Only the first action from the chosen sequence is executed on the robot. Then the system observes the new real frame, re-encodes it, and re-plans from scratch. This keeps the system grounded in reality rather than committed to a plan that will inevitably diverge from what actually happens.

---

## 5. HWM: Hierarchical Planning with Latent World Models

### The Problem with Flat Planning

V-JEPA-AC plans by searching over short action sequences and picking the one that gets closest to the goal. This works for greedy tasks — where every step should move you closer to the goal. But consider pick-and-place: you need to lift the object *up* before moving it sideways. Moving up temporarily increases distance to the goal. A flat planner with a short horizon can't see past the "move up" phase to realize it leads somewhere good. Result: 0% success on pick-and-place without manually provided subgoals.

### The Solution: Plan at Two Timescales

HWM adds a high-level planner (HLP) on top of the existing low-level planner (LLP). The HLP thinks in big jumps — "where should I be in 2 seconds?" — while the LLP handles the fine-grained "how do I get there step by step?"

Both planners operate in the same latent space (same frozen encoder), so the HLP's output can directly serve as a target for the LLP.

### Low-Level World Model (Same as V-JEPA-AC)

Nothing new here. Given current state z_k, proprioceptive state s_k, and primitive action a_k (7D end-effector delta), predict z_{k+1}. Trained with teacher-forcing + rollout loss. This is V-JEPA-AC exactly as described above.

### High-Level World Model Training

This is where the new ideas come in. The high-level model learns to predict over longer time horizons using compressed "macro-actions."

**Step 1 — Pick waypoints from a trajectory.** Take a trajectory with, say, 12 timesteps: T1, T2, ..., T12. Choose N=3 waypoint indices. The first and last are the trajectory endpoints; the middle one is chosen randomly. For example: t₁=T1, t₂=T5, t₃=T12.

**Step 2 — Encode waypoint states.** Use the same frozen encoder E to get latent representations: z_{t₁}, z_{t₂}, z_{t₃}. These live in the same latent space as the low-level model.

**Step 3 — Compress action chunks into macro-actions.** Between each pair of waypoints, there's a variable-length sequence of primitive actions. Between t₁ and t₂: actions a_{T1}, a_{T2}, a_{T3}, a_{T4} (four actions). Between t₂ and t₃: actions a_{T5}, ..., a_{T11} (seven actions). A learned action encoder (a transformer with a CLS token) compresses each chunk into a single latent macro-action vector:

- l_{t₁} = ActionEncoder(a_{T1}, a_{T2}, a_{T3}, a_{T4}) — "everything the robot did to get from waypoint 1 to waypoint 2"
- l_{t₂} = ActionEncoder(a_{T5}, ..., a_{T11}) — "everything the robot did to get from waypoint 2 to waypoint 3"

The transformer handles variable-length action sequences and always outputs one fixed-size vector (4D in the Franka experiments — much smaller than the 7D primitive actions).

**Step 4 — Predict next waypoint representations.** Feed the interleaved sequence (l_{t₁}, z_{t₁}), (l_{t₂}, z_{t₂}) through the high-level predictor. It predicts z_{t₂} and z_{t₃}. Same L1 teacher-forcing loss against the ground-truth encoded waypoints. The predictor and action encoder are trained jointly.

**Why latent macro-actions instead of just using the displacement between waypoints?** Because between two waypoints, the robot might move in a complex, non-greedy path — up, then sideways, then down. A simple delta-pose (endpoint minus start) would collapse all of that into one displacement vector, losing the information about *how* you got there. The learned latent action preserves the structure of the full action sequence in compressed form. The paper shows this leads to better plans that align more closely with expert behavior.

### Hierarchical Planning at Inference Time

**High-level planning:** Starting from current observation z₁ and goal z_g, CEM searches over sequences of H latent macro-actions (4D each). Each candidate sequence is rolled forward through the high-level model. Pick the sequence whose final predicted state is closest to z_g (L1 distance). The intermediate predicted latent states become subgoals: z̃₁, z̃₂, ..., z̃_H.

**Low-level planning:** Take the first subgoal z̃₁. Now do exactly what V-JEPA-AC does — CEM over primitive actions with a short horizon to reach z̃₁. Execute the first action, observe the real next frame, re-plan. Standard MPC loop, but targeting the subgoal instead of the final goal.

**Re-planning:** The agent re-plans every step. When the low-level planner reaches (or gets close to) the first subgoal, the high-level planner runs again from the new position, generating fresh subgoals.

### Why This Works

The high-level planner can "see" that lifting up leads to a state from which moving sideways leads to the goal — even though lifting up temporarily moves away from it. It only needs to predict a few macro-steps ahead (H=2 for Franka), so error accumulation is minimal. The low-level planner handles the precise motor control for each short segment.

The paper shows that for predictions beyond ~1.5 seconds, one step of the high-level model is more accurate than autoregressive rollout of the low-level model, because fewer sequential predictions means less error compounding.

Results: pick-and-place goes from 0% (flat planner, no subgoals) to 70% (hierarchical planner, only final goal image provided).

---

## Key Design Philosophy

The entire JEPA pipeline maintains a clean separation of concerns:

1. **Representation learning** (V-JEPA 2): Learn what matters in video by predicting masked regions in latent space, discarding unpredictable pixel-level noise.
2. **World modeling** (V-JEPA-AC / HWM): Learn how the world responds to actions — at multiple timescales — by predicting next-state representations given current state + action.
3. **Planning** (CEM): Search for actions that lead to desired outcomes by simulating forward through the world model. Hierarchy lets you search coarsely over long horizons and precisely over short ones.

No component tries to do another component's job. The world model doesn't choose actions. The planner doesn't learn representations. The encoder doesn't model actions. Each piece does one thing and does it in representation space, not pixel space.