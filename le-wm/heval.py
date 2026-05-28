"""Hierarchical LeWM evaluation.

Two-level CEM policy:
  L2 CEM: optimises macro-action embedding sequences → produces subgoal embedding.
  L1 CEM: optimises primitive actions to reach the L2 subgoal.

Usage:
    python heval.py --config-name hpusht policy=<hjepa_checkpoint_path>

Flat L1 evaluation is unchanged — use eval.py with a plain JEPA checkpoint.
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import time
from collections import deque
from pathlib import Path

import gymnasium
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from hjepa import HJEPA


# ---------------------------------------------------------------------------
# Shared helpers (identical to eval.py)
# ---------------------------------------------------------------------------

def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


# ---------------------------------------------------------------------------
# L1 subgoal adapter
# ---------------------------------------------------------------------------

class SubgoalAdapter:
    """Wraps JEPA so that get_cost targets a mutable subgoal embedding.

    The L1 CEMSolver calls model.get_cost(expanded_info, candidates). This adapter
    intercepts that call and routes to get_cost_from_emb using the subgoal embedding
    set by the L2 planner at each replan step.
    """

    def __init__(self, jepa):
        self._jepa = jepa
        self.subgoal_emb = None   # (n_envs_replanning, D) — set before each L1 solve

    def set_subgoal(self, emb):
        """emb: (R, D) where R = number of envs currently replanning."""
        self.subgoal_emb = emb

    def get_cost(self, info_dict, action_candidates):
        assert self.subgoal_emb is not None, "call set_subgoal before L1 solver"
        return self._jepa.get_cost_from_emb(info_dict, action_candidates, self.subgoal_emb)

    def __getattr__(self, name):
        # Forward attribute lookups (e.g. .parameters(), .encoder) to the wrapped JEPA.
        return getattr(self._jepa, name)


# ---------------------------------------------------------------------------
# Hierarchical two-level CEM policy
# ---------------------------------------------------------------------------

class HierarchicalWorldModelPolicy(swm.policy.BasePolicy):
    """Two-level CEM policy backed by HJEPA.

    At each replan step:
      1. L2 CEM: CEMSolver calls hjepa.get_cost = get_l2_cost.
         Optimises macro-action embeddings (dim = embed_dim = 192) directly.
         MacroActionEncoder is NOT called at inference — CEM searches in this space.
      2. Extract first subgoal: run rollout_l2 one step with best macro → (R, D).
      3. L1 CEM: CEMSolver calls l1_adapter.get_cost = l1_jepa.get_cost_from_emb(subgoal).
         Optimises primitive action sequences to reach the subgoal embedding.
      4. Buffer receding_horizon steps from L1 plan; pop one per env step.

    CEMSolver.solve(info_dict) returns {'actions': tensor(R, H, action_dim), ...}.
    The L2 solver's action_dim = macro_dim = embed_dim = 192 (fake action space).
    The L1 solver's action_dim = env.action_space.shape[1:] * action_block.
    """

    def __init__(
        self, hjepa, l2_solver, l1_adapter, l1_solver,
        l2_plan_config, l1_plan_config, process, transform,
    ):
        super().__init__()
        self.hjepa = hjepa
        self.l2_solver = l2_solver
        self.l1_adapter = l1_adapter
        self.l1_solver = l1_solver
        self.l2_cfg = l2_plan_config
        self.l1_cfg = l1_plan_config
        self.process = process
        self.transform = transform
        self._action_buffer = None

    def set_env(self, env):
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)

        # Configure L1 solver with the real environment action space.
        self.l1_solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.l1_cfg
        )

        # Configure L2 solver with a fake action space.
        # CEMSolver._action_dim = np.prod(action_space.shape[1:]).
        # We want _action_dim = macro_dim = embed_dim = 192, action_block = 1.
        # So action_space.shape = (1, macro_dim).
        macro_dim = self.hjepa.l2_predictor.pos_embedding.shape[-1]
        l2_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, macro_dim), dtype=np.float32
        )
        self.l2_solver.configure(
            action_space=l2_action_space, n_envs=n_envs, config=self.l2_cfg
        )

        rh = self.l1_cfg.receding_horizon * self.l1_cfg.action_block
        self._action_buffer = [deque(maxlen=rh) for _ in range(n_envs)]

    @torch.no_grad()
    def get_action(self, info_dict):
        info_dict = self._prepare_info(info_dict)
        n_envs = getattr(self.env, 'num_envs', 1)
        device = next(self.hjepa.parameters()).device

        # Identify envs that need replanning (empty action buffer).
        replan_idx = [i for i in range(n_envs) if len(self._action_buffer[i]) == 0]

        if replan_idx:
            idx_t = torch.as_tensor(replan_idx, dtype=torch.long)

            # Slice info_dict to only the envs that need replanning.
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_t]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                else:
                    sliced[k] = v

            # --- L2 CEM: optimise macro-action embedding sequence ---
            l2_out = self.l2_solver(sliced)
            best_macro = l2_out['actions'].to(device)  # (R, H2, macro_dim)

            # Extract first subgoal: one L2 forward step with best macro.
            # sliced['pixels']: (R, T, C, H, W) — take last timestep as starting waypoint.
            cur_pixels = sliced['pixels'].to(device)[:, -1:]   # (R, 1, C, H, W)
            wp0 = self.hjepa.encode_waypoints(cur_pixels)       # (R, 1, D)
            # rollout_l2 expects (B, S, H, D) — add S=1
            pred = self.hjepa.rollout_l2(wp0, best_macro.unsqueeze(1))  # (R, 1, H2, D)
            subgoal_emb = pred[:, 0, 0, :]                              # (R, D)

            # --- L1 CEM: optimise primitives to reach subgoal ---
            self.l1_adapter.set_subgoal(subgoal_emb)
            l1_out = self.l1_solver(sliced)
            actions = l1_out['actions']  # (R, H1, action_dim)

            # Buffer the receding_horizon steps from the L1 plan.
            rh = self.l1_cfg.receding_horizon
            flat_rh = rh * self.l1_cfg.action_block
            plan = actions[:, :rh].reshape(len(replan_idx), flat_rh, -1)
            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        # Pop one action per env.
        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float('nan'))
        for i in range(n_envs):
            if self._action_buffer[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.float().numpy()
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/eval", config_name="hpusht")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # Load HJEPA
    hjepa: HJEPA = swm.wm.utils.load_pretrained(cfg.policy)
    hjepa = hjepa.to("cuda").eval()
    hjepa.requires_grad_(False)
    hjepa.interpolate_pos_encoding = True

    # L2 solver: model=hjepa so solver calls hjepa.get_cost = get_l2_cost
    l2_solver = hydra.utils.instantiate(cfg.l2_solver, model=hjepa)

    # L1 solver: model=l1_adapter so solver calls l1_adapter.get_cost = get_cost_from_emb
    l1_adapter = SubgoalAdapter(hjepa.l1_jepa)
    l1_solver = hydra.utils.instantiate(cfg.l1_solver, model=l1_adapter)

    l2_plan_config = swm.PlanConfig(**cfg.l2_plan_config)
    l1_plan_config = swm.PlanConfig(**cfg.l1_plan_config)

    policy = HierarchicalWorldModelPolicy(
        hjepa=hjepa,
        l2_solver=l2_solver,
        l1_adapter=l1_adapter,
        l1_solver=l1_solver,
        l2_plan_config=l2_plan_config,
        l1_plan_config=l1_plan_config,
        process=process,
        transform=transform,
    )

    world.set_policy(policy)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
