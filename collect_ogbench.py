"""Collect OGBench expert demonstrations and save as HDF5.

Usage:
    python collect_ogbench.py --env cube --episodes 1000
    python collect_ogbench.py --env scene --episodes 1000

The resulting files are written to $STABLEWM_HOME/datasets/ and can
be used immediately for training.
"""

import argparse
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

ENVS = {
    "cube": {
        "env_id": "swm/OGBCube-v0",
        "env_kwargs": {"env_type": "single", "ob_type": "states"},
        "output_name": "ogbench--cube_single_expert.h5",
    },
    "scene": {
        "env_id": "swm/OGBScene-v0",
        "env_kwargs": {"ob_type": "states"},
        "output_name": "ogbench--scene_single_expert.h5",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS.keys(), required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ENVS[args.env]
    output_path = get_cache_dir(sub_folder="datasets") / cfg["output_name"]

    if output_path.exists():
        print(f"Dataset already exists at {output_path}. Delete it first to recollect.")
        return

    print(f"Collecting {args.episodes} episodes for {args.env} → {output_path}")

    world = swm.World(
        cfg["env_id"],
        num_envs=1,
        image_shape=(224, 224),
        **cfg["env_kwargs"],
    )

    policy = ExpertPolicy(policy_type="markov_oracle", action_noise=0.1, seed=args.seed)
    world.set_policy(policy)

    world.collect(
        path=output_path,
        episodes=args.episodes,
        seed=args.seed,
        format="hdf5",
    )

    world.close()
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
