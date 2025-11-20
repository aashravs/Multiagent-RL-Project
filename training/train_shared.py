from __future__ import annotations

import argparse
import os
from typing import Callable

from utils.config import load_config
from envs.make_env import make_parallel_env
from utils.wrappers import JointObsActionWrapper
from agents.ppo_shared import make_shared_ppo


def build_env_fn(config_path: str) -> Callable:
    cfg = load_config(config_path)

    def _make():
        parallel_env = make_parallel_env(cfg.env)
        joint_env = JointObsActionWrapper(parallel_env)
        return joint_env

    return _make, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    env_fn, cfg = build_env_fn(args.config)

    os.makedirs(cfg.log_dir, exist_ok=True)

    model = make_shared_ppo(
        env_fn,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
    )

    # NO CHECKPOINTS → prevents disk spam
    model.learn(
        total_timesteps=cfg.ppo.total_timesteps,
        callback=None,
    )

    model.save(os.path.join(cfg.log_dir, "final_model"))


if __name__ == "__main__":
    main()
