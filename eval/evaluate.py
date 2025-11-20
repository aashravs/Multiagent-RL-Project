from __future__ import annotations

import argparse
import os

import numpy as np

from stable_baselines3 import PPO

from utils.config import load_config
from envs.make_env import make_parallel_env
from utils.wrappers import JointObsActionWrapper


def evaluate(model_path: str, config_path: str, episodes: int = 5):
    cfg = load_config(config_path)
    parallel_env = make_parallel_env(cfg.env)
    env = JointObsActionWrapper(parallel_env)

    model = PPO.load(model_path)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
        returns.append(ep_ret)
        print(f"Episode {ep + 1}: return={ep_ret:.2f}")

    print(f"Average return over {episodes} episodes: {np.mean(returns):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for training")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.model, args.config, args.episodes)


if __name__ == "__main__":
    main()
