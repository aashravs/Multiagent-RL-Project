# evaluate.py
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import List

import imageio
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from utils.config import load_config
from envs.make_env import make_parallel_env
from utils.wrappers import JointObsActionWrapper


def try_render_frame(env) -> "List[np.ndarray] | None":
    """
    Try one environment step-free render to check if rgb frames are available.
    Returns an empty list if render is supported but no frames captured,
    or None if render isn't supported at all.
    """
    try:
        frame = env.render(mode="rgb_array")
        if frame is None:
            return []
        return [frame]
    except Exception:
        # render not supported / not in this environment
        return None


def evaluate(model_path: str, config_path: str, episodes: int = 5, save_dir: str | None = None):
    cfg = load_config(config_path)
    parallel_env = make_parallel_env(cfg.env)
    env = JointObsActionWrapper(parallel_env)

    model = PPO.load(model_path)

    # Determine run directory to save artifacts
    model_path_obj = pathlib.Path(model_path)
    if save_dir:
        run_dir = pathlib.Path(save_dir)
    else:
        run_dir = model_path_obj.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    returns = []
    all_frames = []  # will store frames for the first episode only (sample)
    capture_frames = None  # None = unknown, [] = supported but empty, list = captured frames

    for ep in range(episodes):
        # If env.reset() returns tuple or dict like Gymnasium API, handle accordingly
        reset_ret = env.reset()
        # JointObsActionWrapper might return obs or (obs, info)
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
            obs, _ = reset_ret
        else:
            obs = reset_ret

        done = False
        ep_ret = 0.0
        frames = []
        # If we haven't checked render capability yet, try it
        if capture_frames is None:
            capture_frames = try_render_frame(env)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_ret = env.step(action)
            # support both Gymnasium and older APIs
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, _info = step_ret
            elif len(step_ret) == 4:
                obs, reward, done_flag, _info = step_ret
                terminated = done_flag
                truncated = False
            else:
                # unexpected return signature
                raise RuntimeError("Unexpected env.step() return signature")

            done = bool(terminated or truncated)
            ep_ret += float(reward)

            # capture frames only if render returns arrays
            if capture_frames is not None:
                try:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    # render not available (race condition), stop capturing
                    capture_frames = None

        returns.append(ep_ret)
        print(f"Episode {ep + 1}: return={ep_ret:.2f}")

        # store only first episode frames as a sample replay
        if ep == 0 and frames:
            all_frames = frames

    avg = float(np.mean(returns))
    std = float(np.std(returns))
    print(f"Average return over {episodes} episodes: {avg:.2f}")

    # Write metrics.csv with per-episode returns and summary row
    metrics_rows = [{"episode": i + 1, "return": float(r)} for i, r in enumerate(returns)]
    df = pd.DataFrame(metrics_rows)
    summary = pd.DataFrame([{"episode": "mean", "return": avg}, {"episode": "std", "return": std}])
    metrics_path = run_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    # append summary rows
    with open(metrics_path, "a") as f:
        summary.to_csv(f, header=False, index=False)
    print(f"Wrote metrics to {metrics_path}")

    # Save replay gif if we captured any frames
    if all_frames:
        try:
            gif_path = run_dir / "replay.gif"
            imageio.mimsave(gif_path, all_frames, fps=15)
            print(f"Wrote replay gif to {gif_path}")
        except Exception as e:
            print(f"Failed to write replay gif: {e}")
    else:
        print("No frames captured (env.render not supported or returned no frames). No replay.gif created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for training")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", type=str, default=None, help="Optional output directory for artifacts (overrides model parent)")
    args = parser.parse_args()

    evaluate(args.model, args.config, args.episodes, args.out)


if __name__ == "__main__":
    main()
