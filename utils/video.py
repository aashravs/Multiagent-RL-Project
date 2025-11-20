from __future__ import annotations

import os
from typing import Callable

import imageio
import numpy as np


def record_episode_to_mp4(env, policy_fn: Callable[[np.ndarray], np.ndarray], out_path: str, max_steps: int = 1000):
    """Roll out a policy in a vectorized or single env and save RGB frames to an mp4.

    `policy_fn` takes a batched observation (or single obs) and returns an action.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []
    obs, _ = env.reset()
    step = 0
    while step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if getattr(terminated, "any", None):
            if terminated.any():
                break
        elif isinstance(terminated, (bool, np.bool_)):
            if terminated or truncated:
                break

    if frames:
        imageio.mimsave(out_path, frames, fps=30)
