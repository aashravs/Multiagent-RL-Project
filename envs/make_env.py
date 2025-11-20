from __future__ import annotations

from typing import Callable, Dict
import supersuit as ss

from pettingzoo.butterfly import pistonball_v6, cooperative_pong_v5
from pettingzoo.mpe import simple_tag_v3

from utils.config import EnvConfig


ENV_REGISTRY: Dict[str, Callable[..., object]] = {
    "pistonball_v6": pistonball_v6.parallel_env,
    "simple_tag_v3": simple_tag_v3.parallel_env,
    "cooperative_pong_v5": cooperative_pong_v5.parallel_env,
}


PIXEL_ENVS = {"pistonball_v6", "cooperative_pong_v5"}


def make_parallel_env(cfg: EnvConfig):
    """Create a correct PettingZoo parallel-env with ONLY safe wrappers."""

    if cfg.env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_name: {cfg.env_name}")

    env = ENV_REGISTRY[cfg.env_name](render_mode=cfg.render_mode)

    # -------------------------------
    # PIXEL ENV WRAPPERS
    # -------------------------------
    if cfg.env_name in PIXEL_ENVS:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_skip_v0(env, 2)

        if getattr(cfg, "frame_stack", None) and cfg.frame_stack > 1:
            env = ss.frame_stack_v1(env, cfg.frame_stack)

    # -------------------------------
    # VECTOR ENV WRAPPERS (simple_tag)
    # -------------------------------
    else:
        # Do NOT normalize here — simple_tag has infinite bounds
        # Do NOT resize
        pass

    # apply dtype for ALL envs
    env = ss.dtype_v0(env, dtype="float32")

    # ONLY pixel envs can be normalized safely
    if cfg.env_name in PIXEL_ENVS:
        env = ss.normalize_obs_v0(env, env_min=0.0, env_max=1.0)

    # padding is safe for all
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    return env
