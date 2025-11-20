from __future__ import annotations

from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def make_shared_ppo(env_fn, learning_rate: float, n_steps: int, batch_size: int, gamma: float,
                    gae_lambda: float, clip_range: float, ent_coef: float, vf_coef: float,
                    device: str = "auto") -> PPO:
    """Create a shared-policy PPO agent for a joint-control env.

    `env_fn` should be a callable returning a gym.Env (e.g. JointObsActionWrapper).
    """

    def _thunk():
        return env_fn()

    vec_env: VecEnv = DummyVecEnv([_thunk])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1,
        device=device,
    )
    return model
