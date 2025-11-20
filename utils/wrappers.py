from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete


class JointObsActionWrapper(gym.Env):
    """Wrap a PettingZoo parallel env as a single-agent joint-control env.

    Observations from all agents are concatenated into a single flat vector.
    Actions are a MultiDiscrete: one discrete action per agent.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, parallel_env):
        super().__init__()
        self.env = parallel_env
        self.agent_ids: List[str] = list(self.env.possible_agents)

        # Assume all agents share the same observation and action space.
        sample_agent = self.agent_ids[0]
        obs_space = self.env.observation_space(sample_agent)
        act_space = self.env.action_space(sample_agent)

        if not isinstance(obs_space, Box):
            raise ValueError("JointObsActionWrapper currently expects Box observations.")
        if hasattr(act_space, "n"):
            # Discrete
            self.action_space = MultiDiscrete([act_space.n] * len(self.agent_ids))
        else:
            # Continuous shared Box
            low = np.concatenate([act_space.low for _ in self.agent_ids], axis=0)
            high = np.concatenate([act_space.high for _ in self.agent_ids], axis=0)
            self.action_space = Box(low=low, high=high, dtype=act_space.dtype)

        flat_dim = int(np.prod(obs_space.shape)) * len(self.agent_ids)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def _flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        ordered = [obs[a].reshape(-1) for a in self.agent_ids]
        return np.concatenate(ordered, axis=0).astype(np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        flat = self._flatten_obs(obs)
        return flat, info

    def step(self, action):
        # Convert joint action into dict per agent
        if isinstance(self.action_space, MultiDiscrete):
            assert len(action) == len(self.agent_ids)
            act_dict = {aid: int(a) for aid, a in zip(self.agent_ids, action)}
        else:
            # Box action: split vector equally
            act_dict = {}
            per_dim = action.shape[0] // len(self.agent_ids)
            for i, aid in enumerate(self.agent_ids):
                act_dict[aid] = action[i * per_dim : (i + 1) * per_dim]

        obs, rewards, terminations, truncations, infos = self.env.step(act_dict)
        flat_obs = self._flatten_obs(obs)

        # Aggregate rewards and done flags (cooperative setting)
        reward = float(sum(rewards.values()))
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        info = {"per_agent_rewards": rewards, "raw_infos": infos}
        return flat_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
