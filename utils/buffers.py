from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TrajectoryStep:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray


class PerAgentReplayBuffer:
    """Simple on-policy-like buffer storing trajectories per agent.

    This is primarily useful for independent PPO style updates where each
    agent has its own policy but we still want to batch experience.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.storage: Dict[str, List[TrajectoryStep]] = {}

    def add(self, agent_id: str, step: TrajectoryStep) -> None:
        buf = self.storage.setdefault(agent_id, [])
        buf.append(step)
        if len(buf) > self.max_size:
            buf.pop(0)

    def get_agent_trajectory(self, agent_id: str) -> List[TrajectoryStep]:
        return list(self.storage.get(agent_id, []))

    def clear(self) -> None:
        self.storage.clear()
