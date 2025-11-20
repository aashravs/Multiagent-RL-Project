from __future__ import annotations

from typing import Dict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_independent_ppo_agents(env_fn, agent_ids) -> Dict[str, PPO]:
    """Create a PPO instance per agent using its own observation/action spaces.

    `env_fn` should construct a fresh PettingZoo parallel env.
    We then wrap each agent with a simple single-agent view inside DummyVecEnv.
    This is a basic, educational implementation and not heavily optimized.
    """

    agents: Dict[str, PPO] = {}

    for agent_id in agent_ids:
        def make_single_agent_env(agent_id=agent_id):  # default capture
            # Create a new env each time
            parallel_env = env_fn()

            class SingleAgentEnv:
                def __init__(self, env, aid):
                    self.env = env
                    self.agent_id = aid
                    self.observation_space = env.observation_space(aid)
                    self.action_space = env.action_space(aid)

                def reset(self):
                    obs, _ = self.env.reset()
                    return obs[self.agent_id], {}

                def step(self, action):
                    act_dict = {a: self.env.action_space(a).sample() for a in self.env.agents}
                    act_dict[self.agent_id] = action
                    obs, rewards, terminations, truncations, infos = self.env.step(act_dict)
                    ob = obs[self.agent_id]
                    r = float(rewards[self.agent_id])
                    terminated = bool(terminations[self.agent_id])
                    truncated = bool(truncations[self.agent_id])
                    done = terminated or truncated
                    info = infos.get(self.agent_id, {})
                    return ob, r, terminated, truncated, info

                def render(self):
                    return self.env.render()

                def close(self):
                    self.env.close()

            return SingleAgentEnv(parallel_env, agent_id)

        vec_env = DummyVecEnv([make_single_agent_env])
        model = PPO("MlpPolicy", vec_env, verbose=1)
        agents[agent_id] = model

    return agents
