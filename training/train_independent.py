from __future__ import annotations

import argparse

from envs.make_env import make_parallel_env
from utils.config import load_config
from agents.ppo_independent import make_independent_ppo_agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    def env_fn():
        return make_parallel_env(cfg.env)

    # Probe env for agent ids
    env = env_fn()
    agent_ids = list(env.possible_agents)
    env.close()

    agents = make_independent_ppo_agents(env_fn, agent_ids)

    # Basic round-robin training for each agent
    # This is intentionally simple: each agent is trained in isolation
    # while others follow random policies (see ppo_independent.make_independent_ppo_agents).

    for aid, model in agents.items():
        print(f"Training independent PPO for agent: {aid}")
        model.learn(total_timesteps=cfg.ppo.total_timesteps)
        model.save(f"{cfg.log_dir}/independent_{aid}")


if __name__ == "__main__":
    main()
