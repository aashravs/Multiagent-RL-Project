import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EnvConfig:
    env_name: str
    frame_stack: int = 4
    time_limit: int | None = None
    render_mode: str | None = None
    num_envs: int = 1


@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5


@dataclass
class TrainingConfig:
    env: EnvConfig
    ppo: PPOConfig
    log_dir: str = "runs"
    wandb_project: str | None = None
    save_every_steps: int = 100_000
    eval_episodes: int = 5


def load_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    env_cfg = EnvConfig(**raw["env"])
    ppo_cfg = PPOConfig(**raw["ppo"])

    return TrainingConfig(
        env=env_cfg,
        ppo=ppo_cfg,
        log_dir=raw.get("log_dir", "runs"),
        wandb_project=raw.get("wandb_project"),
        save_every_steps=raw.get("save_every_steps", 100_000),
        eval_episodes=raw.get("eval_episodes", 5),
    )
