# Multi-Agent RL with PettingZoo + PPO (Shared Policy)

This project demonstrates multi-agent reinforcement learning using **PettingZoo** environments, **Supersuit** wrappers, and **Stable-Baselines3 PPO** with a **shared policy** controlling all agents jointly.

Default example: **Pistonball (cooperative pushing)** with joint-control PPO.

## Structure

- `envs/`
  - `make_env.py` – PettingZoo env factory + Supersuit preprocessing.
- `agents/`
  - `ppo_shared.py` – helper for constructing a shared-policy PPO agent.
- `training/`
  - `train_shared.py` – training entrypoint for shared-policy PPO.
- `eval/`
  - `evaluate.py` – load a trained model and run evaluation episodes.
- `utils/`
  - `config.py` – YAML config loading into typed dataclasses.
  - `wrappers.py` – `JointObsActionWrapper` converting multi-agent env to joint single-agent control.
  - `video.py` – helper for recording episodes to mp4.
- `configs/`
  - `pistonball.yaml` – default training config.
  - `simple_tag.yaml` – alternative environment config.

## Installation

```bash
pip install -r requirements.txt
```

## Training (Shared Policy PPO)

Run shared-policy PPO on Pistonball from the project root:

```bash
python training/train_shared.py --config configs/pistonball.yaml
```

This will:
- Create a PettingZoo `pistonball_v6` parallel environment.
- Apply Supersuit preprocessing (resize, gray, frame-stack, time-limit).
- Wrap as a joint-control Gymnasium env where one PPO policy controls all pistons.
- Train PPO and periodically checkpoint to `runs/pistonball_shared/`.

## Evaluation

After training, evaluate a saved model:

```bash
python eval/evaluate.py \
  --model runs/pistonball_shared/final_model.zip \
  --config configs/pistonball.yaml
```

This will print average returns over a number of episodes.

## Notes

- The code is CPU-friendly and does not assume a GPU.
- WandB/TensorBoard hooks can be added on top of SB3's logging if desired.
- The current implementation focuses on the **shared policy** pathway; an independent-agent PPO variant can be added following a similar pattern with per-agent wrappers.
