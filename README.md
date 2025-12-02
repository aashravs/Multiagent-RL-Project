Multi-Agent Reinforcement Learning: Shared-Policy PPO Framework

This project implements a clean, modular Multi-Agent Reinforcement Learning (MARL) framework using PPO (Proximal Policy Optimization) for both shared-policy and independent-policy setups.
It supports two PettingZoo environments:

Pistonball (cooperative, continuous dynamics)

Simple Tag (mixed cooperative/competitive)

A full Streamlit UI is provided to train, evaluate, visualize metrics, and play stored replays — making the entire project presentable and interactive for demonstration.

Features

Shared PPO training (one neural policy controlling all agents)

Independent PPO training (each agent has its own policy)

PettingZoo + SuperSuit wrappers

Training & evaluation pipelines

Automatic metric logging (metrics.csv)

Replay generation (replay.gif)

Streamlit UI dashboard for:

launching training/evaluation

viewing metrics

viewing downloadable GIF replays

model selection and override paths

Example demo included (runs/example_demo/)

replay.gif

metrics.csv

Works instantly without training anything

Repository Structure
.
├── agents/                 # PPO implementations (shared & independent)
│   ├── ppo_shared.py
│   └── ppo_independent.py
│
├── configs/                # YAML configuration files
│   ├── pistonball.yaml
│   └── simple_tag.yaml
│
├── envs/                   # PettingZoo environment factory
│   └── make_env.py
│
├── eval/                   # Evaluation script
│   └── evaluate.py
│
├── runs/
│   ├── example_demo/       # Small demo (pre-generated replay + metrics)
│   │   ├── replay.gif
│   │   └── metrics.csv
│   ├── pistonball_shared/
│   └── simple_tag_shared/
│
├── scripts/
│   └── make_demo_replay.py # Helper script for generating tiny demo assets
│
├── training/
│   ├── train_shared.py
│   └── train_independent.py
│
├── utils/                  # Buffers, wrappers, config loader, video writer
│
└── streamlit_app.py        # Full Web UI

Installation & Setup
1. Clone the repository
git clone https://github.com/aashravs/Multiagent-RL-Project.git
cd Multiagent-RL-Project

2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

Quick Demo (No Training Required)

If you want to show the project immediately, this demo folder has valid metrics + a working replay:

runs/example_demo/
    - replay.gif  
    - metrics.csv

Launch UI
streamlit run streamlit_app.py


Then in the sidebar, select:

Run → example_demo


You will instantly see:

a replay animation

plotted metrics

summary statistics

Perfect for demonstrations.

Training
Train Shared-Policy PPO on Pistonball
python -m training.train_shared --config configs/pistonball.yaml

Train Shared-Policy PPO on Simple Tag
python -m training.train_shared --config configs/simple_tag.yaml

About NO_SAVE

To prevent filling your disk with hundreds of checkpoints, the training script disables model.save() when the environment variable is set:

set NO_SAVE=1        # Windows
export NO_SAVE=1     # macOS/Linux


This keeps your storage safe.

Evaluation (also generates replay.gif)
Evaluate a trained model
python -m eval.evaluate \
    --model runs/pistonball_shared/final_model.zip \
    --config configs/pistonball.yaml \
    --episodes 5


This automatically generates:

runs/<run_name>/metrics.csv
runs/<run_name>/replay.gif

Streamlit UI Guide

Launch with:

streamlit run streamlit_app.py

UI Capabilities

Train models (Pistonball / Simple Tag)

Evaluate models

Real-time logs while training

Replay viewer

Metrics visualizer

Downloadable checkpoint button

Model override path input (load any .zip)

Run selector

Pick any folder inside runs/ to view:

metrics plot

metrics table

mean/stdev return

replay GIF

file list

Methodology & Architecture
Environments

PettingZoo environments wrapped in SuperSuit:

frame stacking

resizing

normalization

vectorization

Shared Policy PPO

All agents share:

same neural network

same optimizer

same PPO update

Independent PPO

Each agent trains individually:

slower

more flexible

better for competitive tasks

Config-Driven System

All hyperparameters live in configs/*.yaml:

learning rate

batch size

gamma

gae lambda

PPO clip range

total timesteps

Outputs

Every evaluation writes:

metrics.csv    # episode returns
replay.gif     # visual rollout

Example Results (Demo)

The included demo has:

Mean return: 548.76
Std: 230.18
Episodes: 5


Replay: a moving white square (tiny handcrafted environment used only for preview).

Limitations & Future Work

Training PPO for many timesteps can be slow on CPU.

Replay recording for 20+ agents can get heavy.

No hyperparameter sweeps yet.

No multi-environment dashboard support yet.

Planned:

Independent vs shared performance comparison

More PettingZoo environments

WandB experiment tracking option

License

MIT License.
Feel free to use, modify, and build upon this project.