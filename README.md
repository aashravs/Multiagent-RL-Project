Multi-Agent Reinforcement Learning Framework (PPO + PettingZoo)

This project implements a modular framework for Multi-Agent Reinforcement Learning (MARL) using Proximal Policy Optimization (PPO).
It supports both shared-policy (one policy controlling all agents) and independent-policy (each agent trains separately) setups.

Two PettingZoo environments are currently integrated:

Pistonball — cooperative continuous-control task

Simple Tag — mixed cooperative/competitive task

A Streamlit dashboard is included for training, evaluation, visualization of metrics, and replay playback.

1. Project Features
1.1 Reinforcement Learning

Shared-policy PPO

Independent-agent PPO

PettingZoo parallel API

SuperSuit wrappers for preprocessing

Joint observation/action wrapper for shared policy

1.2 Tooling & Logging

Evaluation script outputs:

metrics.csv (per-episode returns)

replay.gif (episode visualization)

Streamlit UI for:

Running training

Running evaluation

Viewing metrics plots

Viewing GIF replays

Downloading checkpoints

Example demo included (runs/example_demo/)



2. Installation
2.1 Clone the repository
git clone https://github.com/aashravs/Multiagent-RL-Project.git
cd Multiagent-RL-Project

2.2 Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate    # macOS/Linux

2.3 Install dependencies
pip install -r requirements.txt

3. Quick Demo (No Training Required)

The folder runs/example_demo/ provides:

replay.gif — simple visual demonstration

metrics.csv — example metrics

This allows the UI to run without training any models.

Launch Streamlit:
streamlit run streamlit_app.py


In the sidebar, choose:

Run → example_demo


You will see:

Metrics plot

Episode table

Mean & standard deviation

Replay GIF

This is suitable for demonstrations where compute time is limited.

4. Training
4.1 Shared-Policy PPO (default)

Train Pistonball:

python -m training.train_shared --config configs/pistonball.yaml


Train Simple Tag:

python -m training.train_shared --config configs/simple_tag.yaml

4.2 Independent PPO
python -m training.train_independent --config configs/pistonball.yaml

4.3 Controlling Checkpoint Saving

To avoid large .zip files filling disk space:

Disable saving:

set NO_SAVE=1          # Windows
export NO_SAVE=1       # macOS/Linux


Enable saving (default):

set NO_SAVE=0

5. Evaluation

Evaluation computes per-episode returns and generates a replay animation.

Example:

python -m eval.evaluate \
    --model runs/pistonball_shared/final_model.zip \
    --config configs/pistonball.yaml \
    --episodes 5


Outputs appear under the corresponding run directory:

runs/<run_name>/metrics.csv
runs/<run_name>/replay.gif

6. Streamlit Dashboard

Start:

streamlit run streamlit_app.py

Dashboard Capabilities

Train Pistonball or Simple Tag

Evaluate models

View training/evaluation logs live

Select any run directory

Render replay GIF

Plot metrics (step-based or episode-based)

Display summary tables

Download model checkpoints

7. Methodology
7.1 Environment Processing

PettingZoo parallel API

SuperSuit preprocessing:

resizing

grayscale

normalization

frame stacking

7.2 Shared-Policy PPO

A single PPO network controls all agents.
The environment is wrapped using:

JointObsActionWrapper


which:

concatenates all agent observations

outputs a joint action vector

Advantages:

consistent coordination

fewer parameters

simpler training

7.3 Independent PPO

Each agent:

has its own PPO model

trains separately in a round-robin cycle

Useful for:

competitive tasks

asymmetric roles

7.4 Configuration System

Hyperparameters stored in YAML:

learning rate

batch size

n_steps

gamma

GAE lambda

clip_range

8. Example Results (Demo)

For the included example:

Episodes: 5
Mean return: 548.76
Standard deviation: 230.18


These values are written into:

runs/example_demo/metrics.csv

9. Limitations and Future Work
Limitations

CPU training is slow for large MARL environments

Replay generation for many agents can be expensive

No hyperparameter sweep or tuning tools included

Shared-policy PPO may underperform on competitive tasks

Planned Improvements

WandB integration for experiment logging

Additional PettingZoo environments

Multi-run comparison interface

More stable replay recording for large environments

10. License

This project is licensed under the MIT License.
You are free to use, modify, and redistribute the code.
