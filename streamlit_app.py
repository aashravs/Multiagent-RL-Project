# streamlit_app.py
import streamlit as st
from pathlib import Path
import subprocess
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Multi-Agent RL Control Panel", layout="wide")
st.title("Multi-Agent RL Control Panel")
st.write("Use this UI to train or evaluate Simple Tag and Pistonball.")

# ---- Config ----
RUNS_DIR = Path("runs")
EXPERIMENT_PREFIXES = ["pistonball", "simple_tag"]  # used for nicer display if you want

# ---- Helpers ----
def list_runs():
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])

def stream_subprocess(cmd, output_area, stop_flag):
    """Run command and stream stdout to output_area (st.empty())"""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    buf = ""
    for line in proc.stdout:
        buf += line
        output_area.text(buf)
        if stop_flag["stop"]:
            proc.terminate()
            break
    proc.wait()
    return buf

def plot_metrics_if_exists(run_path, container):
    """
    Robust metric plotting.
    Accepts either:
      - CSV with columns: step, mean_reward, std_reward  (old behavior)
      - CSV with columns: episode, return (your evaluate.py output)
    Handles summary rows like 'mean'/'std' by ignoring them for plotting and using numeric rows to compute statistics.
    """
    metrics_csv = run_path / "metrics.csv"
    if not metrics_csv.exists():
        return False

    try:
        df = pd.read_csv(metrics_csv, dtype=str)  # read as strings to handle mixed rows
    except Exception as e:
        container.text(f"Failed to read metrics.csv: {e}")
        return False

    # Case A: expected old format "step,mean_reward"
    if {"step", "mean_reward"}.issubset(df.columns):
        try:
            df_num = df.dropna(subset=["step", "mean_reward"]).astype({"step": float, "mean_reward": float})
            fig, ax = plt.subplots()
            ax.plot(df_num['step'], df_num['mean_reward'], label="mean_reward")
            if 'std_reward' in df_num.columns:
                # convert if present
                ax.fill_between(df_num['step'],
                                df_num['mean_reward'] - df_num['std_reward'].astype(float),
                                df_num['mean_reward'] + df_num['std_reward'].astype(float),
                                alpha=0.2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.legend()
            container.pyplot(fig)
            return True
        except Exception as e:
            container.text(f"Failed to plot step/mean_reward metrics: {e}")
            return False

    # Case B: your evaluate.py format: "episode,return"
    if {"episode", "return"}.issubset(df.columns):
        # try to separate numeric episode rows from summary rows like 'mean' or 'std'
        df_plot = df.copy()
        # coerce episode -> numeric where possible, other rows become NaN
        df_plot['episode_num'] = pd.to_numeric(df_plot['episode'], errors='coerce')
        df_plot['return_num'] = pd.to_numeric(df_plot['return'], errors='coerce')

        numeric_rows = df_plot[df_plot['episode_num'].notnull() & df_plot['return_num'].notnull()]
        if not numeric_rows.empty:
            try:
                fig, ax = plt.subplots()
                ax.plot(numeric_rows['episode_num'], numeric_rows['return_num'], marker='o', linestyle='-')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Return")
                ax.set_title("Per-episode returns")
                container.pyplot(fig)

                # show small table of per-episode returns
                show_df = numeric_rows[['episode_num', 'return_num']].rename(columns={'episode_num': 'episode', 'return_num': 'return'})
                container.dataframe(show_df.reset_index(drop=True))

                # compute mean/std either from numeric rows or from summary rows if present
                mean_row = df_plot[df_plot['episode'].str.lower() == 'mean']
                std_row = df_plot[df_plot['episode'].str.lower() == 'std']

                if not mean_row.empty:
                    mean_val = float(pd.to_numeric(mean_row['return'].iloc[0], errors='coerce'))
                else:
                    mean_val = float(numeric_rows['return_num'].mean())

                if not std_row.empty:
                    std_val = float(pd.to_numeric(std_row['return'].iloc[0], errors='coerce'))
                else:
                    std_val = float(numeric_rows['return_num'].std())

                container.markdown(f"**Mean return:** {mean_val:.2f}  \n**Std:** {std_val:.2f}")
                return True
            except Exception as e:
                container.text(f"Failed to plot episode/return metrics: {e}")
                return False
        else:
            container.text("metrics.csv found but contains no numeric episode rows to plot.")
            return True

    # Unknown format
    container.text("metrics.csv present but in an unexpected format. Expected columns: (step,mean_reward) or (episode,return).")
    return False

# ---- Session state for run lock ----
if "busy" not in st.session_state:
    st.session_state.busy = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = {"stop": False}

# ---- Sidebar: experiments + options ----
st.sidebar.title("Experiment Controls")
available_runs = list_runs()
run_names = [p.name for p in available_runs]
selected_run = st.sidebar.selectbox("Choose run (for metrics / replay / checkpoint)", ["<none>"] + run_names)
st.sidebar.markdown("---")
st.sidebar.write("Quick actions")
if selected_run != "<none>":
    sel_path = RUNS_DIR / selected_run
    ckpt = sel_path / "final_model.zip"
    replay = sel_path / "replay.gif"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            st.sidebar.download_button("Download checkpoint", f, file_name=f"{selected_run}_checkpoint.zip")
    else:
        st.sidebar.write("No checkpoint found")
    if replay.exists():
        st.sidebar.image(str(replay), caption="Replay (sample)", use_column_width=True)
    else:
        st.sidebar.write("No replay.gif found")

st.sidebar.markdown("---")
st.sidebar.write("Environment")
env_choice = st.sidebar.selectbox("Environment", ["pistonball", "simple_tag"])
st.sidebar.write("Model path (override)")
model_path = st.sidebar.text_input("Model path (optional)", value="")

# ---- Main layout: Training / Evaluation columns ----
train_col, eval_col = st.columns(2)

# ----- TRAINING -----
with train_col:
    st.header("Training")
    st.subheader("Simple Tag")
    if st.session_state.busy:
        st.button("Train Simple Tag", disabled=True)
    else:
        if st.button("Train Simple Tag"):
            st.session_state.busy = True
            st.session_state.stop_flag = {"stop": False}
            output_area = st.empty()
            cmd = "python -m training.train_shared --config configs/simple_tag.yaml"
            # run in thread so UI doesn't freeze
            def run_and_unlock():
                try:
                    stream_subprocess(cmd, output_area, st.session_state.stop_flag)
                    output_area.text(output_area._value + "\n\nTraining Finished.")
                finally:
                    st.session_state.busy = False
            threading.Thread(target=run_and_unlock, daemon=True).start()

    st.subheader("Pistonball")
    if st.session_state.busy:
        st.button("Train Pistonball", disabled=True)
    else:
        if st.button("Train Pistonball"):
            st.session_state.busy = True
            st.session_state.stop_flag = {"stop": False}
            output_area = st.empty()
            cmd = "python -m training.train_shared --config configs/pistonball.yaml"
            def run_and_unlock():
                try:
                    stream_subprocess(cmd, output_area, st.session_state.stop_flag)
                    output_area.text(output_area._value + "\n\nTraining Finished.")
                finally:
                    st.session_state.busy = False
            threading.Thread(target=run_and_unlock, daemon=True).start()

# ----- EVALUATION -----
with eval_col:
    st.header("Evaluation")
    st.subheader("Simple Tag")
    eval_model = ""
    if model_path.strip() != "":
        eval_model = model_path.strip()
    else:
        candidate = RUNS_DIR / "simple_tag_shared" / "final_model.zip"
        if candidate.exists():
            eval_model = str(candidate)

    if st.session_state.busy:
        st.button("Evaluate Simple Tag", disabled=True)
    else:
        if st.button("Evaluate Simple Tag"):
            if not eval_model:
                st.error("No model found. Set a model path in the sidebar or ensure runs/simple_tag_shared/final_model.zip exists.")
            else:
                st.session_state.busy = True
                st.session_state.stop_flag = {"stop": False}
                out = st.empty()
                cmd = f"python -m eval.evaluate --config configs/simple_tag.yaml --model {eval_model}"
                def run_eval():
                    try:
                        stream_subprocess(cmd, out, st.session_state.stop_flag)
                        out.text(out._value + "\n\nEvaluation Finished.")
                    finally:
                        st.session_state.busy = False
                threading.Thread(target=run_eval, daemon=True).start()

    st.subheader("Pistonball")
    eval_model_pb = ""
    if model_path.strip() != "":
        eval_model_pb = model_path.strip()
    else:
        candidate = RUNS_DIR / "pistonball_shared" / "final_model.zip"
        if candidate.exists():
            eval_model_pb = str(candidate)

    if st.session_state.busy:
        st.button("Evaluate Pistonball", disabled=True)
    else:
        if st.button("Evaluate Pistonball"):
            if not eval_model_pb:
                st.error("No model found. Set a model path in the sidebar or ensure runs/pistonball_shared/final_model.zip exists.")
            else:
                st.session_state.busy = True
                st.session_state.stop_flag = {"stop": False}
                out = st.empty()
                cmd = f"python -m eval.evaluate --config configs/pistonball.yaml --model {eval_model_pb}"
                def run_eval():
                    try:
                        stream_subprocess(cmd, out, st.session_state.stop_flag)
                        out.text(out._value + "\n\nEvaluation Finished.")
                    finally:
                        st.session_state.busy = False
                threading.Thread(target=run_eval, daemon=True).start()

# ---- Bottom area: metrics and replay for selected run ----
st.write("---")
st.header("Artifacts & Metrics for selected run")
if selected_run != "<none>":
    run_path = RUNS_DIR / selected_run
    metrics_area = st.empty()
    plotted = plot_metrics_if_exists(run_path, metrics_area)
    if not plotted:
        metrics_area.text("No metrics.csv to plot in this run (expected: runs/<run>/metrics.csv).")

    # show replay if exists
    replay_path = run_path / "replay.gif"
    if replay_path.exists():
        st.image(str(replay_path), caption="Replay (sample)", use_column_width=True)
        with open(replay_path, "rb") as f:
            st.download_button("Download replay.gif", f, file_name=f"{selected_run}_replay.gif")
    else:
        st.write("No replay.gif found for this run. Make evaluate.py save one to runs/<run>/replay.gif")

    # quick summary files list
    files = [p.name for p in run_path.iterdir()] if run_path.exists() else []
    st.write("Files in run folder:", files)
else:
    st.write("Select a run from the sidebar to view metrics and replay")

# ---- Stop/kill button (if something is running) ----
if st.session_state.busy:
    if st.button("STOP current job"):
        st.session_state.stop_flag["stop"] = True
        st.success("Stop requested. The process will terminate shortly.")
else:
    st.write("No job running.")

st.write("---")
st.write("Streamlit UI Ready.")
