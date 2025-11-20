import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Multi-Agent RL UI", layout="wide")

st.title("Multi‑Agent RL Control Panel")

st.write("Use this UI to train or evaluate Simple Tag and Pistonball.")

# Helper to run commands and stream output
def run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_container = st.empty()
    full_output = ""
    for line in process.stdout:
        full_output += line
        output_container.text(full_output)
    process.wait()
    return full_output

# TRAINING SECTION
st.header("Training")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Simple Tag")
    if st.button("Train Simple Tag"):
        st.write("Running training…")
        cmd = "python -m training.train_shared --config configs/simple_tag.yaml"
        output = run_command(cmd)
        st.success("Training Finished")

with col2:
    st.subheader("Pistonball")
    if st.button("Train Pistonball"):
        st.write("Running training…")
        cmd = "python -m training.train_shared --config configs/pistonball.yaml"
        output = run_command(cmd)
        st.success("Training Finished")

# EVALUATION SECTION
st.header("Evaluation")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Simple Tag")
    if st.button("Evaluate Simple Tag"):
        st.write("Running evaluation…")
        cmd = "python -m eval.evaluate --config configs/simple_tag.yaml --model runs/simple_tag_shared/final_model.zip"
        output = run_command(cmd)
        st.success("Evaluation Finished")

with col4:
    st.subheader("Pistonball")
    if st.button("Evaluate Pistonball"):
        st.write("Running evaluation…")
        cmd = "python -m eval.evaluate --config configs/pistonball.yaml --model runs/pistonball_shared/final_model.zip"
        output = run_command(cmd)
        st.success("Evaluation Finished")

st.write("---")
st.write("Streamlit UI Ready.")
