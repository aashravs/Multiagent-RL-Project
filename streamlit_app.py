# streamlit_app.py
"""
Multi-Agent RL Control Panel (fixed for duplicate IDs, safe rerun, optional demo gif)
Drop this file into your project root and run: streamlit run streamlit_app.py
"""

import streamlit as st
import subprocess
import threading
import queue
import time
import os
import csv
from typing import List, Optional

# -------------------------
# Configuration / commands
# -------------------------
# Adjust these commands to match your scripts. Use list form (no shell=True).
CMD_EVAL_PISTON = ["python", "eval/evaluate.py", "--env", "pistonball", "--shared"]
CMD_TRAIN_PISTON = ["python", "training/train_shared.py", "--env", "pistonball"]
CMD_EVAL_SIMPLE = ["python", "eval/evaluate.py", "--env", "simple_tag", "--independent"]
CMD_TRAIN_SIMPLE = ["python", "training/train_independent.py", "--env", "simple_tag"]

RUNS_DIR = "runs"  # directory where each run creates a folder with metrics.csv, replay.gif, model.zip

# -------------------------
# Compatibility-safe rerun
# -------------------------
def safe_rerun():
    """
    Try a version-robust rerun:
    1. st.experimental_rerun() if present
    2. try assigning to st.query_params (modern API) or st.set_query_params
    3. fallback: flip a session_state sentinel
    Must be called from main thread only.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    # Try modern query params API if available
    try:
        ts = str(int(time.time() * 1000))
        try:
            # Most modern Streamlit expose st.query_params as a dict-like property
            st.query_params = {"_rerun": ts}
            return
        except Exception:
            # Fallback to set_query_params if available
            if hasattr(st, "set_query_params"):
                try:
                    st.set_query_params(_rerun=ts)
                    return
                except Exception:
                    pass
    except Exception:
        pass

    # Last resort: toggle a session_state value
    try:
        st.session_state["_last_rerun_ts"] = time.time()
    except Exception:
        pass

# -------------------------
# Session state bootstrap
# -------------------------
def ensure_state():
    if "busy" not in st.session_state:
        st.session_state.busy = False
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = None
    if "worker_thread" not in st.session_state:
        st.session_state.worker_thread = None
    if "stop_holder" not in st.session_state:
        st.session_state.stop_holder = None
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "proc_info" not in st.session_state:
        st.session_state.proc_info = None
    if "_last_rerun_ts" not in st.session_state:
        st.session_state._last_rerun_ts = None

ensure_state()

# -------------------------
# Worker thread for subprocess
# -------------------------
def _subproc_worker(cmd: List[str], q: queue.Queue, stop_holder: dict, run_tag: str):
    """
    Background worker - starts subprocess, streams stdout/stderr lines into q,
    monitors stop_holder['stop'] and attempts to terminate & kill as needed.
    NEVER calls Streamlit APIs.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        q.put(f"[worker] started pid={proc.pid} tag={run_tag}")
        if proc.stdout is not None:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                line = raw_line.rstrip("\n")
                q.put(line)
                if stop_holder.get("stop"):
                    q.put("[worker] stop requested -> terminating subprocess")
                    try:
                        proc.terminate()
                    except Exception as e:
                        q.put(f"[worker] error during terminate: {e}")
                    break
        # give it a moment, then kill if necessary and stop requested
        if proc.poll() is None:
            try:
                proc.wait(timeout=3)
            except Exception:
                if stop_holder.get("stop") and proc.poll() is None:
                    try:
                        proc.kill()
                        q.put("[worker] process killed")
                    except Exception as e:
                        q.put(f"[worker] error during kill: {e}")
        q.put(f"[worker] finished returncode={proc.returncode if proc is not None else 'N/A'}")
    except Exception as e:
        q.put(f"[worker] exception: {repr(e)}")
    finally:
        q.put("__WORKER_DONE__")


# -------------------------
# Control helpers
# -------------------------
def start_job(cmd: List[str], run_tag: str):
    if st.session_state.busy:
        st.warning("A job is already running. Stop it first.")
        return
    q = queue.Queue()
    stop_holder = {"stop": False}
    worker = threading.Thread(target=_subproc_worker, args=(cmd, q, stop_holder, run_tag), daemon=True)
    worker.start()

    st.session_state.log_queue = q
    st.session_state.worker_thread = worker
    st.session_state.stop_holder = stop_holder
    st.session_state.busy = True
    st.session_state.last_run = run_tag
    st.session_state.proc_info = {"cmd": cmd, "started_at": time.time()}

    # immediate UI rerun so disabled flags appear
    safe_rerun()


def request_stop():
    if not st.session_state.busy:
        return
    if st.session_state.stop_holder is not None:
        st.session_state.stop_holder["stop"] = True
    # stay busy until worker reports done
    st.session_state.busy = True
    safe_rerun()


def drain_log_queue(max_lines=5000) -> bool:
    """
    Drain available lines from the log queue into session_state.log_lines.
    Return True if worker signaled done.
    """
    q = st.session_state.log_queue
    if q is None:
        return False
    finished = False
    try:
        while True:
            line = q.get_nowait()
            if line == "__WORKER_DONE__":
                finished = True
                break
            st.session_state.log_lines.append(line)
            # safety cap
            if len(st.session_state.log_lines) > max_lines:
                st.session_state.log_lines = st.session_state.log_lines[-max_lines:]
    except queue.Empty:
        pass
    return finished

# -------------------------
# Utility: list runs
# -------------------------
def list_runs(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    entries = [
        name for name in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, name))
    ]
    entries.sort(reverse=True)
    return entries

def read_metrics_csv(path: str, max_rows: int = 200):
    rows = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(r)
    except Exception:
        pass
    return rows

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(page_title="Multi-Agent RL Control Panel", layout="wide")
st.title("Multi-Agent RL Control Panel")

left, right = st.columns([1, 3])

with left:
    st.header("Experiment Controls")

    # runs dropdown (unique key)
    runs = list_runs(RUNS_DIR)
    run_choice = st.selectbox("Choose run (for metrics/replay/checkpoint)", options=["<none>"] + runs, index=0, key="select_run")

    st.markdown("---")
    st.subheader("Environment")

    env_choice = st.selectbox("Environment", options=["pistonball", "simple_tag"], index=0, key="select_env")

    st.markdown("---")
    st.subheader("Actions")

    disabled = st.session_state.busy

    # unique keys for every button to avoid duplicate element IDs
    if st.button("Evaluate Pistonball", key="btn_eval_piston", disabled=disabled):
        start_job(CMD_EVAL_PISTON, run_tag="eval_pistonball")

    if st.button("Train Pistonball (shared PPO)", key="btn_train_piston", disabled=disabled):
        start_job(CMD_TRAIN_PISTON, run_tag="train_pistonball")

    if st.button("Evaluate Simple Tag", key="btn_eval_simple", disabled=disabled):
        start_job(CMD_EVAL_SIMPLE, run_tag="eval_simpletag")

    if st.button("Train Simple Tag (independent PPO)", key="btn_train_simple", disabled=disabled):
        start_job(CMD_TRAIN_SIMPLE, run_tag="train_simpletag")

    st.markdown("")
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("STOP current job", key="btn_stop", disabled=not st.session_state.busy):
            request_stop()
    with cols[1]:
        if st.button("Force-clear logs", key="btn_clear"):
            st.session_state.log_lines = []
            st.session_state.log_queue = None
            st.session_state.worker_thread = None
            st.session_state.stop_holder = None
            st.session_state.busy = False

    st.markdown("---")
    st.write("Last run:", st.session_state.last_run)
    st.write("Status:", "BUSY" if st.session_state.busy else "IDLE")

with right:
    st.header("Live logs")
    # drain logs if queue present
    if st.session_state.log_queue is not None:
        done = drain_log_queue()
        if done:
            # worker done -> reset busy and cleanup
            st.session_state.busy = False
            if st.session_state.proc_info is not None:
                st.session_state.proc_info["ended_at"] = time.time()
            st.session_state.log_queue = None
            st.session_state.worker_thread = None
            st.session_state.stop_holder = None
            # force UI update so buttons re-enable
            safe_rerun()

    # show code box with logs (cap to last 2000 lines)
    if st.session_state.log_lines:
        st.code("\n".join(st.session_state.log_lines[-2000:]), language="bash")
    else:
        st.info("No logs yet. Start a run or check your terminal where you started Streamlit.")

    with st.expander("Debug (session)"):
        st.write({
            "busy": st.session_state.busy,
            "last_run": st.session_state.last_run,
            "proc_info": st.session_state.proc_info,
            "has_worker": st.session_state.worker_thread is not None,
            "selected_run": run_choice,
            "env_choice": env_choice,
        })

    # If a run is selected, show metrics.csv and replay.gif if present
    if run_choice and run_choice != "<none>":
        run_path = os.path.join(RUNS_DIR, run_choice)
        st.markdown("---")
        st.subheader(f"Run: {run_choice}")

        # metrics
        metrics_path = os.path.join(run_path, "metrics.csv")
        if os.path.isfile(metrics_path):
            st.write("Metrics (first rows):")
            rows = read_metrics_csv(metrics_path, max_rows=50)
            if rows:
                # display CSV as table (first row header if present)
                try:
                    # attempt to render as table
                    header = rows[0]
                    data = rows[1:]
                    st.table([dict(zip(header, r)) for r in data[:30]])
                except Exception:
                    st.text("\n".join([",".join(r) for r in rows[:30]]))
            else:
                st.info("metrics.csv is empty or unreadable.")
        else:
            st.info("No metrics.csv found for this run.")

        # replay gif
        gif_path = os.path.join(run_path, "replay.gif")
        if os.path.isfile(gif_path):
            try:
                st.write("Replay:")
                st.image(gif_path, use_column_width=True)
            except Exception as e:
                st.error(f"Could not display replay.gif: {e}")
        else:
            st.warning("No replay.gif found for this run (that's OK). If your evaluation script creates a gif, ensure it writes to runs/<run>/replay.gif and that Streamlit process has filesystem access to that path.")

# Polling behavior: if busy, sleep a tiny bit and rerun to fetch logs (main thread only)
if st.session_state.busy:
    time.sleep(0.25)
    safe_rerun()
