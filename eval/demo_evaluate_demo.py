"""
Quick demo evaluator: creates a runs/demo_<env>_<timestamp> folder
with a fake metrics.csv and a small replay.gif. No RL libs required.
Usage:
    python eval/demo_evaluate_demo.py --env pistonball --shared
    python eval/demo_evaluate_demo.py --env simple_tag --independent
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

try:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except Exception:
    print("This demo requires imageio, pillow, numpy. Please install if missing (pip install imageio pillow numpy).")
    raise

def make_demo_gif(path, frames=20, w=320, h=240):
    imgs = []
    for i in range(frames):
        img = Image.new("RGB", (w, h), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        # moving circle
        cx = int((w - 40) * (i / max(1, frames-1))) + 20
        cy = h // 2
        r = 20
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(200, 60, 60))
        # overlay text
        draw.text((8, 8), f"Demo run frame {i+1}/{frames}", fill=(220,220,220))
        imgs.append(np.array(img))
    imageio.mimsave(path, imgs, duration=0.06)
    print(f"Saved demo gif to {path}")

def make_demo_metrics(path, steps=50):
    header = ["step", "reward", "loss"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(1, steps+1):
            reward = round(100 * (1 - (0.8 ** (step/5))) + (step % 5) * 0.5, 3)
            loss = round(1.0 / (step ** 0.5), 4)
            writer.writerow([step, reward, loss])
    print(f"Saved demo metrics to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["pistonball","simple_tag"])
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()

    ts = int(time.time())
    runname = f"demo_{args.env}_{'shared' if args.shared else 'indep' if args.independent else 'run'}_{ts}"
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    run_dir = runs_dir / runname
    run_dir.mkdir(parents=True, exist_ok=True)

    # write metrics.csv
    metrics_path = run_dir / "metrics.csv"
    make_demo_metrics(metrics_path, steps=60)

    # write replay.gif
    gif_path = run_dir / "replay.gif"
    make_demo_gif(gif_path, frames=30)

    # optional model placeholder
    (run_dir / "model.zip").write_text("demo model placeholder\n")

    print("Demo evaluation complete. Run folder:", str(run_dir.resolve()))
    print("Now open Streamlit and select run:", runname)

if __name__ == "__main__":
    main()
