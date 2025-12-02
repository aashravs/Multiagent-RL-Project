import imageio
import numpy as np
import pathlib
import pandas as pd

p = pathlib.Path("runs/example_demo")
p.mkdir(parents=True, exist_ok=True)

frames = []
H, W = 120, 160

for t in range(40):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    x = int((t / 39) * (W - 20))
    img[40:60, x:x+20] = 255  # moving white square
    frames.append(img)

# save GIF
imageio.mimsave(p / "replay.gif", frames, fps=12)

# save simple metrics
df = pd.DataFrame([
    {"episode": 1, "return": 548.76},
    {"episode": "mean", "return": 548.76},
])
df.to_csv(p / "metrics.csv", index=False)

print("Created runs/example_demo/replay.gif and metrics.csv")
