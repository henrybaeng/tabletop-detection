"""Generate the README figures from the saved result files (no training needed)."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, BLUE = "#1f2933", "#6b7785", "#e3e7ec", "#2a6fb0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": GRID})

# 1) experiments: mean mAP50-95 on val over 3 seeds, one dot per seed
main = pd.read_csv("results/kaggle_all_runs.csv")
aug = pd.read_csv("results/kaggle_aug_all_runs.csv")
df = pd.concat([main, aug[aug.experiment.isin(["aug_color", "aug_rot10"])]])
labels = {"baseline_yolo11n": "YOLO11n baseline (COCO init, 30 ep)", "yolo11s": "YOLO11s (bigger model)",
          "epochs60": "60 epochs", "imgsz960": "image size 960", "aug_color": "+ colour augmentation",
          "aug_rot10": "+ rotation 10 deg", "aug_strong": "+ strong augmentation (incl. rotation)",
          "scratch_yolo11n": "from scratch (no COCO weights, 100 ep)"}
order = ["yolo11s", "epochs60", "imgsz960", "aug_color", "baseline_yolo11n", "aug_rot10", "aug_strong", "scratch_yolo11n"]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, k in enumerate(order[::-1]):
    v = df[df.experiment == k]["mAP50-95"].values
    ax.barh(i, v.mean(), height=0.55, color=BLUE if k != "baseline_yolo11n" else "#17446e", zorder=2)
    ax.scatter(v, [i] * len(v), s=14, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.text(max(v.max(), v.mean()) + 0.022, i, f"{v.mean():.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([labels[k] for k in order[::-1]])
ax.set_xlim(0, 1)
ax.set_xlabel("mAP50-95, validation session (bar = mean of 3 seeds, dots = seeds)")
ax.xaxis.grid(True, color=GRID, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "experiments.png", dpi=160)
plt.close(fig)

# 2) confusion matrices (val and test, YOLO11n seed 0), counts, rows = true class
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (title, path) in zip(axes, [("Validation (s05)", "results/val_yolo11n_seed0.json"),
                                    ("Test (s06)", "results/test_yolo11n_seed0.json")]):
    d = json.load(open(path))["confusion_matrix"]
    m, lab = np.array(d["matrix"]), d["labels"]
    lab_y = [l if l != "none" else "background" for l in lab]
    ax.imshow(m, cmap="Blues", vmin=0, vmax=40)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 22 else INK)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels([l if l != "none" else "missed" for l in lab], rotation=40, ha="right")
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(lab_y)
    ax.set_xlabel("predicted class ('missed' = not detected)")
    ax.set_ylabel("true class ('background' = false detection)")
    ax.set_title(f"{title}, conf >= 0.25", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=160)
plt.close(fig)

# 3) dataset montage: one multi-object frame from each of the 6 sessions
tiles = []
for s, split in [("s01", "train"), ("s02", "train"), ("s03", "train"), ("s04", "train"), ("s05", "val"), ("s06", "test")]:
    img = cv2.imread(f"data/dataset/{split}/images/{s}_040.jpg")
    img = cv2.resize(img, (426, 240))
    cv2.rectangle(img, (0, 0), (150, 30), (0, 0, 0), -1)
    cv2.putText(img, f"{s} ({split})", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)
cv2.imwrite(str(OUT / "sessions.jpg"), cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("figures written to", OUT)
