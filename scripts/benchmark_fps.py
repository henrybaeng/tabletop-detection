"""Measure inference speed (frames per second) of trained weights on this machine.

Frames are real val images (decoded once, so disk I/O is excluded). Each frame goes through the full
Ultralytics pipeline: preprocessing + network + NMS. Webcam capture and drawing are NOT included.

Example:
  python scripts/benchmark_fps.py --weights yolo11n=runs/yolo11n_e30/weights/best.pt yolo11s=kaggle/results/weights/yolo11s_s0_best.pt
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", nargs="+", required=True, help="name=path pairs")
parser.add_argument("--devices", nargs="+", default=["mps", "cpu"])
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--warmup", type=int, default=15)
parser.add_argument("--iters", type=int, default=100)
args = parser.parse_args()

frames = [cv2.imread(str(p)) for p in sorted(Path("data/dataset/val/images").glob("*.jpg"))[:50]]
print(f"{len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}, imgsz={args.imgsz}, torch threads={torch.get_num_threads()}")

results = {}
for item in args.weights:
    name, path = item.split("=", 1)
    for device in args.devices:
        model = YOLO(path)
        for i in range(args.warmup):
            model.predict(frames[i % len(frames)], imgsz=args.imgsz, device=device, verbose=False)
        lat, parts = [], []
        for i in range(args.iters):
            t0 = time.perf_counter()
            r = model.predict(frames[i % len(frames)], imgsz=args.imgsz, device=device, verbose=False)[0]
            lat.append((time.perf_counter() - t0) * 1000)
            parts.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
        lat = np.array(lat)
        p = np.array(parts).mean(axis=0)
        results[f"{name}/{device}"] = {"fps_mean": 1000 / lat.mean(), "latency_ms_median": float(np.median(lat)),
                                       "latency_ms_p95": float(np.percentile(lat, 95)),
                                       "pre_ms": p[0], "net_ms": p[1], "post_ms": p[2]}
        r_ = results[f"{name}/{device}"]
        print(f"{name:8s} {device:4s}  {r_['fps_mean']:6.1f} FPS  median {r_['latency_ms_median']:6.1f} ms  p95 {r_['latency_ms_p95']:6.1f} ms"
              f"  (pre {p[0]:.1f} + net {p[1]:.1f} + post {p[2]:.1f} ms)")

Path("runs").mkdir(exist_ok=True)
Path("runs/benchmark_fps.json").write_text(json.dumps(results, indent=2))
