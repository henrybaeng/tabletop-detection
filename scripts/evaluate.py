"""Evaluate a trained model: per-class metrics + our own confusion matrix and error counts.

Default split is `val` (s05). The `test` split (s06) is only allowed with --final, and must be
run once, with the model that was chosen using the val split.

Example:
  python scripts/evaluate.py --weights runs/yolo11n_e30/weights/best.pt --name yolo11n_e30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--final", action="store_true", help="required to evaluate on the test split")
parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the confusion matrix")
parser.add_argument("--iou", type=float, default=0.5, help="IoU to match a prediction to a ground-truth box")
parser.add_argument("--device", default="mps")
args = parser.parse_args()
if args.split == "test" and not args.final:
    raise SystemExit("Refusing to touch the test split without --final.")

cfg = yaml.safe_load(Path("data/dataset.yaml").read_text())
names = cfg["names"]
root = Path("data") / cfg["path"]
img_dir = root / args.split / "images"
lab_dir = root / args.split / "labels"

model = YOLO(args.weights)

# 1) standard Ultralytics metrics (mAP computed at low confidence, as usual)
abs_cfg = dict(cfg)
abs_cfg["path"] = str(root.resolve())
tmp = Path("runs") / "_eval_dataset_abs.yaml"
tmp.write_text(yaml.safe_dump(abs_cfg, sort_keys=False))
m = model.val(data=str(tmp), split=args.split, device=args.device, plots=False, verbose=False)
per_class = {names[c]: {"P": float(m.box.p[i]), "R": float(m.box.r[i]),
                        "AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
             for i, c in enumerate(m.box.ap_class_index)}
overall = {"P": float(m.box.mp), "R": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}

# 2) our confusion matrix at a fixed confidence: rows = true class (+ missed), cols = predicted (+ background)
n = len(names)
cm = np.zeros((n + 1, n + 1), dtype=int)  # index n = "none"
for img in sorted(img_dir.glob("*.jpg")):
    lab = lab_dir / f"{img.stem}.txt"
    rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
    gt_cls = [int(r[0]) for r in rows]
    h, w = __import__("PIL.Image", fromlist=["Image"]).open(img).size[::-1]
    gt_box = torch.tensor([[(float(r[1]) - float(r[3]) / 2) * w, (float(r[2]) - float(r[4]) / 2) * h,
                            (float(r[1]) + float(r[3]) / 2) * w, (float(r[2]) + float(r[4]) / 2) * h] for r in rows]).reshape(-1, 4)
    res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
    pr_box = res.boxes.xyxy.cpu()
    pr_cls = res.boxes.cls.int().tolist()
    matched_pred = set()
    if len(gt_cls) and len(pr_cls):
        iou = box_iou(gt_box, pr_box)
        for g in range(len(gt_cls)):
            best = int(torch.argmax(iou[g]))
            if iou[g, best] >= args.iou and best not in matched_pred:
                matched_pred.add(best)
                cm[gt_cls[g], pr_cls[best]] += 1
            else:
                cm[gt_cls[g], n] += 1  # missed
    else:
        for g in gt_cls:
            cm[g, n] += 1
    for p in range(len(pr_cls)):
        if p not in matched_pred:
            cm[n, pr_cls[p]] += 1  # false positive on background

labels = names + ["none"]
print(f"\nsplit={args.split}  overall: " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()))
print(f"{'class':8s} {'P':>6s} {'R':>6s} {'AP50':>6s} {'AP50-95':>8s}")
for k, v in per_class.items():
    print(f"{k:8s} {v['P']:6.3f} {v['R']:6.3f} {v['AP50']:6.3f} {v['AP50-95']:8.3f}")
print(f"\nconfusion matrix (rows = true, cols = predicted, conf>={args.conf}, IoU>={args.iou}); 'none' = missed / false positive")
print(f"{'':9s}" + "".join(f"{l:>9s}" for l in labels))
for i, l in enumerate(labels):
    print(f"{l:9s}" + "".join(f"{cm[i, j]:9d}" for j in range(n + 1)))

Path("runs").mkdir(exist_ok=True)
out = {"split": args.split, "weights": args.weights, "overall": overall, "per_class": per_class,
       "confusion_matrix": {"labels": labels, "matrix": cm.tolist(), "conf": args.conf, "iou": args.iou}}
(Path("runs") / f"eval_{args.name}_{args.split}.json").write_text(json.dumps(out, indent=2))
