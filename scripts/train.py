"""Fine-tune a YOLO model on the tabletop dataset (train = s01-s04, val = s05).

The test split (s06) is never used here: it is only evaluated once, at the very end.

Example:
  python scripts/train.py --name yolo11n_e30 --model yolo11n.pt --epochs 30
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolo11n.pt", help="pretrained weights, or a .yaml to train from scratch")
parser.add_argument("--name", required=True, help="run name, saved under runs/<name>")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--patience", type=int, default=10, help="early stopping on val fitness")
parser.add_argument("--device", default="mps")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Ultralytics resolves a relative `path:` against its own datasets dir, so write an absolute copy.
src = Path("data/dataset.yaml").resolve()
cfg = yaml.safe_load(src.read_text())
cfg["path"] = str((src.parent / cfg["path"]).resolve())
cfg.pop("test", None)  # keep the test split out of training runs
tmp = Path("runs") / "_dataset_abs.yaml"
tmp.parent.mkdir(exist_ok=True)
tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))

model = YOLO(args.model)
model.train(
    data=str(tmp),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    patience=args.patience,
    device=args.device,
    seed=args.seed,
    deterministic=True,
    workers=4,
    project=str(Path("runs").resolve()),
    name=args.name,
    exist_ok=False,
    plots=True,
)
