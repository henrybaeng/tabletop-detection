"""Baseline 1: the COCO-pretrained YOLO11n used as-is (no fine-tuning).

COCO only knows 2 of our 6 classes: fork (COCO id 42) and spoon (id 44). We build a temporary copy of
the split with only those two classes (re-indexed to the COCO ids) and evaluate the pretrained model
restricted to those ids. The 4 other classes (pen, usb_key, charger, airpod) do not exist in COCO, so
a COCO model cannot detect them by construction (AP = 0).

Default split is val (s05); test needs --final.

Example: python scripts/baseline_coco.py --name coco_yolo11n
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="yolo11n.pt")
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--final", action="store_true")
parser.add_argument("--device", default="mps")
args = parser.parse_args()
if args.split == "test" and not args.final:
    raise SystemExit("Refusing to touch the test split without --final.")

cfg = yaml.safe_load(Path("data/dataset.yaml").read_text())
our_names = cfg["names"]
to_coco = {our_names.index("fork"): 42, our_names.index("spoon"): 44}

model = YOLO(args.weights)
assert model.names[42] == "fork" and model.names[44] == "spoon", "unexpected COCO class ids"

src = (Path("data") / cfg["path"] / args.split).resolve()
dst = Path("data/_coco_baseline") / args.split
if dst.parent.exists():
    shutil.rmtree(dst.parent)
(dst / "images").mkdir(parents=True)
(dst / "labels").mkdir(parents=True)
for img in (src / "images").glob("*.jpg"):
    (dst / "images" / img.name).symlink_to(img)
    kept = []
    for line in (src / "labels" / f"{img.stem}.txt").read_text().splitlines():
        parts = line.split()
        if parts and int(parts[0]) in to_coco:
            kept.append(" ".join([str(to_coco[int(parts[0])])] + parts[1:]))
    (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))

data_yaml = dst.parent / "coco_two_classes.yaml"
# Ultralytics requires train/val keys; all point to the single split we evaluate (nothing is trained).
rel = f"{args.split}/images"
data_yaml.write_text(yaml.safe_dump({"path": str(dst.parent.resolve()), "train": rel, "val": rel, "test": rel,
                                      "names": model.names}, sort_keys=False))

m = model.val(data=str(data_yaml), split=args.split, classes=[42, 44], device=args.device, plots=False, verbose=False)
res = {}
for i, c in enumerate(m.box.ap_class_index):
    res[model.names[int(c)]] = {"P": float(m.box.p[i]), "R": float(m.box.r[i]),
                                "AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
print(f"\nCOCO baseline ({args.weights}) on {args.split}, fork + spoon only:")
for k, v in res.items():
    print(f"  {k:6s} P={v['P']:.3f} R={v['R']:.3f} AP50={v['AP50']:.3f} AP50-95={v['AP50-95']:.3f}")
Path("runs").mkdir(exist_ok=True)
(Path("runs") / f"baseline_{args.name}_{args.split}.json").write_text(json.dumps(res, indent=2))
