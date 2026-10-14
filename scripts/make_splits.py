"""Rebuild the dataset split BY SESSION from a Roboflow YOLO export.

Roboflow splits images at random, which leaks near-duplicate frames of the same session
across train/val/test. Here every image goes to a split according to its session prefix:
  train: s01-s04   val: s05   test: s06 (touched only for the final evaluation)

Usage: python scripts/make_splits.py data/_export_v2
Output: data/dataset/{train,val,test}/{images,labels} and data/dataset.yaml
"""
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = {"train": {"s01", "s02", "s03", "s04"}, "val": {"s05"}, "test": {"s06"}}

export = Path(sys.argv[1])
out = Path("data/dataset")
names = yaml.safe_load((export / "data.yaml").read_text())["names"]

if out.exists():
    shutil.rmtree(out)
for split in SPLITS:
    (out / split / "images").mkdir(parents=True)
    (out / split / "labels").mkdir(parents=True)

session_to_split = {s: split for split, sessions in SPLITS.items() for s in sessions}
n_files = Counter()
class_counts = defaultdict(Counter)
seen = set()

for roboflow_split in ("train", "valid", "test"):
    for img in sorted((export / roboflow_split / "images").glob("*.jpg")):
        m = re.match(r"(s\d+)_(\d+)_jpg\.rf\.", img.name)
        if not m:
            raise SystemExit(f"Unexpected file name: {img.name}")
        session, idx = m.groups()
        clean = f"{session}_{idx}"
        assert clean not in seen, f"duplicate image {clean}"
        seen.add(clean)
        split = session_to_split[session]
        label = export / roboflow_split / "labels" / (img.stem + ".txt")
        shutil.copy(img, out / split / "images" / f"{clean}.jpg")
        shutil.copy(label, out / split / "labels" / f"{clean}.txt")
        n_files[split] += 1
        for line in label.read_text().splitlines():
            if line.strip():
                class_counts[split][names[int(line.split()[0])]] += 1

(Path("data") / "dataset.yaml").write_text(
    yaml.safe_dump({"path": "dataset", "train": "train/images", "val": "val/images",
                    "test": "test/images", "nc": len(names), "names": names}, sort_keys=False))

print(f"{'split':6s} {'images':>6s} " + " ".join(f"{n:>8s}" for n in names))
for split in SPLITS:
    print(f"{split:6s} {n_files[split]:6d} " + " ".join(f"{class_counts[split][n]:8d}" for n in names))
print(f"total images: {sum(n_files.values())}")
