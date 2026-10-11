"""Audit YOLO-format labels: class counts, empty files, invalid boxes, tiny boxes, per-session stats.

Usage: python scripts/audit_labels.py data/_export_s01
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

root = Path(sys.argv[1])
names = yaml.safe_load((root / "data.yaml").read_text())["names"]

per_class = Counter()
per_session = defaultdict(Counter)
n_images = n_empty = 0
problems = []
areas = defaultdict(list)

for split in ("train", "valid", "test"):
    img_dir, lab_dir = root / split / "images", root / split / "labels"
    if not img_dir.exists():
        continue
    for img in sorted(img_dir.glob("*.jpg")):
        n_images += 1
        session = re.match(r"(s\d+)_", img.name).group(1)
        lab = lab_dir / (img.stem + ".txt")
        if not lab.exists():
            problems.append(f"{img.name}: label file missing")
            continue
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            n_empty += 1
        for parts in lines:
            if len(parts) != 5:
                problems.append(f"{img.name}: malformed line {parts}")
                continue
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            per_class[names[c]] += 1
            per_session[session][names[c]] += 1
            areas[names[c]].append(w * h)
            if not (0 <= x - w / 2 and x + w / 2 <= 1.001 and 0 <= y - h / 2 and y + h / 2 <= 1.001):
                problems.append(f"{img.name}: box outside image ({names[c]})")
            if w * h < 0.002:
                problems.append(f"{img.name}: very small box ({names[c]}, area={w*h:.4f})")

print(f"images: {n_images}, empty-label images (negatives): {n_empty}")
print("\nannotations per class:")
for c in names:
    a = areas[c]
    mean = sum(a) / len(a) if a else 0
    print(f"  {c:8s} {per_class[c]:4d}   mean box area = {mean:.3f} of image")
print(f"\ntotal annotations: {sum(per_class.values())}")
print(f"\nproblems: {len(problems)}")
for p in problems:
    print("  -", p)
