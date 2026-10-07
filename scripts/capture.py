"""Capture frames from the Mac webcam into data/raw/<session>/.

SPACE saves a frame (resized to 640 px wide), Q quits.
Files are named <session>_<NNN>.jpg so the session stays in the filename.
"""
import argparse
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True, help="e.g. s01")
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

out_dir = Path("data/raw") / args.session
out_dir.mkdir(parents=True, exist_ok=True)
count = len(list(out_dir.glob("*.jpg")))

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Check macOS Settings > Privacy > Camera for your terminal.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    cv2.putText(view, f"{args.session}: {count} saved | SPACE=save Q=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("capture", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, int(h * 640 / w)))
        count += 1
        cv2.imwrite(str(out_dir / f"{args.session}_{count:03d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
