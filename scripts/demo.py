"""Real-time tabletop object detection from the webcam (or on a folder of images).

Webcam:  python scripts/demo.py                    (add --camera N if your iPhone is camera 0, see capture.py --list)
Video:   python scripts/demo.py --video my_clip.mp4 --save demo_out.mp4          (annotated copy of a recorded clip)
Images:  python scripts/demo.py --images data/dataset/val/images --out screenshots/val_demo

Keys (webcam window):  q quit | m switch model (YOLO11n <-> YOLO11s) | + / - confidence threshold | s save screenshot
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODELS = {"yolo11n": "weights/yolo11n_tabletop.pt", "yolo11s": "weights/yolo11s_tabletop.pt"}
# One BGR colour per class, fixed so that a class keeps its colour across frames.
COLORS = {"airpod": (255, 144, 30), "charger": (230, 200, 0), "fork": (60, 60, 230),
          "pen": (80, 200, 80), "spoon": (200, 60, 160), "usb_key": (0, 165, 255)}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolo11n", choices=list(MODELS))
parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0.5 hides most low-score false positives)")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--video", help="run on a recorded video file instead of the webcam")
parser.add_argument("--save", help="write the annotated video to this .mp4 path (webcam or --video)")
parser.add_argument("--no-display", action="store_true", help="do not open a window (useful with --video --save)")
parser.add_argument("--images", help="annotate every .jpg of this folder instead of using the webcam")
parser.add_argument("--out", default="screenshots/demo", help="output folder for --images")
parser.add_argument("--limit", type=int, default=12, help="max images with --images")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"


def draw(frame, result, hud):
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
        name = result.names[cls]
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ytop = y1 - th - 8 if y1 - th - 8 > 30 else y1 + th + 8  # keep the label below the HUD bar, inside the box if needed
        cv2.rectangle(frame, (x1, ytop), (x1 + tw + 6, ytop + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 3, ytop + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def load(name):
    path = Path(MODELS[name])
    if not path.exists():
        raise SystemExit(f"Missing weights: {path}")
    return YOLO(str(path))


model_name = args.model
model = load(model_name)
conf = args.conf

if args.images:  # batch mode, no window: used for tests and README figures
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.images).glob("*.jpg"))[:: max(1, len(list(Path(args.images).glob("*.jpg"))) // args.limit)][: args.limit]:
        frame = cv2.imread(str(path))
        result = model.predict(frame, conf=conf, imgsz=args.imgsz, device=device, verbose=False)[0]
        cv2.imwrite(str(out / path.name), draw(frame, result, f"{model_name} | conf>={conf:.2f} | {len(result.boxes)} objects"))
    print(f"saved annotated images to {out}")
    raise SystemExit

cap = cv2.VideoCapture(args.video if args.video else args.camera)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {args.video}." if args.video else
                     "Cannot open the camera. Check macOS Settings > Privacy > Camera, or try --camera 1.")
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
writer = None
n_frames = 0

fps, last = 0.0, time.perf_counter()
shots = Path("screenshots")
shots.mkdir(exist_ok=True)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model.predict(frame, conf=conf, imgsz=args.imgsz, device=device, verbose=False)[0]
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
    last = now
    view = draw(frame.copy(), result, f"{model_name} | {fps:4.1f} FPS | conf>={conf:.2f} | q quit, m model, +/- conf, s save")
    if args.save:
        if writer is None:
            h, w = view.shape[:2]
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        writer.write(view)
    n_frames += 1
    if args.no_display:
        continue
    cv2.imshow("Tabletop detection", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        model_name = "yolo11s" if model_name == "yolo11n" else "yolo11n"
        model = load(model_name)
        fps = 0.0
    elif key in (ord("+"), ord("=")):
        conf = min(0.95, conf + 0.05)
    elif key in (ord("-"), ord("_")):
        conf = max(0.05, conf - 0.05)
    elif key == ord("s"):
        cv2.imwrite(str(shots / f"demo_{int(time.time())}.jpg"), view)

cap.release()
if writer is not None:
    writer.release()
    print(f"wrote {n_frames} annotated frames to {args.save}")
cv2.destroyAllWindows()
