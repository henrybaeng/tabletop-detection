# Tabletop Object Detection for Robotic Perception

Fine-tuning a YOLO nano detector on a self-collected dataset of 6 tabletop objects
(pen, USB key, spoon, fork, charger, AirPod), with a real-time webcam demo.

> Work in progress. Results, method and limitations will be filled in as the project advances.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Capture images (webcam)
```bash
python scripts/capture.py --session s01
```
Press SPACE to save a frame, Q to quit.
