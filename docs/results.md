# Results log

Split by session: train = s01-s04 (319 img), val = s05 (80 img), test = s06 (80 img, not used yet).
All numbers below are on **val** unless stated. Best-epoch selection also used val, so val numbers are slightly optimistic;
the unbiased number will be the single final evaluation on test.

## Run 1 - YOLO11n fine-tuned from COCO weights (30 epochs, imgsz 640, batch 16, seed 0)
| class | P | R | AP50 | AP50-95 |
|---|---|---|---|---|
| airpod | 0.965 | 1.000 | 0.994 | 0.956 |
| charger | 0.947 | 0.988 | 0.984 | 0.812 |
| usb_key | 1.000 | 0.882 | 0.984 | 0.860 |
| pen | 0.931 | 0.768 | 0.892 | 0.740 |
| fork | 0.808 | 0.879 | 0.877 | 0.720 |
| spoon | 0.986 | 0.676 | 0.828 | 0.743 |
| **all** | 0.939 | 0.865 | **0.927** | **0.805** |

Error analysis (conf >= 0.25, IoU >= 0.5): 28 false `fork` detections, mostly the nail clipper that is present in every
s05 image but never in train; 7 spoons predicted as fork and 7 missed; pen: 3 missed, 7 false positives.

## Baseline 1 - COCO-pretrained YOLO11n, no fine-tuning (only fork/spoon exist in COCO)
| class | P | R | AP50 | AP50-95 |
|---|---|---|---|---|
| fork | 0.871 | 0.411 | 0.571 | 0.427 |
| spoon | 0.731 | 0.586 | 0.730 | 0.599 |

The 4 other classes (pen, usb_key, charger, airpod) are absent from COCO: a COCO model cannot detect them (AP = 0 by construction).

## Fine-tuned vs COCO baseline (val, same boxes, 2 shared classes)
| class | AP50 COCO -> fine-tuned | AP50-95 COCO -> fine-tuned |
|---|---|---|
| fork | 0.571 -> 0.877 | 0.427 -> 0.720 |
| spoon | 0.730 -> 0.828 | 0.599 -> 0.743 |
