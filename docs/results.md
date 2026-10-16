# Results log

Split by session: train = s01-s04 (319 img), val = s05 (80 img), test = s06 (80 img, not used yet).
All numbers below are on **val** unless stated. Best-epoch selection also used val, so val numbers are slightly optimistic;
the unbiased number will be the single final evaluation on test.

## Run 1: YOLO11n fine-tuned from COCO weights (30 epochs, imgsz 640, batch 16, seed 0)
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

## Baseline 1: COCO-pretrained YOLO11n, no fine-tuning (only fork/spoon exist in COCO)
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

## Controlled experiments on Kaggle (Tesla T4), 6 variants x 3 seeds, evaluated on val (s05)
Raw data: `results/kaggle_all_runs.csv`, `results/kaggle_summary_mean_std.csv`. Notebook: `notebooks/kaggle_experiments.ipynb`.
Mean +/- std over seeds 0, 1, 2. Each variant changes one factor relative to the baseline (except scratch, see note).

| variant | mAP50 | mAP50-95 | minutes/run | note |
|---|---|---|---|---|
| baseline_yolo11n (COCO init, 30 ep, 640 px) | 0.940 +/- 0.010 | 0.815 +/- 0.002 | 1.9 | reproduces the Mac run (0.927 / 0.805) |
| scratch_yolo11n (no pretrained weights, 100 ep) | 0.632 +/- 0.024 | 0.407 +/- 0.028 | 5.3 | given 100 epochs so random init is not handicapped |
| aug_strong (hsv_s .9, hsv_v .6, rot 10 deg, translate .2, mixup .1) | 0.906 +/- 0.014 | 0.657 +/- 0.004 | 1.6 | worse, mostly on mAP50-95 |
| yolo11s (bigger model) | 0.946 +/- 0.005 | 0.835 +/- 0.007 | 2.5 | +2.0 pt mAP50-95, consistent over seeds |
| epochs60 | 0.933 +/- 0.003 | 0.823 +/- 0.001 | 3.3 | small gain, converged |
| imgsz960 | 0.934 +/- 0.010 | 0.821 +/- 0.013 | 3.2 | within noise |

Per-class AP50-95 (mean of 3 seeds): fork, pen, spoon stay around 0.76-0.82 for every pretrained variant; airpod ~0.94, usb_key ~0.82-0.86.
Caveat: the best epoch of every run is picked on val, so all val numbers are slightly optimistic. The test split (s06) has not been used.

## Inference speed on the development Mac (Apple M1, 8 GB RAM), 640x360 frames, imgsz 640, batch 1
`scripts/benchmark_fps.py`: preprocessing + network + NMS on real val frames; webcam capture and drawing are not included.
Machine was not idle (other apps open); two MPS runs shown to give the spread. torch used 1 CPU thread (CPU numbers are pessimistic).

| model | device | FPS | median latency | breakdown (pre + net + post) |
|---|---|---|---|---|
| YOLO11n | MPS (GPU) | 52-69 | 14-18 ms | ~1 + 11-15 + ~3 ms |
| YOLO11s | MPS (GPU) | 26-28 | 36 ms | ~1 + 11-12 + ~24 ms (NMS dominates, not investigated) |
| YOLO11n | CPU | 20 | 46 ms | |
| YOLO11s | CPU | 11 | 90 ms | |

Trade-off: YOLO11s gains ~2 points of mAP50-95 on val but runs at roughly half the speed; with webcam capture and drawing added, it would sit around 20 FPS.

## Augmentation ablation (Kaggle, YOLO11n, 3 seeds, val)
Raw: `results/kaggle_aug_all_runs.csv`. Baseline was re-run in the same notebook: identical numbers (runs are reproducible).
| variant | mAP50 | mAP50-95 |
|---|---|---|
| baseline | 0.940 +/- 0.010 | 0.815 +/- 0.002 |
| aug_color (hsv_s .9, hsv_v .6) | 0.938 +/- 0.011 | 0.820 +/- 0.017 (noise) |
| aug_rot10 (rotation 10 deg only) | 0.898 +/- 0.018 | 0.691 +/- 0.011 |
Rotation alone costs ~12 points of mAP50-95 and explains most of the drop of `aug_strong`. Likely cause (not directly verified):
axis-aligned boxes around a rotated object are looser. Extra colour jitter does not help. No extra augmentation is kept.

## FINAL evaluation on the test split (s06, 80 images, marble table, never used for any decision)
Evaluated once, with the two candidates fixed beforehand: seed-0 weights of YOLO11n and YOLO11s (same settings as the baseline run). Single seed per model.
| model | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLO11n (fine-tuned) | 0.974 | 0.975 | 0.981 | 0.854 |
| YOLO11s (fine-tuned) | 0.979 | 0.973 | 0.982 | 0.869 |

Per-class AP50-95: airpod .907/.921, charger .812/.841, fork .871/.831, pen .851/.893, spoon .817/.822, usb_key .866/.906 (n / s).

COCO zero-shot baseline on test (fork / spoon only): fork AP50 0.715, AP50-95 0.489; spoon AP50 0.899, AP50-95 0.717.
Fine-tuned YOLO11n vs COCO: fork 0.715 -> 0.992 (AP50), 0.489 -> 0.871 (AP50-95); spoon 0.899 -> 0.916, 0.717 -> 0.817.

### Caveats (read before quoting any number)
- Test scores are HIGHER than val scores (0.981 vs 0.940 mAP50). This was not expected: s06 is a different surface and background but it is
  also a *cleaner* scene (no clutter, no nail clipper, no distractors), whereas s05 was built to be cluttered. Sanity checks found no leakage:
  sessions are disjoint, no shared file names, and test images are farther from train/val images than train images are from each other.
  Do not present this as "the model generalises better than validation"; present it as "scene difficulty differs".
- One seed per model, 80 images: the n vs s difference (0.854 vs 0.869 mAP50-95) is within noise here, although it matches the val trend.
- YOLO11s produces low-confidence (0.27-0.51) false `spoon` boxes in 19/80 test images: mostly a small box stuck in the top-left corner
  (dark marble vein), and once the phone (a distractor) labelled as spoon. With a confidence threshold of 0.5 most disappear.
  YOLO11n has far fewer false positives (2, 2, 3, 1, 1 per class at conf 0.25).
- Error analysis on test was done after the fact and not used to change any model.
