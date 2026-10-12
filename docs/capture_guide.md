# Capture guide (one session = 80 images)

Same script for every session (s01 ... s06). Only the lighting and the surface change from one session
to the next. Inside a session, lighting and surface never change.

Objects: `pen`, `usb_key`, `spoon`, `fork`, `charger`, `airpod`.

## Rules for every single image
- Move or rotate at least one object between two images (never take two identical shots).
- Do not put the same object at the same place twice in a row.
- Vary the orientation: 0 deg (parallel to the table edge), 45 deg, 90 deg.
- Vary the position: left, center, right, near the camera, far from the camera.
- Objects are fully visible in the frame, except in the blocks that ask for partial hiding.
- Do not tidy up the desk between images: some clutter around is good.

Command:
```bash
python scripts/capture.py --session s01
```
SPACE = save, Q = quit. Tick the boxes as you go. Running total in brackets.

## Block 1 - Empty table (2 images) [2]
- [ ] 1. Empty table
- [ ] 2. Empty table, with one unrelated object (phone, bottle...)

## Block 2 - One object alone (12 images) [14]
Two images per object: first 0 deg in the center, then 45 deg off-center.
- [ ] pen x2
- [ ] usb_key x2
- [ ] spoon x2
- [ ] fork x2
- [ ] charger x2
- [ ] airpod x2

## Block 3 - Pairs (24 images) [38]
Each pair is photographed twice: once, then swap the sides and change the orientation.
- [ ] pen + usb_key (x2)
- [ ] spoon + fork (x2)
- [ ] charger + airpod (x2)
- [ ] pen + spoon (x2)
- [ ] usb_key + charger (x2)
- [ ] fork + airpod (x2)
- [ ] pen + fork (x2)
- [ ] usb_key + airpod (x2)
- [ ] spoon + charger (x2)
- [ ] pen + charger (x2)
- [ ] usb_key + fork (x2)
- [ ] spoon + airpod (x2)

## Block 4 - Groups of 3 or 4 (24 images) [62]
Each group is photographed 4 times: rearrange all the objects between each shot (positions and angles).
- [ ] pen, usb_key, charger (x4)
- [ ] spoon, fork, airpod (x4)
- [ ] pen, spoon, airpod, usb_key (x4)
- [ ] fork, charger, usb_key (x4)
- [ ] pen, fork, airpod, charger (x4)
- [ ] spoon, usb_key, charger (x4)

## Block 5 - All 6 objects (10 images) [72]
- [ ] 3 images spread out, no contact
- [ ] 3 images clustered, objects touching
- [ ] 2 images with partial overlap (an object half under another, at most 50% hidden)
- [ ] 2 images with the camera farther away (objects look small)

## Block 6 - Hand interaction (6 images) [78]
One image per object: a hand touches or holds the object on the table (rest of the scene: 1-2 other objects).
- [ ] pen
- [ ] usb_key
- [ ] spoon
- [ ] fork
- [ ] charger
- [ ] airpod

## Block 7 - Empty table again (2 images) [80]
- [ ] 79. Empty table
- [ ] 80. Table with only background clutter

## Final check
- About 80 files in `data/raw/sXX/`.
- Open 5-6 random images: objects visible, not blurry, not too dark.
- Do not delete "ugly" images: hard cases belong in the dataset.
- Do not label or move anything yet; tell Claude when the session is done.

## Planned sessions
| Session | Lighting | Surface | Role |
|---|---|---|---|
| s01 (done) | desk lamp | desk + black mouse pad | train |
| s02 | desk lamp moved to the side (or aimed at the wall), ceiling off | desk + black mouse pad | train |
| s03 | desk lamp above, as in s01 | white sheet of paper (mouse pad removed) | train |
| s04 | ceiling light + phone flashlight placed at a distance | dark cloth | train |
| s05 | ceiling light + desk lamp | messy desk | val |
| s06 | living-room light | white marble table (glossy, wall + wooden chair behind) | test (never seen: shoot last, do not tune on it) |
