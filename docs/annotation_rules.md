# Annotation rules

Classes: `pen`, `usb_key`, `spoon`, `fork`, `charger`, `airpod`.

- `charger`: the power adapter block only (not the cable).
- `airpod`: each individual earbud, and the closed case.
- Box tightly around the visible part of the object.
- Do not annotate an object more than 70% hidden or cut by another object or the image border.
- Annotate every visible instance, including objects held in a hand.
- Keep 5-10% negative images (no object): empty label file.
- Never move an image between train/val/test after the split.

## Splits (by session, never random)
- train: sessions s01-s04
- val: session s05
- test: session s06 (new background, not looked at until final evaluation)
