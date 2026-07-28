# AI Interview Proctoring System

Real-time interview monitoring: face presence/positioning, identity
verification, prohibited device detection, and progressive infraction
handling with auto-termination.

## What's here vs. the original version

The original version only monitored face count and position, with
separate hardcoded timers per violation type (no shared infraction
count, no identity check, no device detection). This version adds the
pieces that were on the resume but not actually built yet:

| Feature | Status |
|---|---|
| Face presence/position monitoring | Original, working |
| **Identity verification (LBPH)** | **New — tested, see `test_identity_verifier.py`** |
| **Progressive infraction counter (3-strike)** | **New — tested, see `test_infraction_counter.py`** |
| **Prohibited device detection (phone/book/laptop)** | **New — written, NOT runtime-tested (see below)** |

## Important: what's actually verified

- `identity_verifier.py` and `infraction_counter.py` were both run and
  tested in the environment they were built in (synthetic images for
  identity, direct logic tests for the counter — see the two test files).
  Run them yourself too: `python test_identity_verifier.py` and
  `python test_infraction_counter.py`.
- `device_detector.py` was **written but not executed** — the `ultralytics`
  install (needs PyTorch, 500MB+) wasn't feasible to fully verify in a
  sandboxed environment with no webcam anyway. **You must test this
  yourself** before trusting or describing it as working:
  1. Download a general COCO-pretrained model: `yolov8n.pt` (not the
     face-only `yolov8n-face.pt` already in this repo) — the `ultralytics`
     library auto-downloads it on first use if `models/yolov8n.pt` isn't
     present, or download manually from the Ultralytics releases page.
  2. Run `main.py` with a phone or book visibly in frame and confirm it's
     detected and boxed in red.
  3. If class IDs don't match (unlikely but possible across model
     versions), `device_detector.py` reads them from `model.names`
     dynamically, so it should self-correct — but verify anyway.

## Why LBPH instead of dlib 128D embeddings

The original resume claimed dlib-based embeddings, but no dlib code
ever existed in this project. LBPH (used here) is a lighter, more
reliably-installable alternative already available through
`opencv-contrib-python`. It's less accurate than a modern deep embedding
model, and is sensitive to lighting/pose changes — mitigated here by
requiring several consecutive mismatched frames (not just one) before
flagging impersonation. See the docstring in `identity_verifier.py` for
the full tradeoff explanation — know this if asked "why not dlib."

## Files

```
main.py                    integrates everything, run this
identity_verifier.py       LBPH-based identity verification (tested)
infraction_counter.py      progressive 3-strike warning/termination logic (tested)
device_detector.py         prohibited device detection (NOT runtime-tested -- verify yourself)
utils.py                   alert_user / cancel_interview helpers (original)
test_identity_verifier.py  run this to verify identity logic yourself
test_infraction_counter.py run this to verify infraction logic yourself
models/yolov8n-face.pt     face detection model (included)
models/yolov8n.pt          NOT included -- auto-downloads via ultralytics, or fetch manually
requirements.txt
```

## Setup & run

```bash
pip install -r requirements.txt
python test_identity_verifier.py    # sanity check identity logic
python test_infraction_counter.py   # sanity check infraction logic
python main.py                      # run the full system (needs a webcam)
```

## Known limitations

- Identity enrollment uses a single reference frame at session start —
  no multi-angle enrollment.
- Device detection confidence threshold (0.4) is a starting point, not
  tuned against real test footage yet.
- No persistence/logging of session infraction history to disk — the
  `InfractionCounter.log` list exists in-memory only.
