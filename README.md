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
| **Identity verification (LBPH, multi-frame enrollment)** | **New — tested, see `test_identity_verifier.py`** |
| **Progressive infraction counter (3-strike)** | **New — tested, see `test_infraction_counter.py`** |
| **Prohibited device detection (phone/book/laptop)** | **New — detection logic tested against a fake model, real YOLO inference NOT runtime-tested (see below)** |
| **Session persistence (infraction log + snapshots to disk)** | **New — tested, see `session_logger.py`** |
| **Config file + CLI flags for all tunables** | **New — see `config.py`** |
| **CI running the test suite on every push** | **New — see `.github/workflows/tests.yml`** |

## Important: what's actually verified

- `identity_verifier.py`, `infraction_counter.py`, `session_logger.py`,
  and the violation-priority logic in `main.py` are all covered by tests
  that run without a webcam or any model file, and pass in this
  environment. Run them yourself:
  `python test_identity_verifier.py`, `python test_infraction_counter.py`,
  `python test_device_detector.py`, `python test_main.py`,
  `python test_session_logger.py`.
- `device_detector.py`'s class-ID resolution and confidence filtering are
  tested against a fake model object (no PyTorch needed), but **real YOLO
  inference on real footage was NOT run** — the `ultralytics` install
  (needs PyTorch, 500MB+) wasn't feasible to fully install and verify
  against a live webcam in this environment. **You must test this
  yourself** before trusting or describing device detection as fully
  working end-to-end:
  1. Download a general COCO-pretrained model: `yolov8n.pt` (not the
     face-only `yolov8n-face.pt` already in this repo) — the `ultralytics`
     library auto-downloads it on first use if `models/yolov8n.pt` isn't
     present, or download manually from the Ultralytics releases page.
  2. Run `main.py` with a phone or book visibly in frame and confirm it's
     detected and boxed in red.
  3. If class IDs don't match (unlikely but possible across model
     versions), `device_detector.py` reads them from `model.names`
     dynamically, so it should self-correct — but verify anyway.
  4. If `models/yolov8n.pt` is missing and unreachable (no network),
     `main.py` now degrades gracefully: it logs a warning and runs the
     rest of the session with device detection disabled, instead of
     crashing.

## Why LBPH instead of dlib 128D embeddings

The original resume claimed dlib-based embeddings, but no dlib code
ever existed in this project. LBPH (used here) is a lighter, more
reliably-installable alternative already available through
`opencv-contrib-python`. It's less accurate than a modern deep embedding
model, and is sensitive to lighting/pose changes — mitigated here by
requiring several consecutive mismatched frames (not just one) before
flagging impersonation, and by enrolling from several frames instead of
one (see "Known limitations" below, now addressed). See the docstring in
`identity_verifier.py` for the full tradeoff explanation — know this if
asked "why not dlib."

## Files

```
main.py                    integrates everything, run this
config.py                  all tunable constants + CLI flag parsing
violation_types.py         ViolationType enum -- single source of truth for violation strings
session_logger.py          persists infraction log + snapshots to session_logs/ (tested)
identity_verifier.py       LBPH-based identity verification, multi-frame enrollment (tested)
infraction_counter.py      progressive 3-strike warning/termination logic (tested)
device_detector.py         prohibited device detection (logic tested against a fake model -- verify real YOLO inference yourself)
utils.py                   alert_user / cancel_interview helpers (now logging-based)
test_identity_verifier.py  run this to verify identity logic yourself
test_infraction_counter.py run this to verify infraction logic yourself
test_device_detector.py    run this to verify device detection logic (no PyTorch needed)
test_main.py               run this to verify the violation-priority logic
test_session_logger.py     run this to verify session persistence logic
models/yolov8n-face.pt     face detection model (included)
models/yolov8n.pt          NOT included -- auto-downloads via ultralytics, or fetch manually
requirements.txt
pyproject.toml             ruff lint config
LICENSE                    MIT
.github/workflows/tests.yml  CI: lints with ruff, then runs all five test files, on every push/PR
```

## Setup & run

```bash
pip install -r requirements.txt
python test_identity_verifier.py    # sanity check identity logic
python test_infraction_counter.py   # sanity check infraction logic
python test_device_detector.py      # sanity check device detection logic (no PyTorch needed)
python test_main.py                 # sanity check violation-priority logic
python test_session_logger.py       # sanity check session persistence logic
python main.py                      # run the full system (needs a webcam)
```

Linting (`ruff check .`) runs in CI on every push; run it locally the same way
if you want to check before pushing. `ruff format` is intentionally NOT
enforced -- the codebase doesn't follow ruff's default style, and running it
would reformat every file for style alone.

All tunables (webcam index, thresholds, max infractions, etc.) live in
`config.py` and can be overridden via CLI flags — run `python main.py
--help` for the full list, e.g.:

```bash
python main.py --webcam-index 1 --max-infractions 5 --mismatch-threshold 70
```

Each run writes `session_logs/session_<timestamp>/infractions.json`
(the full infraction history) and, unless `--no-snapshots` is passed, a
snapshot image for every recorded infraction. Console + session output
is also logged to `session_logs/session.log`.

## Known limitations

- Device detection confidence threshold (0.4) is a starting point, not
  tuned against real test footage yet.
- Identity enrollment now collects multiple frames (`--enrollment-frames-required`,
  default 5) before training, but all are still captured within one short
  window as the candidate settles in — not truly multi-angle/multi-session
  enrollment.
- `device_detector.py`'s real YOLO inference path (as opposed to its
  detection/filtering logic) still has not been runtime-verified in this
  environment — see "Important: what's actually verified" above.
