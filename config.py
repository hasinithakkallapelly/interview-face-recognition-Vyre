"""
config.py

Every tunable constant for the proctoring system in one place, instead of
scattered across main.py / identity_verifier.py / device_detector.py as
hardcoded literals. `parse_args()` builds a Config from the defaults below,
overridden by any CLI flags passed on the command line.
"""

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # camera
    webcam_index: int = 0

    # models
    face_model_path: str = "models/yolov8n-face.pt"
    device_model_path: str = "models/yolov8n.pt"

    # face detection
    face_confidence_threshold: float = 0.5

    # identity verification (see identity_verifier.py for what these mean)
    enrollment_frames_to_wait: int = 15
    enrollment_frames_required: int = 5
    mismatch_threshold: float = 80.0
    consecutive_required: int = 5

    # device detection
    device_confidence_threshold: float = 0.4

    # infractions
    max_infractions: int = 3

    # session logging (infraction log + snapshot frames on each infraction)
    log_dir: str = "session_logs"
    save_snapshots: bool = True


def parse_args(argv=None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="AI interview proctoring system")
    parser.add_argument("--webcam-index", type=int, default=defaults.webcam_index)
    parser.add_argument("--face-model-path", default=defaults.face_model_path)
    parser.add_argument("--device-model-path", default=defaults.device_model_path)
    parser.add_argument("--face-confidence-threshold", type=float,
                         default=defaults.face_confidence_threshold)
    parser.add_argument("--enrollment-frames-to-wait", type=int,
                         default=defaults.enrollment_frames_to_wait)
    parser.add_argument("--enrollment-frames-required", type=int,
                         default=defaults.enrollment_frames_required)
    parser.add_argument("--mismatch-threshold", type=float, default=defaults.mismatch_threshold)
    parser.add_argument("--consecutive-required", type=int, default=defaults.consecutive_required)
    parser.add_argument("--device-confidence-threshold", type=float,
                         default=defaults.device_confidence_threshold)
    parser.add_argument("--max-infractions", type=int, default=defaults.max_infractions)
    parser.add_argument("--log-dir", default=defaults.log_dir)
    parser.add_argument("--no-snapshots", dest="save_snapshots", action="store_false",
                         default=defaults.save_snapshots)
    args = parser.parse_args(argv)
    return Config(**vars(args))
