"""
main.py

Real-time interview proctoring, combining:
  - Face presence/position monitoring (existing, from the original project)
  - Identity verification via LBPH        (tested, see test_identity_verifier.py)
  - Progressive infraction counter        (tested, see test_infraction_counter.py)
  - Prohibited device detection           (see device_detector.py header)

Run: python main.py [--webcam-index 0] [--max-infractions 3] ...
     (run `python main.py --help` for the full list of tunables)
Press ESC to quit manually. Session also auto-terminates after
max_infractions distinct violations (see infraction_counter.py). Either
way, the infraction log is written to disk under session_logs/ on exit.
"""

import logging
import os
import sys
import time

import cv2
import contextlib
import io

from config import parse_args
from utils import alert_user, cancel_interview
from identity_verifier import IdentityVerifier
from infraction_counter import InfractionCounter
from device_detector import DeviceDetector
from session_logger import SessionLogger

logger = logging.getLogger("interview_proctor")


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "session.log")),
        ],
    )


def determine_violation_type(count: int, face_in_center: bool, impersonation_flag: bool,
                              device_hits: list):
    """
    Pure function for picking the current violation type from this frame's
    signals. Priority order matters (checked top to bottom): no_face and
    multiple_faces are the most unambiguous/serious, then impersonation,
    then a detected prohibited device, then simply being off-center.
    Returns a violation_type string, or None if nothing is wrong.

    Pulled out of the main loop so it can be unit-tested without a webcam
    or any model -- see test_main.py.
    """
    if count == 0:
        return "no_face"
    if count > 1:
        return "multiple_faces"
    if impersonation_flag:
        return "impersonation"
    if device_hits:
        return "device_detected"
    if not face_in_center:
        return "off_center"
    return None


def load_face_model(model_path: str):
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Face detection model not found at '{model_path}'. This model is required "
            f"(see README) -- the interview cannot run without it."
        )
    return YOLO(model_path)


def build_device_detector(model_path: str, confidence_threshold: float):
    """
    Returns a DeviceDetector, or None if the model isn't available. Device
    detection is treated as optional-but-important: a missing yolov8n.pt
    (which the README says is NOT bundled and must be fetched separately)
    should degrade the session to "no device detection" with a loud
    warning, not crash the whole interview.
    """
    if not os.path.exists(model_path):
        logger.warning(
            "Device model not found at '%s' -- device detection DISABLED for this session. "
            "See README for how to fetch yolov8n.pt.", model_path
        )
        return None
    try:
        return DeviceDetector(model_path, confidence_threshold=confidence_threshold)
    except Exception:
        logger.exception("Failed to load device detection model -- device detection DISABLED.")
        return None


def main():
    cfg = parse_args()
    setup_logging(cfg.log_dir)

    try:
        face_model = load_face_model(cfg.face_model_path)
    except Exception:
        logger.exception("Could not load face detection model. Exiting.")
        return

    device_detector = build_device_detector(cfg.device_model_path, cfg.device_confidence_threshold)
    identity = IdentityVerifier(
        mismatch_threshold=cfg.mismatch_threshold,
        consecutive_required=cfg.consecutive_required,
        enrollment_frames_required=cfg.enrollment_frames_required,
    )
    infractions = InfractionCounter(max_infractions=cfg.max_infractions)
    session_logger = SessionLogger(log_dir=cfg.log_dir, save_snapshots=cfg.save_snapshots)

    cap = cv2.VideoCapture(cfg.webcam_index)
    if not cap.isOpened():
        logger.error("Could not open webcam (index %d).", cfg.webcam_index)
        return

    logger.info("Interview session started. Please stay centered.")
    frame_count = 0
    terminated = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame from webcam. Ending session.")
                break

            frame_count += 1
            height, width, _ = frame.shape
            frame_center_x, frame_center_y = width // 2, height // 2
            margin_x, margin_y = width * 0.25, height * 0.25

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    results = face_model(frame)[0]
            except Exception:
                logger.exception("Face detection failed on this frame; skipping it.")
                continue

            faces = [b for b in results.boxes if b.conf > cfg.face_confidence_threshold]
            count = len(faces)
            face_in_center = False
            primary_box = None

            for b in faces:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if abs(cx - frame_center_x) < margin_x and abs(cy - frame_center_y) < margin_y:
                    face_in_center = True
                    primary_box = (x1, y1, x2, y2)

            # ---- Identity enrollment (multi-frame, once the candidate has settled in frame) ----
            enrollment_status = None
            if not identity.enrolled and count == 1 and face_in_center and frame_count >= cfg.enrollment_frames_to_wait:
                enrollment_status = identity.collect_enrollment_frame(frame, primary_box)
                if enrollment_status["enrolled"]:
                    logger.info("Identity enrolled for this session.")

            # ---- Identity verification (after enrollment) ----
            impersonation_flag = False
            if identity.enrolled and count == 1 and primary_box is not None:
                id_result = identity.verify(frame, primary_box)
                impersonation_flag = id_result["flag_impersonation"]

            # ---- Prohibited device detection ----
            device_hits = []
            if device_detector is not None:
                try:
                    device_hits = device_detector.detect(frame)
                except Exception:
                    logger.exception("Device detection failed on this frame; skipping it.")
                for d in device_hits:
                    x1, y1, x2, y2 = d["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, d["label"], (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            violation_type = determine_violation_type(count, face_in_center, impersonation_flag,
                                                        device_hits)

            timestamp = time.time()
            result = infractions.notify(violation_type, timestamp=timestamp)
            if result["new_infraction"]:
                alert_user(result["message"])
                session_logger.record_infraction(violation_type, timestamp, frame)
            if result["terminated"]:
                terminated = True
                cancel_interview(result["message"])
                cv2.putText(frame, "SESSION TERMINATED", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.imshow("AI Interview", frame)
                cv2.waitKey(2000)
                break

            # ---- On-screen status ----
            if not identity.enrolled and enrollment_status is not None:
                status = f"Enrolling identity ({enrollment_status['buffered']}/{enrollment_status['required']})"
            else:
                status = violation_type if violation_type else "OK"
            cv2.putText(frame, f"Status: {status} | Infractions: {infractions.count}/{cfg.max_infractions}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("AI Interview", frame)
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Ending session.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_path = session_logger.write_summary(infractions.log, terminated)
        logger.info("Session log written to %s", log_path)


if __name__ == "__main__":
    main()
