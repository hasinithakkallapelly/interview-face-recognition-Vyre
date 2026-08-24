import argparse
import contextlib
import io
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from device_detector import DeviceDetector
from identity_verifier import IdentityVerifier
from infraction_counter import InfractionCounter
from utils import alert_user, cancel_interview


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FACE_MODEL = PROJECT_ROOT / "models" / "yolov8n-face.pt"
DEFAULT_DEVICE_MODEL = "yolov8n.pt"


def _detect_faces(model, frame, confidence=0.5):
    with contextlib.redirect_stdout(io.StringIO()):
        results = model(frame, verbose=False)[0]
    return [tuple(map(int, box.xyxy[0])) for box in results.boxes
            if float(box.conf[0]) >= confidence]


def _largest_box(boxes):
    return max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))


def _load_reference_samples(paths, face_model):
    samples = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read reference image: {path}")
        faces = _detect_faces(face_model, image)
        if len(faces) != 1:
            raise ValueError(
                f"Reference image {path} must contain exactly one detectable face; found {len(faces)}"
            )
        samples.append((image, faces[0]))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time interview monitoring")
    parser.add_argument("--reference-image", action="append", required=True,
                        help="Pre-session candidate image; repeat for multiple images")
    parser.add_argument("--face-model", default=str(DEFAULT_FACE_MODEL))
    parser.add_argument("--device-model", default=str(DEFAULT_DEVICE_MODEL))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--object-every", type=int, default=5,
                        help="Run object detection once every N frames")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.object_every < 1:
        raise ValueError("--object-every must be at least 1")
    if not Path(args.face_model).is_file():
        raise FileNotFoundError(f"Required face model not found: {args.face_model}")

    face_model = YOLO(args.face_model)
    device_detector = DeviceDetector(args.device_model)
    identity = IdentityVerifier()
    identity.enroll_many(_load_reference_samples(args.reference_image, face_model))
    infractions = InfractionCounter(max_infractions=3, minimum_duration=2.0, cooldown=5.0)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    frame_count = 0
    device_hits = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")
            frame_count += 1
            height, width = frame.shape[:2]
            faces = _detect_faces(face_model, frame)

            primary_box = _largest_box(faces) if len(faces) == 1 else None
            face_in_center = False
            for x1, y1, x2, y2 in faces:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                face_in_center |= (abs(cx - width // 2) < width * 0.25 and
                                   abs(cy - height // 2) < height * 0.25)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            impersonation = False
            if primary_box is not None:
                impersonation = identity.verify(frame, primary_box)["flag_impersonation"]

            if frame_count % args.object_every == 1:
                device_hits = device_detector.detect(frame)
            for hit in device_hits:
                x1, y1, x2, y2 = hit["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, hit["label"], (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            violation = None
            if not faces:
                violation = "no_face"
            elif len(faces) > 1:
                violation = "multiple_faces"
            elif impersonation:
                violation = "identity_mismatch"
            elif device_hits:
                violation = "device_detected"
            elif not face_in_center:
                violation = "off_center"

            result = infractions.notify(violation, time.monotonic())
            if result["new_infraction"]:
                alert_user(result["message"])
            if result["terminated"]:
                cancel_interview(result["message"])
                break

            status = violation or "OK"
            cv2.putText(frame, f"Status: {status} | Infractions: {infractions.count}/3",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("AI Interview", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
