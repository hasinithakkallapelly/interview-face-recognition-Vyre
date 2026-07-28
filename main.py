"""
main.py

Real-time interview proctoring, combining:
  - Face presence/position monitoring (existing, from the original project)
  - Identity verification via LBPH        (NEW -- tested, see test_identity_verifier.py)
  - Progressive infraction counter        (NEW -- tested, see test_infraction_counter.py)
  - Prohibited device detection           (NEW -- NOT runtime-verified, see device_detector.py header)

Run: python main.py
Press ESC to quit manually. Session also auto-terminates after
max_infractions distinct violations (see infraction_counter.py).
"""

import cv2
import time
import contextlib
import io
from ultralytics import YOLO

from utils import alert_user, cancel_interview
from identity_verifier import IdentityVerifier
from infraction_counter import InfractionCounter
from device_detector import DeviceDetector

FACE_MODEL_PATH = "models/yolov8n-face.pt"
DEVICE_MODEL_PATH = "models/yolov8n.pt"  # general COCO model, NOT the face model
MAX_INFRACTIONS = 3
ENROLLMENT_FRAMES_TO_WAIT = 15  # let the candidate settle in frame before enrolling identity


def main():
    face_model = YOLO(FACE_MODEL_PATH)
    device_detector = DeviceDetector(DEVICE_MODEL_PATH)
    identity = IdentityVerifier()
    infractions = InfractionCounter(max_infractions=MAX_INFRACTIONS)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Interview session started. Please stay centered.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame from webcam.")
            break

        frame_count += 1
        height, width, _ = frame.shape
        frame_center_x, frame_center_y = width // 2, height // 2
        margin_x, margin_y = width * 0.25, height * 0.25

        with contextlib.redirect_stdout(io.StringIO()):
            results = face_model(frame)[0]

        faces = [b for b in results.boxes if b.conf > 0.5]
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

        # ---- Identity enrollment (once, after candidate has settled in frame) ----
        if not identity.enrolled and count == 1 and face_in_center and frame_count >= ENROLLMENT_FRAMES_TO_WAIT:
            if identity.enroll(frame, primary_box):
                print("[INFO] Identity enrolled for this session.")

        # ---- Identity verification (after enrollment) ----
        impersonation_flag = False
        if identity.enrolled and count == 1 and primary_box is not None:
            id_result = identity.verify(frame, primary_box)
            impersonation_flag = id_result["flag_impersonation"]

        # ---- Prohibited device detection ----
        device_hits = device_detector.detect(frame)
        for d in device_hits:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, d["label"], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ---- Determine current violation type (priority order matters) ----
        violation_type = None
        if count == 0:
            violation_type = "no_face"
        elif count > 1:
            violation_type = "multiple_faces"
        elif impersonation_flag:
            violation_type = "impersonation"
        elif device_hits:
            violation_type = "device_detected"
        elif not face_in_center:
            violation_type = "off_center"

        result = infractions.notify(violation_type, timestamp=time.time())
        if result["new_infraction"]:
            alert_user(result["message"])
        if result["terminated"]:
            cancel_interview(result["message"])
            cv2.putText(frame, "SESSION TERMINATED", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.imshow("AI Interview", frame)
            cv2.waitKey(2000)
            break

        # ---- On-screen status ----
        status = violation_type if violation_type else "OK"
        cv2.putText(frame, f"Status: {status} | Infractions: {infractions.count}/{MAX_INFRACTIONS}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("AI Interview", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
