"""
device_detector.py

Prohibited device detection: phones, books, laptops in frame during an
interview.

*** NOT RUNTIME-VERIFIED ***
Unlike identity_verifier.py and infraction_counter.py, this module was
NOT actually run in the environment this was built in -- `ultralytics`
requires PyTorch (~500MB+ of downloads), which wasn't feasible to fully
install and test there. This is written to be correct based on the
documented ultralytics API and standard COCO class usage, but you must
verify it runs on your own machine before relying on it or claiming
"tested" anywhere. Treat this file as a strong draft, not a proven one --
that distinction matters if you're asked about it.

HOW IT WORKS:
- Loads a SEPARATE, general-purpose YOLOv8n model (COCO-pretrained,
  `yolov8n.pt` -- NOT the face-only `yolov8n-face.pt` model already used
  elsewhere in this project). The face model can only detect faces; COCO
  classes are needed to recognize objects like phones and books.
- COCO class IDs relevant here (fixed, standard COCO ordering):
    67 = cell phone, 73 = book, 63 = laptop
  (Other COCO models/versions may order classes differently -- always
  print `model.names` to confirm before trusting these IDs on a new
  model file.)
- Runs inference on the same frame already captured for face detection,
  filtering for just these class IDs above a confidence threshold.
"""

from ultralytics import YOLO

PROHIBITED_CLASS_NAMES = {"cell phone", "book", "laptop"}


class DeviceDetector:
    def __init__(self, model_path="models/yolov8n.pt", confidence_threshold=0.4):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Build a set of class IDs matching PROHIBITED_CLASS_NAMES by
        # reading model.names directly, rather than hardcoding IDs --
        # this avoids the "IDs might differ by model version" trap noted
        # above.
        self.prohibited_ids = {
            class_id for class_id, name in self.model.names.items()
            if name in PROHIBITED_CLASS_NAMES
        }

    def detect(self, frame) -> list:
        """Returns a list of dicts for each prohibited object detected
        this frame: [{"label": str, "confidence": float, "box": (x1,y1,x2,y2)}, ...]
        Empty list if none detected."""
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id in self.prohibited_ids and confidence >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": self.model.names[class_id],
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                })

        return detections
