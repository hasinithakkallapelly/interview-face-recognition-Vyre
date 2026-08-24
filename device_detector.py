from pathlib import Path

from ultralytics import YOLO


PROHIBITED_CLASS_NAMES = {"cell phone", "book", "laptop"}


class DeviceDetector:
    def __init__(self, model_path, confidence_threshold=0.4):
        # A local path is used when present. A bare Ultralytics model name
        # (for example ``yolov8n.pt``) is also supported and downloaded by
        # Ultralytics on first use.
        model_path = Path(model_path)
        if not model_path.is_file() and model_path.parent != Path("."):
            raise FileNotFoundError(f"Object-detection model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.prohibited_ids = {
            class_id for class_id, name in self.model.names.items()
            if name in PROHIBITED_CLASS_NAMES
        }
        if not self.prohibited_ids:
            raise ValueError("The selected model has no supported prohibited-object classes")

    def detect(self, frame) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id in self.prohibited_ids and confidence >= self.confidence_threshold:
                detections.append({
                    "label": self.model.names[class_id],
                    "confidence": confidence,
                    "box": tuple(map(int, box.xyxy[0])),
                })
        return detections
