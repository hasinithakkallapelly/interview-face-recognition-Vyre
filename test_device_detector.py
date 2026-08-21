"""
Tests DeviceDetector's class-ID resolution and filtering logic using a
fake model object (injected via the `model=` param added to
DeviceDetector.__init__), instead of a real ultralytics YOLO model. This
verifies the detection logic itself without needing PyTorch installed --
the actual YOLO/COCO inference path is still not exercised here and
should be verified separately per the README (real webcam + real model).
"""

from device_detector import DeviceDetector


class FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stands in for an ultralytics YOLO model: has `.names` and is
    callable, returning a list with one results object (matching the
    real API's `model(frame)[0]` usage)."""

    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def __call__(self, frame, verbose=False):
        return [FakeResults(self._boxes)]


# Deliberately shuffled/non-standard IDs, to confirm resolution happens
# via model.names rather than hardcoded COCO IDs (see device_detector.py
# docstring on this exact risk).
FAKE_NAMES = {0: "person", 1: "cell phone", 2: "chair", 3: "book", 4: "laptop", 5: "dog"}


def test_only_prohibited_classes_are_returned():
    boxes = [
        FakeBox(0, 0.9, (0, 0, 10, 10)),   # person -- not prohibited
        FakeBox(1, 0.8, (1, 1, 11, 11)),   # cell phone -- prohibited
        FakeBox(5, 0.95, (2, 2, 12, 12)),  # dog -- not prohibited
    ]
    detector = DeviceDetector(model=FakeModel(FAKE_NAMES, boxes), confidence_threshold=0.4)
    detections = detector.detect(frame=None)
    assert len(detections) == 1
    assert detections[0]["label"] == "cell phone"


def test_confidence_threshold_filters_low_confidence_hits():
    boxes = [FakeBox(3, 0.1, (0, 0, 5, 5))]  # book, but below threshold
    detector = DeviceDetector(model=FakeModel(FAKE_NAMES, boxes), confidence_threshold=0.4)
    assert detector.detect(frame=None) == []


def test_multiple_prohibited_hits_all_returned():
    boxes = [
        FakeBox(1, 0.6, (0, 0, 5, 5)),
        FakeBox(3, 0.7, (5, 5, 10, 10)),
        FakeBox(4, 0.9, (10, 10, 15, 15)),
    ]
    detector = DeviceDetector(model=FakeModel(FAKE_NAMES, boxes), confidence_threshold=0.4)
    labels = {d["label"] for d in detector.detect(frame=None)}
    assert labels == {"cell phone", "book", "laptop"}


def test_prohibited_ids_resolved_from_model_names_not_hardcoded():
    # Standard COCO has cell phone=67, but this fake model uses id=1 --
    # if the detector hardcoded 67 this would wrongly find nothing.
    detector = DeviceDetector(model=FakeModel(FAKE_NAMES, []), confidence_threshold=0.4)
    assert detector.prohibited_ids == {1, 3, 4}


if __name__ == "__main__":
    test_only_prohibited_classes_are_returned()
    test_confidence_threshold_filters_low_confidence_hits()
    test_multiple_prohibited_hits_all_returned()
    test_prohibited_ids_resolved_from_model_names_not_hardcoded()
    print("All device_detector.py tests passed (fake model -- verify real YOLO inference yourself).")
