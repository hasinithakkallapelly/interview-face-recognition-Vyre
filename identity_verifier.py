import cv2
import numpy as np


class IdentityVerifier:
    """Session identity verifier backed by OpenCV LBPH.

    Reference face crops must come from images captured before the monitored
    session. Live webcam frames are used only for verification.
    """

    def __init__(self, mismatch_threshold: float = 70.0, consecutive_required: int = 8):
        if consecutive_required < 1:
            raise ValueError("consecutive_required must be at least 1")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.enrolled = False
        self.mismatch_threshold = mismatch_threshold
        self.consecutive_required = consecutive_required
        self._consecutive_mismatches = 0

    @staticmethod
    def _crop_and_prep(frame, box, size=(200, 200)):
        if frame is None or box is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.resize(gray, size)

    def enroll_many(self, samples) -> int:
        """Train on ``(frame, face_box)`` reference samples.

        Returns the number of usable samples and raises when none are valid.
        """
        prepared = []
        for frame, box in samples:
            face = self._crop_and_prep(frame, box)
            if face is not None:
                prepared.append(face)
        if not prepared:
            raise ValueError("No valid reference face samples were supplied")
        labels = np.zeros(len(prepared), dtype=np.int32)
        self.recognizer.train(prepared, labels)
        self.enrolled = True
        self._consecutive_mismatches = 0
        return len(prepared)

    def enroll(self, frame, box) -> bool:
        """Compatibility wrapper for one reference sample."""
        try:
            self.enroll_many([(frame, box)])
            return True
        except ValueError:
            return False

    def verify(self, frame, box) -> dict:
        if not self.enrolled:
            return {"matched": None, "distance": None, "flag_impersonation": False}
        gray = self._crop_and_prep(frame, box)
        if gray is None:
            return {"matched": None, "distance": None, "flag_impersonation": False}

        _, distance = self.recognizer.predict(gray)
        distance = float(distance)
        matched = distance <= self.mismatch_threshold
        self._consecutive_mismatches = 0 if matched else self._consecutive_mismatches + 1
        return {
            "matched": matched,
            "distance": distance,
            "flag_impersonation": self._consecutive_mismatches >= self.consecutive_required,
        }

