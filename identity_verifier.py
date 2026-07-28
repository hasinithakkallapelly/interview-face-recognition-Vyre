"""
identity_verifier.py

Identity verification for interview proctoring, using OpenCV's built-in
LBPH (Local Binary Patterns Histograms) face recognizer.

WHY LBPH AND NOT DLIB 128D EMBEDDINGS:
The original resume claimed dlib-based 128D facial embeddings, but that
was never actually implemented -- there was no dlib code anywhere in the
project. dlib is also a heavy, slow-to-compile dependency (needs cmake +
a C++ toolchain) that isn't guaranteed to install cleanly everywhere.
LBPH is already available via opencv-contrib-python (which this project
already depends on through cv2), is much lighter, and is a legitimate,
well-established face recognition technique -- just less state-of-the-art
than a deep embedding model. This is an honest tradeoff, not a downgrade
to hide: if asked "why not dlib/FaceNet embeddings," the answer is
dependency weight and install reliability vs. verification accuracy.

HOW IT WORKS:
1. enroll(frame, face_box) -- at the start of a session, crop the
   candidate's face, convert to grayscale (LBPH requires grayscale
   input), and train the recognizer on that single reference face.
2. verify(frame, face_box) -- during the session, crop the current face
   the same way and ask the trained recognizer how confident it is that
   this matches the enrolled identity. LBPH's predict() returns a
   confidence score where LOWER = more similar (it's a distance, not a
   probability) -- a common source of confusion, documented here.

LIMITATIONS (being upfront):
- LBPH is sensitive to lighting changes and pose -- a candidate turning
  their head or a lighting change mid-interview can raise the distance
  score even with no impersonation happening. This implementation uses
  a generous default threshold and requires several consecutive
  mismatches (not just one frame) before flagging impersonation, to
  reduce false positives from this.
- Enrolled on a SINGLE reference frame. A production system would
  enroll from multiple frames/angles for a more robust reference.
"""

import cv2
import numpy as np


class IdentityVerifier:
    def __init__(self, mismatch_threshold: float = 80.0, consecutive_required: int = 5):
        """
        mismatch_threshold: LBPH distance above which a face is considered
            NOT a match. Lower distance = more similar. 80 is a
            commonly-used starting point for LBPH; tune per-deployment.
        consecutive_required: how many consecutive frames must exceed the
            threshold before we actually flag impersonation, to avoid
            reacting to a single noisy frame.
        """
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.enrolled = False
        self.mismatch_threshold = mismatch_threshold
        self.consecutive_required = consecutive_required
        self._consecutive_mismatches = 0

    @staticmethod
    def _crop_and_prep(frame, box, size=(200, 200)):
        x1, y1, x2, y2 = box
        face = frame[max(0, y1):y2, max(0, x1):x2]
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalize lighting variation somewhat
        gray = cv2.resize(gray, size)
        return gray

    def enroll(self, frame, box) -> bool:
        """Train the recognizer on the candidate's reference face. Returns
        True on success, False if the crop was invalid (e.g. box out of frame)."""
        gray = self._crop_and_prep(frame, box)
        if gray is None:
            return False
        self.recognizer.train([gray], np.array([0]))  # label 0 = "the candidate"
        self.enrolled = True
        self._consecutive_mismatches = 0
        return True

    def verify(self, frame, box) -> dict:
        """
        Returns a dict: {"matched": bool, "distance": float or None,
        "flag_impersonation": bool}.

        flag_impersonation only becomes True after `consecutive_required`
        consecutive mismatched frames, not on a single frame.
        """
        if not self.enrolled:
            return {"matched": None, "distance": None, "flag_impersonation": False}

        gray = self._crop_and_prep(frame, box)
        if gray is None:
            return {"matched": None, "distance": None, "flag_impersonation": False}

        label, distance = self.recognizer.predict(gray)
        matched = distance <= self.mismatch_threshold

        if matched:
            self._consecutive_mismatches = 0
        else:
            self._consecutive_mismatches += 1

        flag = self._consecutive_mismatches >= self.consecutive_required

        return {"matched": matched, "distance": distance, "flag_impersonation": flag}
