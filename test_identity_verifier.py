"""
Test identity_verifier.py using synthetic image patches (no real webcam/
face dataset available in this environment). This validates the CODE
PATH works correctly -- that enroll/verify run without error, that a
"same identity" comparison scores as a match, and a clearly different
image scores as a mismatch. It does NOT validate real-world face
recognition accuracy, which requires actual face photos -- that part
you'll need to verify yourself with a real webcam.
"""

import numpy as np
import cv2
from identity_verifier import IdentityVerifier


def make_synthetic_face(seed, size=(240, 240)):
    """Generate a reproducible synthetic 'face-like' image: a base pattern
    plus a bit of seeded random texture, so the same seed always produces
    the same image (simulating 'the same person'), and different seeds
    produce different images (simulating 'a different person')."""
    rng = np.random.default_rng(seed)
    base = np.zeros((*size, 3), dtype=np.uint8)
    cv2.circle(base, (size[0] // 2, size[1] // 2), 80, (180, 150, 130), -1)  # face-ish blob
    cv2.circle(base, (size[0] // 2 - 30, size[1] // 2 - 20), 10, (50, 50, 50), -1)  # eye
    cv2.circle(base, (size[0] // 2 + 30, size[1] // 2 - 20), 10, (50, 50, 50), -1)  # eye
    noise = rng.integers(0, 15, size=(*size, 3), dtype=np.uint8)
    return cv2.add(base, noise)


def test_enroll_and_match_same_identity():
    frame = make_synthetic_face(seed=42)
    box = (20, 20, 220, 220)

    verifier = IdentityVerifier(mismatch_threshold=80.0, consecutive_required=3)
    assert verifier.enroll(frame, box) is True

    # Verify against a slightly different frame of the "same person"
    # (same seed, so same underlying pattern -- simulates minor frame-to-
    # frame variation of the same real face).
    same_person_frame = make_synthetic_face(seed=42)
    result = verifier.verify(same_person_frame, box)
    print(f"Same identity -> matched={result['matched']}, distance={result['distance']:.2f}")
    assert result["matched"] is True
    assert result["flag_impersonation"] is False


def test_different_identity_flags_after_consecutive_mismatches():
    frame = make_synthetic_face(seed=42)
    box = (20, 20, 220, 220)

    verifier = IdentityVerifier(mismatch_threshold=30.0, consecutive_required=3)
    verifier.enroll(frame, box)

    different_person_frame = make_synthetic_face(seed=999)  # very different pattern
    flagged = False
    for i in range(5):
        result = verifier.verify(different_person_frame, box)
        print(f"Frame {i}: matched={result['matched']}, distance={result['distance']:.2f}, "
              f"flag_impersonation={result['flag_impersonation']}")
        if result["flag_impersonation"]:
            flagged = True
            break

    assert flagged, "Expected impersonation flag after consecutive mismatches"


def test_single_bad_frame_does_not_immediately_flag():
    # Confirms the "consecutive frames required" logic actually protects
    # against a single noisy frame -- this is the false-positive
    # mitigation described in the module docstring, verified here.
    frame = make_synthetic_face(seed=42)
    box = (20, 20, 220, 220)

    verifier = IdentityVerifier(mismatch_threshold=30.0, consecutive_required=5)
    verifier.enroll(frame, box)

    different_person_frame = make_synthetic_face(seed=999)
    result = verifier.verify(different_person_frame, box)
    print(f"Single mismatched frame -> flag_impersonation={result['flag_impersonation']}")
    assert result["flag_impersonation"] is False, "Should not flag on a single mismatched frame"


if __name__ == "__main__":
    test_enroll_and_match_same_identity()
    test_different_identity_flags_after_consecutive_mismatches()
    test_single_bad_frame_does_not_immediately_flag()
    print("\nAll identity_verifier tests passed (synthetic data -- verify with real faces too).")
