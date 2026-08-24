import cv2
import numpy as np
import pytest

from identity_verifier import IdentityVerifier


def make_pattern(seed, size=(240, 240)):
    rng = np.random.default_rng(seed)
    image = np.zeros((*size, 3), dtype=np.uint8)
    cv2.circle(image, (120, 120), 80, (180, 150, 130), -1)
    cv2.circle(image, (90, 100), 10, (30, 30, 30), -1)
    cv2.circle(image, (150, 100), 10, (30, 30, 30), -1)
    return cv2.add(image, rng.integers(0, 15, image.shape, dtype=np.uint8))


def test_multiple_reference_samples_enroll_and_match():
    verifier = IdentityVerifier(mismatch_threshold=80, consecutive_required=3)
    samples = [(make_pattern(42), (20, 20, 220, 220)) for _ in range(3)]
    assert verifier.enroll_many(samples) == 3
    assert verifier.verify(make_pattern(42), (20, 20, 220, 220))["matched"] is True


def test_invalid_reference_samples_are_rejected():
    verifier = IdentityVerifier()
    with pytest.raises(ValueError):
        verifier.enroll_many([(make_pattern(1), (20, 20, 20, 20))])


def test_mismatch_requires_consecutive_frames():
    verifier = IdentityVerifier(mismatch_threshold=30, consecutive_required=3)
    verifier.enroll_many([(make_pattern(42), (20, 20, 220, 220))])
    other = make_pattern(999)
    assert verifier.verify(other, (20, 20, 220, 220))["flag_impersonation"] is False
    assert verifier.verify(other, (20, 20, 220, 220))["flag_impersonation"] is False
    assert verifier.verify(other, (20, 20, 220, 220))["flag_impersonation"] is True

