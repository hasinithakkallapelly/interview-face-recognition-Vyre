"""
Tests for determine_violation_type() in main.py -- the priority-ordered
logic that decides which single violation "wins" when multiple signals
are present in a frame. This is the piece that ties every other module
together, but previously had zero test coverage since it only lived
inline in the camera loop.
"""

from main import determine_violation_type


def test_no_face_wins_over_everything():
    assert determine_violation_type(0, face_in_center=False, impersonation_flag=True,
                                     device_hits=[{"label": "cell phone"}]) == "no_face"


def test_multiple_faces_beats_impersonation_and_devices():
    assert determine_violation_type(2, face_in_center=True, impersonation_flag=True,
                                     device_hits=[{"label": "book"}]) == "multiple_faces"


def test_impersonation_beats_device_and_off_center():
    assert determine_violation_type(1, face_in_center=True, impersonation_flag=True,
                                     device_hits=[{"label": "laptop"}]) == "impersonation"


def test_device_detected_beats_off_center():
    assert determine_violation_type(1, face_in_center=False, impersonation_flag=False,
                                     device_hits=[{"label": "book"}]) == "device_detected"


def test_off_center_when_nothing_else_wrong():
    assert determine_violation_type(1, face_in_center=False, impersonation_flag=False,
                                     device_hits=[]) == "off_center"


def test_none_when_everything_ok():
    assert determine_violation_type(1, face_in_center=True, impersonation_flag=False,
                                     device_hits=[]) is None


if __name__ == "__main__":
    test_no_face_wins_over_everything()
    test_multiple_faces_beats_impersonation_and_devices()
    test_impersonation_beats_device_and_off_center()
    test_device_detected_beats_off_center()
    test_off_center_when_nothing_else_wrong()
    test_none_when_everything_ok()
    print("All main.py violation-priority tests passed.")
