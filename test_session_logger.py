"""
Tests for session_logger.py -- previously only verified manually in a
shell, not part of the automated test suite. Uses a temp directory so it
doesn't leave real session_logs/ output behind.
"""

import json
import os
import shutil
import tempfile

import numpy as np

from session_logger import SessionLogger


def test_write_summary_produces_expected_json():
    tmp_dir = tempfile.mkdtemp()
    try:
        logger = SessionLogger(log_dir=tmp_dir, save_snapshots=False)
        path = logger.write_summary(
            infractions_log=[(1.0, "off_center"), (5.5, "no_face")],
            terminated=True,
        )
        assert os.path.exists(path)
        with open(path) as f:
            summary = json.load(f)
        assert summary["terminated"] is True
        assert summary["infraction_count"] == 2
        assert summary["infractions"] == [
            {"timestamp": 1.0, "violation_type": "off_center"},
            {"timestamp": 5.5, "violation_type": "no_face"},
        ]
        print(f"write_summary -> {path}, {summary}")
    finally:
        shutil.rmtree(tmp_dir)


def test_record_infraction_saves_snapshot_when_enabled():
    tmp_dir = tempfile.mkdtemp()
    try:
        logger = SessionLogger(log_dir=tmp_dir, save_snapshots=True)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        logger.record_infraction("off_center", 123.0, frame)

        snapshot_files = [f for f in os.listdir(logger.session_dir) if f.startswith("infraction_")]
        assert len(snapshot_files) == 1
        assert "off_center" in snapshot_files[0]
        print(f"snapshot saved -> {snapshot_files[0]}")
    finally:
        shutil.rmtree(tmp_dir)


def test_record_infraction_skips_snapshot_when_disabled():
    tmp_dir = tempfile.mkdtemp()
    try:
        logger = SessionLogger(log_dir=tmp_dir, save_snapshots=False)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        logger.record_infraction("off_center", 123.0, frame)

        assert os.listdir(logger.session_dir) == []
        print("save_snapshots=False -> no file written, as expected")
    finally:
        shutil.rmtree(tmp_dir)


def test_record_infraction_is_a_noop_without_a_frame():
    tmp_dir = tempfile.mkdtemp()
    try:
        logger = SessionLogger(log_dir=tmp_dir, save_snapshots=True)
        logger.record_infraction("off_center", 123.0, frame=None)
        assert os.listdir(logger.session_dir) == []
    finally:
        shutil.rmtree(tmp_dir)


def test_concurrent_sessions_get_distinct_directories():
    # Guards against the directory-collision risk that motivated adding a
    # uuid suffix to the session directory name (see session_logger.py).
    tmp_dir = tempfile.mkdtemp()
    try:
        logger1 = SessionLogger(log_dir=tmp_dir, save_snapshots=False)
        logger2 = SessionLogger(log_dir=tmp_dir, save_snapshots=False)
        assert logger1.session_dir != logger2.session_dir
        assert os.path.isdir(logger1.session_dir)
        assert os.path.isdir(logger2.session_dir)
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_write_summary_produces_expected_json()
    test_record_infraction_saves_snapshot_when_enabled()
    test_record_infraction_skips_snapshot_when_disabled()
    test_record_infraction_is_a_noop_without_a_frame()
    test_concurrent_sessions_get_distinct_directories()
    print("\nAll session_logger.py tests passed.")
