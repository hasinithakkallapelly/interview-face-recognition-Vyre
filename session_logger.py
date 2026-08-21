"""
session_logger.py

Persists what InfractionCounter only ever kept in memory: writes the
infraction log to a JSON file, and (optionally) saves a snapshot frame at
the moment each infraction is recorded, so a session can be reviewed after
the fact instead of only being visible in the live console output.

Each session gets its own subdirectory under `log_dir`, named by session
start time plus a short random suffix, so repeated runs don't overwrite
each other -- including two sessions started within the same second.
"""

import json
import os
import time
import uuid

import cv2


class SessionLogger:
    def __init__(self, log_dir: str = "session_logs", save_snapshots: bool = True):
        self.save_snapshots = save_snapshots
        session_name = time.strftime("session_%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        self.session_dir = os.path.join(log_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

    def record_infraction(self, violation_type: str, timestamp: float, frame=None) -> None:
        if self.save_snapshots and frame is not None:
            filename = f"infraction_{int(timestamp)}_{violation_type}.jpg"
            cv2.imwrite(os.path.join(self.session_dir, filename), frame)

    def write_summary(self, infractions_log: list, terminated: bool) -> str:
        """Writes the full infraction log to `<session_dir>/infractions.json`.
        Returns the path written to."""
        path = os.path.join(self.session_dir, "infractions.json")
        summary = {
            "terminated": terminated,
            "infraction_count": len(infractions_log),
            "infractions": [
                {"timestamp": ts, "violation_type": vtype}
                for ts, vtype in infractions_log
            ],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path
