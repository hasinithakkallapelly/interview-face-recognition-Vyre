"""
infraction_counter.py

A proper progressive-warning infraction system, replacing the old
approach where each violation type (multiple faces / no face / off-
center) had its own independent timer that cancelled the session
immediately on its own threshold. That meant, e.g., one single
uninterrupted 3-second glance away could end the interview outright --
no warnings, no accumulation across different violation types.

This version tracks infractions as discrete EVENTS across all violation
types, issues a warning on each of the first N-1 infractions, and only
terminates the session on the Nth (matching what was originally claimed
on the resume but never implemented: "progressive warnings and
auto-terminating sessions upon three infractions").

An infraction is only counted once per continuous violation -- e.g. a
candidate being off-center for 5 straight seconds is ONE infraction, not
one per frame. A new infraction of the same type can't be counted again
until the violation condition clears first (see `notify` logic).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InfractionCounter:
    max_infractions: int = 3
    count: int = 0
    log: list = field(default_factory=list)
    _active_violation: Optional[str] = None  # tracks in-progress violation type

    def notify(self, violation_type: Optional[str], timestamp: float) -> dict:
        """
        Call this every frame/tick with the CURRENT violation type (a
        string like "multiple_faces", "no_face", "off_center", "device_detected",
        "impersonation") or None if no violation is currently active.

        Returns a dict describing what happened this call:
          {"new_infraction": bool, "terminated": bool, "count": int,
           "message": str or None}
        """
        result = {"new_infraction": False, "terminated": False,
                  "count": self.count, "message": None}

        if violation_type is None:
            # violation cleared -- allow the SAME type to count again next time
            self._active_violation = None
            return result

        if violation_type == self._active_violation:
            # still the same ongoing violation -- don't double-count it
            return result

        # a NEW violation (different from whatever was active, including
        # "none was active before")
        self._active_violation = violation_type
        self.count += 1
        self.log.append((timestamp, violation_type))
        result["new_infraction"] = True
        result["count"] = self.count

        if self.count >= self.max_infractions:
            result["terminated"] = True
            result["message"] = (
                f"Session terminated: infraction {self.count}/{self.max_infractions} "
                f"({violation_type})."
            )
        else:
            result["message"] = (
                f"Warning {self.count}/{self.max_infractions}: {violation_type}. "
                f"Session will be terminated after {self.max_infractions} infractions."
            )

        return result

    def reset(self):
        self.count = 0
        self.log = []
        self._active_violation = None
