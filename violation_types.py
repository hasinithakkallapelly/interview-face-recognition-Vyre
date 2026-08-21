"""
violation_types.py

Single source of truth for the violation-type string literals used across
main.py, infraction_counter.py, and their tests. Previously each one
("no_face", "off_center", etc.) was retyped as a bare string literal at
every call site, with nothing catching a typo at import time.

Reference ViolationType.X.value instead of retyping the literal.
Deliberately a plain Enum, not `str, Enum`: on this Python version a
`str`-mixed Enum member renders as "ViolationType.OFF_CENTER" in an
f-string (only str() does not), which would silently corrupt on-screen
status text and log messages -- so `.value` is used explicitly wherever
the plain string is needed, and infraction_counter.py / session_logger.py
keep working with plain strings exactly as before.
"""

from enum import Enum


class ViolationType(Enum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    IMPERSONATION = "impersonation"
    DEVICE_DETECTED = "device_detected"
    OFF_CENTER = "off_center"
