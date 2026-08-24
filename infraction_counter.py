from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InfractionCounter:
    max_infractions: int = 3
    minimum_duration: float = 2.0
    cooldown: float = 5.0
    count: int = 0
    log: list = field(default_factory=list)
    _candidate_violation: Optional[str] = None
    _candidate_since: Optional[float] = None
    _counted_continuous_event: bool = False
    _last_counted_at: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.max_infractions < 1:
            raise ValueError("max_infractions must be at least 1")
        if self.minimum_duration < 0 or self.cooldown < 0:
            raise ValueError("timing values cannot be negative")

    def notify(self, violation_type: Optional[str], timestamp: float) -> dict:
        result = {"new_infraction": False, "terminated": False,
                  "count": self.count, "message": None}

        if violation_type is None:
            self._candidate_violation = None
            self._candidate_since = None
            self._counted_continuous_event = False
            return result

        if violation_type != self._candidate_violation:
            self._candidate_violation = violation_type
            self._candidate_since = timestamp
            self._counted_continuous_event = False
            return result

        if self._counted_continuous_event:
            return result
        if timestamp - self._candidate_since < self.minimum_duration:
            return result

        last = self._last_counted_at.get(violation_type)
        if last is not None and timestamp - last < self.cooldown:
            return result

        self._counted_continuous_event = True
        self._last_counted_at[violation_type] = timestamp
        self.count += 1
        self.log.append((timestamp, violation_type))
        result.update(new_infraction=True, count=self.count)

        if self.count >= self.max_infractions:
            result["terminated"] = True
            result["message"] = (
                f"Session terminated: infraction {self.count}/{self.max_infractions} "
                f"({violation_type})."
            )
        else:
            result["message"] = (
                f"Warning {self.count}/{self.max_infractions}: {violation_type}."
            )
        return result

    def reset(self):
        self.count = 0
        self.log.clear()
        self._candidate_violation = None
        self._candidate_since = None
        self._counted_continuous_event = False
        self._last_counted_at.clear()

