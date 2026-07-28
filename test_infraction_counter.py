from infraction_counter import InfractionCounter


def test_progressive_warnings_then_termination():
    ic = InfractionCounter(max_infractions=3)

    r1 = ic.notify("off_center", timestamp=1.0)
    assert r1["new_infraction"] is True and r1["terminated"] is False and r1["count"] == 1
    print("Infraction 1:", r1["message"])

    r2 = ic.notify("multiple_faces", timestamp=5.0)
    assert r2["new_infraction"] is True and r2["terminated"] is False and r2["count"] == 2
    print("Infraction 2:", r2["message"])

    r3 = ic.notify("no_face", timestamp=9.0)
    assert r3["new_infraction"] is True and r3["terminated"] is True and r3["count"] == 3
    print("Infraction 3:", r3["message"])


def test_ongoing_violation_not_double_counted():
    # A candidate off-center continuously across many ticks should count
    # as ONE infraction, not one per tick.
    ic = InfractionCounter(max_infractions=3)
    for t in range(10):
        ic.notify("off_center", timestamp=float(t))
    assert ic.count == 1, f"Expected 1 infraction for one continuous violation, got {ic.count}"
    print(f"10 ticks of continuous off_center -> count={ic.count} (correct: should be 1)")


def test_violation_clearing_allows_recount():
    ic = InfractionCounter(max_infractions=5)
    ic.notify("off_center", timestamp=1.0)   # infraction 1
    ic.notify(None, timestamp=2.0)           # violation cleared
    ic.notify("off_center", timestamp=3.0)   # infraction 2 (new occurrence, same type)
    assert ic.count == 2, f"Expected 2 infractions after clear+re-trigger, got {ic.count}"
    print(f"Off-center, cleared, off-center again -> count={ic.count} (correct: should be 2)")


if __name__ == "__main__":
    test_progressive_warnings_then_termination()
    test_ongoing_violation_not_double_counted()
    test_violation_clearing_allows_recount()
    print("\nAll infraction_counter tests passed.")
