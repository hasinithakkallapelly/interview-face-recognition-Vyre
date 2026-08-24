from infraction_counter import InfractionCounter


def test_transient_detection_is_ignored():
    counter = InfractionCounter(minimum_duration=2.0)
    counter.notify("no_face", 0.0)
    result = counter.notify(None, 0.5)
    assert result["new_infraction"] is False
    assert counter.count == 0


def test_continuous_violation_counts_once_after_duration():
    counter = InfractionCounter(minimum_duration=2.0)
    counter.notify("off_center", 0.0)
    assert counter.notify("off_center", 1.9)["new_infraction"] is False
    assert counter.notify("off_center", 2.0)["new_infraction"] is True
    assert counter.notify("off_center", 20.0)["new_infraction"] is False
    assert counter.count == 1


def test_cooldown_blocks_immediate_repeat():
    counter = InfractionCounter(minimum_duration=1.0, cooldown=5.0)
    counter.notify("no_face", 0.0)
    counter.notify("no_face", 1.0)
    counter.notify(None, 1.1)
    counter.notify("no_face", 2.0)
    assert counter.notify("no_face", 3.0)["new_infraction"] is False
    counter.notify(None, 6.1)
    counter.notify("no_face", 6.2)
    assert counter.notify("no_face", 7.2)["new_infraction"] is True


def test_terminates_on_third_confirmed_event():
    counter = InfractionCounter(max_infractions=3, minimum_duration=1.0, cooldown=0)
    result = None
    for index, violation in enumerate(("no_face", "off_center", "device_detected")):
        start = index * 3.0
        counter.notify(violation, start)
        result = counter.notify(violation, start + 1.0)
        counter.notify(None, start + 1.1)
    assert result["terminated"] is True
    assert counter.count == 3

