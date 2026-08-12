from bearvision.simulation import BehavioralSimulation, Event


def test_same_seed_and_input_produce_identical_trace() -> None:
    def execute():
        simulation = BehavioralSimulation(duration_s=3_600, seed=42)
        simulation.schedule(Event(10, "tag_enters_range", {"tag_id": "tag-17"}))
        simulation.schedule(Event(10, "person_detected", {"confidence": 0.91}))
        return simulation.run()

    assert execute() == execute()


def test_events_at_same_time_keep_insertion_order() -> None:
    simulation = BehavioralSimulation(duration_s=10)
    simulation.schedule(Event(2, "first"))
    simulation.schedule(Event(2, "second"))

    trace = simulation.run()

    assert [entry.kind for entry in trace] == ["first", "second"]


def test_handler_can_schedule_component_response() -> None:
    simulation = BehavioralSimulation(duration_s=10)

    def detect_person(event, _simulation):
        return [Event(event.at_s + 0.3, "capture_triggered", {"camera": "edge-1"})]

    simulation.subscribe("person_detected", detect_person)
    simulation.schedule(Event(2, "person_detected"))

    trace = simulation.run()

    assert [(entry.at_s, entry.kind) for entry in trace] == [
        (2, "person_detected"),
        (2.3, "capture_triggered"),
    ]
    assert simulation.now_s == 10


def test_rejects_invalid_time_boundaries() -> None:
    simulation = BehavioralSimulation(duration_s=5)

    try:
        simulation.schedule(Event(6, "too_late"))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("event outside duration must be rejected")
