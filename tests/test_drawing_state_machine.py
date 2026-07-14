from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.vision.gesture import Gesture


def test_idle_open_palm_goes_ready() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)

    assert machine.update(Gesture.OPEN_PALM, timestamp_ms=1) == DrawingState.READY


def test_ready_index_only_goes_writing() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.update(Gesture.OPEN_PALM, timestamp_ms=1)

    assert machine.update(Gesture.INDEX_ONLY, timestamp_ms=2) == DrawingState.WRITING


def test_writing_open_palm_goes_paused() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.set_state(DrawingState.WRITING, timestamp_ms=1)

    assert machine.update(Gesture.OPEN_PALM, timestamp_ms=2) == DrawingState.PAUSED


def test_paused_index_only_goes_writing() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.set_state(DrawingState.PAUSED, timestamp_ms=1)

    assert machine.update(Gesture.INDEX_ONLY, timestamp_ms=2) == DrawingState.WRITING


def test_writing_fist_goes_done() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.set_state(DrawingState.WRITING, timestamp_ms=1)

    assert machine.update(Gesture.FIST, timestamp_ms=2) == DrawingState.DONE


def test_done_open_palm_goes_ready() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.set_state(DrawingState.DONE, timestamp_ms=1)

    assert machine.update(Gesture.OPEN_PALM, timestamp_ms=2) == DrawingState.READY


def test_idle_fist_does_not_jump_to_done() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)

    assert machine.update(Gesture.FIST, timestamp_ms=1) != DrawingState.DONE


def test_writing_index_only_stays_writing() -> None:
    machine = DrawingStateMachine(cooldown_ms=0)
    machine.set_state(DrawingState.WRITING, timestamp_ms=1)

    assert machine.update(Gesture.INDEX_ONLY, timestamp_ms=2) == DrawingState.WRITING


def test_cooldown_blocks_repeated_transition() -> None:
    machine = DrawingStateMachine(cooldown_ms=500)
    machine.update(Gesture.OPEN_PALM, timestamp_ms=1000)

    assert machine.update(Gesture.INDEX_ONLY, timestamp_ms=1200) == DrawingState.READY
    assert machine.update(Gesture.INDEX_ONLY, timestamp_ms=1600) == DrawingState.WRITING
