from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_controller import DrawingController
from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.drawing.stroke_manager import StrokeManager
from app.vision.finger_tracking_result import FingerTrackingResult
from app.vision.gesture import Gesture


def make_controller() -> DrawingController:
    return DrawingController(
        state_machine=DrawingStateMachine(cooldown_ms=0),
        stroke_manager=StrokeManager(max_point_distance=120),
        canvas=AirCanvas(width=200, height=200),
    )


def make_finger(point: tuple[int, int] | None, timestamp_ms: int = 1) -> FingerTrackingResult:
    return FingerTrackingResult(
        raw_point=point,
        smoothed_point=point,
        normalized_point=None,
        is_tracked=point is not None,
        timestamp_ms=timestamp_ms,
        hand_index=0 if point is not None else None,
        landmark_index=8,
    )


def test_ready_does_not_draw() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.READY, timestamp_ms=1)

    result = controller.update(Gesture.OPEN_PALM, make_finger((100, 100), 2))

    assert result.state == DrawingState.READY
    assert controller.canvas.is_empty() is True


def test_writing_draws_after_two_points() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.WRITING, timestamp_ms=1)

    first = controller.update(Gesture.INDEX_ONLY, make_finger((100, 100), 2))
    second = controller.update(Gesture.INDEX_ONLY, make_finger((110, 105), 3))

    assert first.did_draw is False
    assert second.did_draw is True
    assert controller.canvas.is_empty() is False


def test_writing_to_paused_resets_stroke() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.WRITING, timestamp_ms=1)
    controller.update(Gesture.INDEX_ONLY, make_finger((100, 100), 2))

    result = controller.update(Gesture.OPEN_PALM, make_finger((110, 110), 3))

    assert result.state == DrawingState.PAUSED
    assert controller.stroke_manager.previous_point is None


def test_paused_does_not_draw() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.PAUSED, timestamp_ms=1)

    controller.update(Gesture.OPEN_PALM, make_finger((100, 100), 2))

    assert controller.canvas.is_empty() is True


def test_resume_from_pause_does_not_connect_old_point() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.WRITING, timestamp_ms=1)
    controller.update(Gesture.INDEX_ONLY, make_finger((100, 100), 2))
    controller.update(Gesture.OPEN_PALM, make_finger((110, 110), 3))

    result = controller.update(Gesture.INDEX_ONLY, make_finger((160, 160), 4))

    assert result.state == DrawingState.WRITING
    assert result.did_draw is False


def test_done_does_not_draw() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.DONE, timestamp_ms=1)

    controller.update(Gesture.FIST, make_finger((100, 100), 2))

    assert controller.canvas.is_empty() is True


def test_clear_resets_canvas_stroke_and_state() -> None:
    controller = make_controller()
    controller.set_state(DrawingState.WRITING, timestamp_ms=1)
    controller.update(Gesture.INDEX_ONLY, make_finger((100, 100), 2))
    controller.update(Gesture.INDEX_ONLY, make_finger((110, 110), 3))

    result = controller.clear(timestamp_ms=4)

    assert result.did_clear is True
    assert result.state == DrawingState.READY
    assert controller.canvas.is_empty() is True
    assert controller.stroke_manager.previous_point is None
