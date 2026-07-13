import numpy as np
import pytest

from app.vision.finger_tracking_renderer import FingerTrackingRenderer
from app.vision.finger_tracking_result import FingerTrackingResult


def make_tracked_result(point: tuple[int, int] = (20, 30)) -> FingerTrackingResult:
    return FingerTrackingResult(
        raw_point=point,
        smoothed_point=point,
        normalized_point=(0.2, 0.3),
        is_tracked=True,
        timestamp_ms=1,
        hand_index=0,
        landmark_index=8,
    )


def make_untracked_result() -> FingerTrackingResult:
    return FingerTrackingResult(
        raw_point=None,
        smoothed_point=None,
        normalized_point=None,
        is_tracked=False,
        timestamp_ms=1,
    )


def test_untracked_result_does_not_crash() -> None:
    renderer = FingerTrackingRenderer()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_untracked_result())

    assert output.shape == frame.shape


def test_tracked_point_keeps_shape() -> None:
    renderer = FingerTrackingRenderer()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_tracked_result())

    assert output.shape == frame.shape


def test_renderer_does_not_mutate_input_frame() -> None:
    renderer = FingerTrackingRenderer()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    original = frame.copy()

    renderer.draw(frame, make_tracked_result())

    assert np.array_equal(frame, original)


def test_drawing_options_can_be_disabled() -> None:
    renderer = FingerTrackingRenderer(draw_raw_point=False, draw_smoothed_point=False)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_tracked_result())

    assert np.array_equal(output, frame)


def test_boundary_point_does_not_crash() -> None:
    renderer = FingerTrackingRenderer()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_tracked_result((99, 79)))

    assert output.shape == frame.shape


def test_invalid_frame_raises_clear_error() -> None:
    renderer = FingerTrackingRenderer()

    with pytest.raises(ValueError, match="frame must be"):
        renderer.draw(np.array([]), make_tracked_result())
