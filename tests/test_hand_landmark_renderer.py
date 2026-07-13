import numpy as np
import pytest

from app.vision.hand_detection_result import (
    DetectedHand,
    HandDetectionResult,
    HandLandmark,
)
from app.vision.hand_landmark_renderer import HandLandmarkRenderer


def make_result() -> HandDetectionResult:
    landmarks = [HandLandmark(x=0.1 + index * 0.01, y=0.2, z=0.0) for index in range(21)]
    hand = DetectedHand(
        normalized_landmarks=landmarks,
        world_landmarks=[],
        handedness="Right",
        handedness_score=0.95,
    )
    return HandDetectionResult(hands=[hand], timestamp_ms=1)


def test_no_hands_returns_same_shape() -> None:
    renderer = HandLandmarkRenderer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = HandDetectionResult(hands=[], timestamp_ms=1)

    output = renderer.draw(frame, result)

    assert output.shape == frame.shape


def test_draw_one_hand_does_not_crash_or_resize() -> None:
    renderer = HandLandmarkRenderer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_result())

    assert output.shape == frame.shape


def test_renderer_does_not_mutate_input_frame() -> None:
    renderer = HandLandmarkRenderer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    original = frame.copy()

    renderer.draw(frame, make_result())

    assert np.array_equal(frame, original)


def test_invalid_frame_raises_clear_error() -> None:
    renderer = HandLandmarkRenderer()

    with pytest.raises(ValueError, match="frame must be"):
        renderer.draw(np.array([]), make_result())


def test_drawing_options_can_be_disabled() -> None:
    renderer = HandLandmarkRenderer(
        draw_landmarks=False,
        draw_connections=False,
        draw_handedness=False,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    output = renderer.draw(frame, make_result())

    assert output.shape == frame.shape
