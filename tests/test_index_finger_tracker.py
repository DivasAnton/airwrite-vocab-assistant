import pytest

from app.vision.hand_detection_result import (
    DetectedHand,
    HandDetectionResult,
    HandLandmark,
)
from app.vision.index_finger_tracker import IndexFingerTracker


def make_result(landmark: HandLandmark, timestamp_ms: int = 1) -> HandDetectionResult:
    landmarks = [HandLandmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    landmarks[8] = landmark
    hand = DetectedHand(normalized_landmarks=landmarks, world_landmarks=[])
    return HandDetectionResult(hands=[hand], timestamp_ms=timestamp_ms)


def test_center_normalized_point_converts_to_pixel() -> None:
    tracker = IndexFingerTracker()
    result = make_result(HandLandmark(x=0.5, y=0.5, z=0.0))

    tracking = tracker.track(result, frame_width=640, frame_height=480)

    assert tracking.raw_point == (320, 240)
    assert tracking.smoothed_point == (320, 240)


def test_top_left_converts_to_origin() -> None:
    tracker = IndexFingerTracker()
    result = make_result(HandLandmark(x=0.0, y=0.0, z=0.0))

    tracking = tracker.track(result, frame_width=640, frame_height=480)

    assert tracking.raw_point == (0, 0)


def test_bottom_right_clamps_to_last_pixel() -> None:
    tracker = IndexFingerTracker()
    result = make_result(HandLandmark(x=1.0, y=1.0, z=0.0))

    tracking = tracker.track(result, frame_width=640, frame_height=480)

    assert tracking.raw_point == (639, 479)


def test_out_of_range_normalized_point_is_clamped() -> None:
    tracker = IndexFingerTracker()
    result = make_result(HandLandmark(x=-0.1, y=1.2, z=0.0))

    tracking = tracker.track(result, frame_width=640, frame_height=480)

    assert tracking.raw_point == (0, 479)


def test_tracker_uses_landmark_index_8() -> None:
    tracker = IndexFingerTracker()
    result = make_result(HandLandmark(x=0.25, y=0.75, z=0.0))

    tracking = tracker.track(result, frame_width=400, frame_height=200)

    assert tracking.raw_point == (100, 150)
    assert tracking.landmark_index == 8


def test_exponential_smoothing() -> None:
    tracker = IndexFingerTracker(smoothing_alpha=0.5)

    first = tracker.track(make_result(HandLandmark(x=0.25, y=0.25, z=0.0), 1), 400, 400)
    second = tracker.track(make_result(HandLandmark(x=0.5, y=0.5, z=0.0), 2), 400, 400)
    third = tracker.track(make_result(HandLandmark(x=0.5, y=0.5, z=0.0), 3), 400, 400)

    assert first.smoothed_point == (100, 100)
    assert second.smoothed_point == (150, 150)
    assert third.smoothed_point == (175, 175)


def test_reset_when_hand_is_lost() -> None:
    tracker = IndexFingerTracker(smoothing_alpha=0.5)
    tracker.track(make_result(HandLandmark(x=0.25, y=0.25, z=0.0), 1), 400, 400)

    lost = tracker.track(HandDetectionResult(hands=[], timestamp_ms=2), 400, 400)
    reacquired = tracker.track(make_result(HandLandmark(x=0.75, y=0.75, z=0.0), 3), 400, 400)

    assert lost.is_tracked is False
    assert lost.raw_point is None
    assert reacquired.smoothed_point == (300, 300)


@pytest.mark.parametrize("width,height", [(0, 480), (640, 0), (-1, 480), (640, -1)])
def test_invalid_frame_dimensions_raise(width: int, height: int) -> None:
    tracker = IndexFingerTracker()

    with pytest.raises(ValueError, match="frame_width and frame_height"):
        tracker.track(make_result(HandLandmark(x=0.5, y=0.5, z=0.0)), width, height)


def test_missing_landmark_resets_tracking() -> None:
    tracker = IndexFingerTracker()
    hand = DetectedHand(normalized_landmarks=[], world_landmarks=[])

    tracking = tracker.track(HandDetectionResult(hands=[hand], timestamp_ms=1), 640, 480)

    assert tracking.is_tracked is False
    assert tracking.raw_point is None
