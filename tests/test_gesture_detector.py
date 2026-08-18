import pytest

from app.vision.gesture import Gesture
from app.vision.gesture_detector import GestureDetector
from app.vision.hand_detection_result import (
    DetectedHand,
    HandDetectionResult,
    HandLandmark,
)


def make_landmarks(
    index_open: bool,
    middle_open: bool,
    ring_open: bool,
    pinky_open: bool,
) -> list[HandLandmark]:
    landmarks = [HandLandmark(x=0.0, y=0.5, z=0.0) for _ in range(21)]
    for tip, pip, is_open in [
        (8, 6, index_open),
        (12, 10, middle_open),
        (16, 14, ring_open),
        (20, 18, pinky_open),
    ]:
        landmarks[pip] = HandLandmark(x=0.0, y=0.5, z=0.0)
        landmarks[tip] = HandLandmark(x=0.0, y=0.3 if is_open else 0.7, z=0.0)
    return landmarks


def make_result(landmarks: list[HandLandmark]) -> HandDetectionResult:
    hand = DetectedHand(normalized_landmarks=landmarks, world_landmarks=[])
    return HandDetectionResult(hands=[hand], timestamp_ms=1)


def test_no_hand_returns_no_hand() -> None:
    detector = GestureDetector()

    assert detector.detect(HandDetectionResult(hands=[], timestamp_ms=1)) == Gesture.NO_HAND


def test_index_only() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result(make_landmarks(True, False, False, False)))

    assert gesture == Gesture.INDEX_ONLY


def test_open_palm() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result(make_landmarks(True, True, True, True)))

    assert gesture == Gesture.OPEN_PALM


def test_fist() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result(make_landmarks(False, False, False, False)))

    assert gesture == Gesture.FIST


def test_two_fingers() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result(make_landmarks(True, True, False, False)))

    assert gesture == Gesture.TWO_FINGERS


def test_mixed_gesture_returns_unknown() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result(make_landmarks(True, False, False, True)))

    assert gesture == Gesture.UNKNOWN


def test_missing_landmarks_returns_unknown() -> None:
    detector = GestureDetector()

    gesture = detector.detect(make_result([]))

    assert gesture == Gesture.UNKNOWN


def test_margin_prevents_near_equal_tip_from_counting_as_open() -> None:
    detector = GestureDetector(finger_extension_margin=0.02)
    landmarks = make_landmarks(False, False, False, False)
    landmarks[6] = HandLandmark(x=0.0, y=0.50, z=0.0)
    landmarks[8] = HandLandmark(x=0.0, y=0.49, z=0.0)

    gesture = detector.detect(make_result(landmarks))

    assert gesture == Gesture.FIST


def test_invalid_margin_raises() -> None:
    with pytest.raises(ValueError, match="finger_extension_margin"):
        GestureDetector(finger_extension_margin=1.0)
