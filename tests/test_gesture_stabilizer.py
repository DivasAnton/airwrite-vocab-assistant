from app.vision.gesture import Gesture
from app.vision.gesture_stabilizer import GestureStabilizer


def test_not_enough_stable_frames_keeps_previous_stable_gesture() -> None:
    stabilizer = GestureStabilizer(stable_frames=5)

    for _ in range(4):
        stable = stabilizer.update(Gesture.INDEX_ONLY)

    assert stable == Gesture.NO_HAND


def test_enough_stable_frames_accepts_gesture() -> None:
    stabilizer = GestureStabilizer(stable_frames=5)

    for _ in range(5):
        stable = stabilizer.update(Gesture.INDEX_ONLY)

    assert stable == Gesture.INDEX_ONLY


def test_candidate_change_resets_count() -> None:
    stabilizer = GestureStabilizer(stable_frames=5)
    for _ in range(3):
        stabilizer.update(Gesture.INDEX_ONLY)

    stable = stabilizer.update(Gesture.OPEN_PALM)

    assert stable == Gesture.NO_HAND
    assert stabilizer.candidate_gesture == Gesture.OPEN_PALM
    assert stabilizer.candidate_frame_count == 1


def test_jitter_does_not_change_stable_gesture() -> None:
    stabilizer = GestureStabilizer(stable_frames=3)

    for gesture in [Gesture.INDEX_ONLY, Gesture.UNKNOWN, Gesture.INDEX_ONLY, Gesture.UNKNOWN]:
        stable = stabilizer.update(gesture)

    assert stable == Gesture.NO_HAND


def test_no_hand_requires_lost_hand_threshold() -> None:
    stabilizer = GestureStabilizer(stable_frames=2, lost_hand_frames=3)
    stabilizer.update(Gesture.INDEX_ONLY)
    stabilizer.update(Gesture.INDEX_ONLY)

    assert stabilizer.update(Gesture.NO_HAND) == Gesture.INDEX_ONLY
    assert stabilizer.update(Gesture.NO_HAND) == Gesture.INDEX_ONLY
    assert stabilizer.update(Gesture.NO_HAND) == Gesture.NO_HAND
