from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.vision.hand_detector import HandDetector


def make_detector(model_path: Path, landmarker: Mock) -> HandDetector:
    return HandDetector(
        model_path=model_path,
        num_hands=1,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        landmarker_factory=lambda: landmarker,
        image_factory=lambda frame: frame,
    )


def make_landmarks(count: int = 21) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.1, y=0.2, z=0.0) for _ in range(count)]


def test_missing_model_path_raises_clear_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.task"

    with pytest.raises(FileNotFoundError, match="Hand landmarker model not found"):
        HandDetector(
            model_path=missing_model,
            num_hands=1,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            landmarker_factory=Mock(),
        )


def test_invalid_confidence_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"model")

    with pytest.raises(ValueError, match="min_detection_confidence"):
        HandDetector(
            model_path=model_path,
            num_hands=1,
            min_detection_confidence=1.5,
            min_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            landmarker_factory=Mock(),
        )


def test_no_hand_result_returns_empty_hands(tmp_path: Path) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"model")
    landmarker = Mock()
    landmarker.detect_for_video.return_value = SimpleNamespace(
        hand_landmarks=[],
        hand_world_landmarks=[],
        handedness=[],
    )
    detector = make_detector(model_path, landmarker)

    result = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), timestamp_ms=1)

    assert result.has_hands is False
    assert result.hands == []
    landmarker.detect_for_video.assert_called_once()


def test_one_hand_maps_21_landmarks_and_handedness(tmp_path: Path) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"model")
    landmarker = Mock()
    landmarker.detect_for_video.return_value = SimpleNamespace(
        hand_landmarks=[make_landmarks()],
        hand_world_landmarks=[make_landmarks()],
        handedness=[[SimpleNamespace(category_name="Right", score=0.95)]],
    )
    detector = make_detector(model_path, landmarker)

    result = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), timestamp_ms=2)

    assert result.has_hands is True
    assert len(result.hands) == 1
    assert len(result.hands[0].normalized_landmarks) == 21
    assert result.hands[0].handedness == "Right"
    assert result.hands[0].handedness_score == 0.95


def test_incomplete_landmarks_are_skipped(tmp_path: Path) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"model")
    landmarker = Mock()
    landmarker.detect_for_video.return_value = SimpleNamespace(
        hand_landmarks=[make_landmarks(20)],
        hand_world_landmarks=[],
        handedness=[],
    )
    detector = make_detector(model_path, landmarker)

    result = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), timestamp_ms=3)

    assert result.has_hands is False
    assert result.hands == []


def test_close_delegates_to_landmarker(tmp_path: Path) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"model")
    landmarker = Mock()
    detector = make_detector(model_path, landmarker)

    detector.close()

    landmarker.close.assert_called_once()
