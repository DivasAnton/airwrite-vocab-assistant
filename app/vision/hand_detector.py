import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.vision.hand_detection_result import (
    DetectedHand,
    HandDetectionResult,
    HandLandmark,
)

LandmarkerFactory = Callable[[], Any]
ImageFactory = Callable[[NDArray[np.uint8]], Any]


class HandDetector:
    """Detect hands from OpenCV BGR frames using MediaPipe Hand Landmarker."""

    def __init__(
        self,
        model_path: Path,
        num_hands: int,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
        logger: logging.Logger | None = None,
        landmarker_factory: LandmarkerFactory | None = None,
        image_factory: ImageFactory | None = None,
    ) -> None:
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logger or logging.getLogger(__name__)
        self.image_factory = image_factory or self._create_mediapipe_image

        self._validate_config()
        self.landmarker = (
            landmarker_factory()
            if landmarker_factory is not None
            else self._create_mediapipe_landmarker()
        )

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def detect(self, frame_bgr: NDArray[np.uint8], timestamp_ms: int) -> HandDetectionResult:
        rgb_frame = cast(NDArray[np.uint8], cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self.image_factory(rgb_frame)
        mediapipe_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._to_detection_result(mediapipe_result, timestamp_ms)

    def close(self) -> None:
        close = getattr(self.landmarker, "close", None)
        if callable(close):
            close()

    def _validate_config(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {self.model_path}. "
                "Download it and place it at models/hand_landmarker.task."
            )
        if self.num_hands <= 0:
            raise ValueError(f"num_hands must be greater than 0, got {self.num_hands}")
        self._validate_confidence("min_detection_confidence", self.min_detection_confidence)
        self._validate_confidence("min_presence_confidence", self.min_presence_confidence)
        self._validate_confidence("min_tracking_confidence", self.min_tracking_confidence)

    @staticmethod
    def _validate_confidence(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

    def _create_mediapipe_landmarker(self) -> Any:
        import mediapipe as mp

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _create_mediapipe_image(rgb_frame: NDArray[np.uint8]) -> Any:
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _to_detection_result(self, result: Any, timestamp_ms: int) -> HandDetectionResult:
        hand_landmarks = list(getattr(result, "hand_landmarks", []) or [])
        world_landmarks = list(getattr(result, "hand_world_landmarks", []) or [])
        handedness_items = list(getattr(result, "handedness", []) or [])
        hands: list[DetectedHand] = []

        for index, landmarks in enumerate(hand_landmarks):
            normalized = self._convert_landmarks(landmarks)
            if len(normalized) != 21:
                self.logger.warning(
                    "Skipping hand with %s landmarks instead of 21", len(normalized)
                )
                continue

            world = (
                self._convert_landmarks(world_landmarks[index])
                if index < len(world_landmarks)
                else []
            )
            handedness, score = self._extract_handedness(handedness_items, index)
            hands.append(
                DetectedHand(
                    normalized_landmarks=normalized,
                    world_landmarks=world,
                    handedness=handedness,
                    handedness_score=score,
                )
            )

        return HandDetectionResult(hands=hands, timestamp_ms=timestamp_ms)

    @staticmethod
    def _convert_landmarks(landmarks: Any) -> list[HandLandmark]:
        return [
            HandLandmark(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
            )
            for landmark in landmarks
        ]

    @staticmethod
    def _extract_handedness(
        handedness_items: list[Any], hand_index: int
    ) -> tuple[str | None, float | None]:
        if hand_index >= len(handedness_items) or not handedness_items[hand_index]:
            return None, None

        category = handedness_items[hand_index][0]
        label = cast(str | None, getattr(category, "category_name", None))
        score_value = getattr(category, "score", None)
        score = float(score_value) if score_value is not None else None
        return label, score
