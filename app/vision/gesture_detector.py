from app.vision.gesture import Gesture
from app.vision.hand_detection_result import HandDetectionResult, HandLandmark

FINGER_PAIRS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


class GestureDetector:
    def __init__(self, finger_extension_margin: float = 0.02) -> None:
        if not 0.0 <= finger_extension_margin < 1.0:
            raise ValueError(
                "finger_extension_margin must be greater than or equal to 0.0 "
                f"and less than 1.0, got {finger_extension_margin}"
            )

        self.finger_extension_margin = finger_extension_margin

    def detect(self, result: HandDetectionResult) -> Gesture:
        if not result.hands:
            return Gesture.NO_HAND

        landmarks = result.hands[0].normalized_landmarks
        if len(landmarks) < 21:
            return Gesture.UNKNOWN

        fingers = {
            name: self._is_finger_extended(landmarks, tip, pip)
            for name, (tip, pip) in FINGER_PAIRS.items()
        }

        if (
            fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            return Gesture.INDEX_ONLY
        if fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            return Gesture.OPEN_PALM
        if (
            not fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            return Gesture.FIST

        return Gesture.UNKNOWN

    def _is_finger_extended(
        self,
        landmarks: list[HandLandmark],
        tip_index: int,
        pip_index: int,
    ) -> bool:
        return landmarks[tip_index].y < landmarks[pip_index].y - self.finger_extension_margin
