from dataclasses import dataclass


@dataclass(frozen=True)
class HandLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DetectedHand:
    normalized_landmarks: list[HandLandmark]
    world_landmarks: list[HandLandmark]
    handedness: str | None = None
    handedness_score: float | None = None


@dataclass(frozen=True)
class HandDetectionResult:
    hands: list[DetectedHand]
    timestamp_ms: int

    @property
    def has_hands(self) -> bool:
        return len(self.hands) > 0
