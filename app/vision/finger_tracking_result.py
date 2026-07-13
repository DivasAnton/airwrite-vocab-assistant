from dataclasses import dataclass


@dataclass(frozen=True)
class FingerTrackingResult:
    raw_point: tuple[int, int] | None
    smoothed_point: tuple[int, int] | None
    normalized_point: tuple[float, float] | None
    is_tracked: bool
    timestamp_ms: int
    hand_index: int | None = None
    landmark_index: int | None = None
