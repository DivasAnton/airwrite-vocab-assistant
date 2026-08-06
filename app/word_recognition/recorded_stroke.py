from dataclasses import dataclass

from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.stroke_point import StrokePoint


@dataclass(frozen=True)
class RecordedStroke:
    stroke_id: str
    points: tuple[StrokePoint, ...]
    started_at_ms: int
    ended_at_ms: int
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if not self.stroke_id.strip():
            raise ValueError("stroke_id must not be empty")
        if not self.points:
            raise ValueError("Recorded stroke must contain points")
        if self.started_at_ms < 0 or self.ended_at_ms < self.started_at_ms:
            raise ValueError("Recorded stroke timestamps are invalid")
