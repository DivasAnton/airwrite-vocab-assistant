from dataclasses import dataclass


@dataclass(frozen=True)
class StrokePoint:
    x: int
    y: int
    timestamp_ms: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Stroke point coordinates must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("Stroke point timestamp must be non-negative")

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y
