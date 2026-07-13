from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class LineSegment:
    start: tuple[int, int]
    end: tuple[int, int]


class StrokeManager:
    def __init__(self, max_point_distance: float = 120.0) -> None:
        if max_point_distance <= 0:
            raise ValueError(f"max_point_distance must be greater than 0, got {max_point_distance}")

        self.max_point_distance = max_point_distance
        self.previous_point: tuple[int, int] | None = None

    def update(self, current_point: tuple[int, int] | None) -> LineSegment | None:
        if current_point is None:
            self.reset()
            return None

        if self.previous_point is None:
            self.previous_point = current_point
            return None

        if self._distance(self.previous_point, current_point) > self.max_point_distance:
            self.previous_point = current_point
            return None

        segment = LineSegment(start=self.previous_point, end=current_point)
        self.previous_point = current_point
        return segment

    def reset(self) -> None:
        self.previous_point = None

    @staticmethod
    def _distance(start: tuple[int, int], end: tuple[int, int]) -> float:
        return hypot(end[0] - start[0], end[1] - start[1])
