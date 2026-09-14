from collections.abc import Callable
from uuid import uuid4

from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.recorded_stroke import RecordedStroke
from app.word_recognition.stroke_point import StrokePoint


class StrokeRecorder:
    def __init__(
        self,
        max_strokes: int = 64,
        max_points_per_stroke: int = 1000,
        min_points_per_stroke: int = 2,
        stroke_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_strokes <= 0 or max_points_per_stroke <= 0:
            raise ValueError("Stroke and point limits must be positive")
        if min_points_per_stroke <= 0:
            raise ValueError("min_points_per_stroke must be positive")
        self.max_strokes = max_strokes
        self.max_points_per_stroke = max_points_per_stroke
        self.min_points_per_stroke = min_points_per_stroke
        self._stroke_id_factory = stroke_id_factory or (lambda: uuid4().hex)
        self._current_points: list[StrokePoint] | None = None
        self._completed_strokes: list[RecordedStroke] = []

    @property
    def is_recording(self) -> bool:
        return self._current_points is not None

    @property
    def completed_strokes(self) -> tuple[RecordedStroke, ...]:
        return tuple(self._completed_strokes)

    def truncate(self, count: int) -> None:
        """Discard completed strokes after ``count`` for pause-level undo."""
        if count < 0 or count > len(self._completed_strokes):
            raise ValueError("Stroke truncate count is out of range")
        self._completed_strokes = self._completed_strokes[:count]

    def start_stroke(self, point: StrokePoint) -> None:
        if self.is_recording:
            raise RuntimeError("Cannot start a stroke while another stroke is active")
        if len(self._completed_strokes) >= self.max_strokes:
            raise ValueError("Maximum whole-word stroke count reached")
        self._current_points = [point]

    def append_point(self, point: StrokePoint) -> None:
        if self._current_points is None:
            raise RuntimeError("Cannot append a point before starting a stroke")
        if len(self._current_points) >= self.max_points_per_stroke:
            raise ValueError("Maximum points per stroke reached")
        if point.timestamp_ms < self._current_points[-1].timestamp_ms:
            raise ValueError("Stroke point timestamps must be non-decreasing")
        if point.coordinates != self._current_points[-1].coordinates:
            self._current_points.append(point)

    def end_stroke(self) -> RecordedStroke | None:
        if self._current_points is None:
            return None
        points = tuple(self._current_points)
        self._current_points = None
        if len(points) < self.min_points_per_stroke:
            return None
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        stroke = RecordedStroke(
            stroke_id=self._stroke_id_factory(),
            points=points,
            started_at_ms=points[0].timestamp_ms,
            ended_at_ms=points[-1].timestamp_ms,
            bounding_box=BoundingBox(
                x=min(xs),
                y=min(ys),
                width=max(xs) - min(xs) + 1,
                height=max(ys) - min(ys) + 1,
            ),
        )
        self._completed_strokes.append(stroke)
        return stroke

    def reset(self) -> None:
        self._current_points = None
        self._completed_strokes.clear()

    def snapshot(self) -> tuple[RecordedStroke, ...]:
        return tuple(self._completed_strokes)
