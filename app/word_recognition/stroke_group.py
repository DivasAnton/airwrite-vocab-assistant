from dataclasses import dataclass

from app.preprocessing.bounding_box import BoundingBox


@dataclass(frozen=True)
class StrokeGroup:
    group_id: str
    bounding_box: BoundingBox
    component_ids: tuple[str, ...]
    source_stroke_ids: tuple[str, ...]
    grouping_confidence: float

    def __post_init__(self) -> None:
        if not self.group_id.strip() or not self.component_ids:
            raise ValueError("Stroke group requires an ID and components")
        if not 0.0 <= self.grouping_confidence <= 1.0:
            raise ValueError("grouping_confidence must be between 0 and 1")
