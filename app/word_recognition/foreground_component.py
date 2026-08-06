from dataclasses import dataclass, field

from app.preprocessing.bounding_box import BoundingBox


@dataclass(frozen=True)
class ForegroundComponent:
    component_id: str
    bounding_box: BoundingBox
    area: int
    centroid_x: float
    centroid_y: float
    pixels: tuple[tuple[int, int], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must not be empty")
        if self.area <= 0:
            raise ValueError("component area must be positive")
