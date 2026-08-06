from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox


@dataclass(frozen=True)
class CharacterSegment:
    segment_id: str
    position: int
    bounding_box: BoundingBox
    source_stroke_ids: tuple[str, ...]
    grayscale_image: NDArray[np.uint8]
    segmentation_confidence: float

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or self.position < 0:
            raise ValueError("Character segment ID and position are invalid")
        if self.grayscale_image.dtype != np.uint8 or self.grayscale_image.ndim != 2:
            raise ValueError("Segment image must be a 2D uint8 array")
        if self.grayscale_image.size == 0 or not np.any(self.grayscale_image > 0):
            raise ValueError("Segment image must contain foreground")
        if not 0.0 <= self.segmentation_confidence <= 1.0:
            raise ValueError("segmentation_confidence must be between 0 and 1")
        frozen_image = self.grayscale_image.copy()
        frozen_image.flags.writeable = False
        object.__setattr__(self, "grayscale_image", frozen_image)
