from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.word_recognition.recorded_stroke import RecordedStroke
from app.word_recognition.word_writing_region import WordWritingRegion


@dataclass(frozen=True)
class WordInputSnapshot:
    snapshot_id: str
    canvas_image: NDArray[np.uint8]
    strokes: tuple[RecordedStroke, ...]
    writing_region: WordWritingRegion
    completed_at_ms: int

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        if not isinstance(self.canvas_image, np.ndarray) or self.canvas_image.dtype != np.uint8:
            raise ValueError("canvas_image must be a uint8 NumPy array")
        if self.canvas_image.ndim not in {2, 3}:
            raise ValueError("canvas_image must be grayscale or color")
        height, width = self.canvas_image.shape[:2]
        if self.writing_region.right > width or self.writing_region.bottom > height:
            raise ValueError("writing region must fit inside the canvas image")
        if self.completed_at_ms < 0:
            raise ValueError("completed_at_ms must be non-negative")
        frozen_image = self.canvas_image.copy()
        frozen_image.flags.writeable = False
        object.__setattr__(self, "canvas_image", frozen_image)
