from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.foreground_component import ForegroundComponent
from app.word_recognition.segmentation_status import SegmentationStatus


@dataclass(frozen=True)
class WordSegmentationResult:
    status: SegmentationStatus
    segments: tuple[CharacterSegment, ...]
    word_bounding_box: BoundingBox | None
    foreground_mask: NDArray[np.uint8] | None
    components: tuple[ForegroundComponent, ...]
    message: str
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not self.message.strip() or self.elapsed_ms < 0.0:
            raise ValueError("Segmentation message and elapsed time are invalid")
        if self.status is SegmentationStatus.SUCCESS and not self.segments:
            raise ValueError("Successful segmentation requires segments")
