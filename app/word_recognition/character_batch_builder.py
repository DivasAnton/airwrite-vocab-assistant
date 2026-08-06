from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.preprocessing_result import PreprocessingResult
from app.word_recognition.character_batch import CharacterBatch
from app.word_recognition.character_segment import CharacterSegment


class SegmentPreprocessor(Protocol):
    def process(self, image: NDArray[np.uint8]) -> PreprocessingResult: ...


class CharacterBatchBuilder:
    def __init__(self, preprocessor: SegmentPreprocessor, max_characters: int = 12) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")
        self.preprocessor = preprocessor
        self.max_characters = max_characters

    def build(self, segments: tuple[CharacterSegment, ...]) -> CharacterBatch:
        if not 1 <= len(segments) <= self.max_characters:
            raise ValueError(f"Segment count must be between 1 and {self.max_characters}")
        ordered = tuple(sorted(segments, key=lambda segment: segment.position))
        if tuple(segment.position for segment in ordered) != tuple(range(len(ordered))):
            raise ValueError("Character segment positions must be continuous from zero")
        normalized = tuple(
            self.preprocessor.process(segment.grayscale_image).normalized_image
            for segment in ordered
        )
        tensor = np.stack(normalized, axis=0).astype(np.float32, copy=False)[..., np.newaxis]
        return CharacterBatch(
            segment_ids=tuple(segment.segment_id for segment in ordered),
            tensor=tensor,
        )
