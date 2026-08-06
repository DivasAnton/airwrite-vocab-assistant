from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.segmentation_status import SegmentationStatus


class SegmentationValidator:
    def __init__(self, min_characters: int = 2, max_characters: int = 12) -> None:
        if not 1 <= min_characters <= max_characters:
            raise ValueError("Segmentation character limits are invalid")
        self.min_characters = min_characters
        self.max_characters = max_characters

    def validate(self, segments: tuple[CharacterSegment, ...]) -> SegmentationStatus:
        if len(segments) < self.min_characters:
            return SegmentationStatus.TOO_FEW_SEGMENTS
        if len(segments) > self.max_characters:
            return SegmentationStatus.TOO_MANY_SEGMENTS
        centers = [segment.bounding_box.center[0] for segment in segments]
        if centers != sorted(centers) or len(set(centers)) != len(centers):
            return SegmentationStatus.FAILED
        if any(
            first.bounding_box.right > second.bounding_box.x + second.bounding_box.width * 0.5
            for first, second in zip(segments, segments[1:], strict=False)
        ):
            return SegmentationStatus.AMBIGUOUS
        return SegmentationStatus.SUCCESS
