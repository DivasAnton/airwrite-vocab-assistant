from dataclasses import dataclass

from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.whole_word_prediction_status import WholeWordPredictionStatus
from app.word_recognition.word_prediction_character import WordPredictionCharacter


@dataclass(frozen=True)
class WholeWordPredictionResult:
    prediction_id: str
    status: WholeWordPredictionStatus
    characters: tuple[WordPredictionCharacter, ...]
    predicted_word: str
    segmentation_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    segmentation_status: SegmentationStatus
    segments: tuple[CharacterSegment, ...]
    message: str

    def __post_init__(self) -> None:
        if not self.prediction_id.strip() or not self.message.strip():
            raise ValueError("Whole-word prediction ID and message must not be empty")
        if any(value < 0.0 for value in self.timings):
            raise ValueError("Whole-word prediction timings must be non-negative")
        if self.predicted_word != "".join(
            character.rendered_character for character in self.characters
        ):
            raise ValueError("predicted_word must match ordered prediction characters")

    @property
    def timings(self) -> tuple[float, float, float, float]:
        return (
            self.segmentation_time_ms,
            self.preprocessing_time_ms,
            self.inference_time_ms,
            self.total_time_ms,
        )
