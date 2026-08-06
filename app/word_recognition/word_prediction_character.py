from dataclasses import dataclass
from math import isfinite

from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_status import PredictionStatus
from app.preprocessing.bounding_box import BoundingBox
from app.word_builder.character_source import CharacterSource


@dataclass(frozen=True)
class WordPredictionCharacter:
    position: int
    identity: str
    rendered_character: str
    case_mode: CharacterCaseMode
    confidence: float
    candidates: tuple[PredictionCandidate, ...]
    status: PredictionStatus
    source: CharacterSource
    segment_id: str | None
    bounding_box: BoundingBox | None

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("Word character position must be non-negative")
        if len(self.identity) != 1 or self.identity not in "abcdefghijklmnopqrstuvwxyz":
            raise ValueError("Word character identity must be one lowercase letter")
        expected = (
            self.identity
            if self.case_mode is CharacterCaseMode.LOWERCASE
            else self.identity.upper()
        )
        if self.rendered_character != expected:
            raise ValueError("Rendered word character does not match identity and case")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Word character confidence must be between 0 and 1")
        if not self.candidates or self.candidates[0].identity != self.identity:
            raise ValueError("Word character candidates must start with the selected identity")
