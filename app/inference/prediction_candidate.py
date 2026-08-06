from dataclasses import dataclass
from math import isfinite

from app.inference.character_case_mode import CharacterCaseMode


@dataclass(frozen=True)
class PredictionCandidate:
    label: str
    class_index: int
    confidence: float
    rank: int
    identity: str = ""
    rendered_character: str = ""
    case_mode: CharacterCaseMode | None = None

    def __post_init__(self) -> None:
        identity = self.identity or self.label.lower()
        rendered_character = self.rendered_character or self.label
        case_mode = self.case_mode or (
            CharacterCaseMode.UPPERCASE
            if rendered_character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            else CharacterCaseMode.LOWERCASE
        )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "rendered_character", rendered_character)
        object.__setattr__(self, "case_mode", case_mode)
        if self.label != self.rendered_character:
            raise ValueError("label must match rendered_character")
        if len(self.identity) != 1 or self.identity not in "abcdefghijklmnopqrstuvwxyz":
            raise ValueError("identity must be one canonical lowercase letter from a to z")
        expected_character = (
            self.identity if case_mode is CharacterCaseMode.LOWERCASE else self.identity.upper()
        )
        if self.rendered_character != expected_character:
            raise ValueError("rendered_character does not match identity and case_mode")
        if not 0 <= self.class_index < 26:
            raise ValueError("class_index must be between 0 and 25")
        if self.class_index != ord(identity) - ord("a"):
            raise ValueError("class_index does not match identity")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0.0 and 1.0")
        if self.rank < 1:
            raise ValueError("rank must be greater than or equal to 1")
