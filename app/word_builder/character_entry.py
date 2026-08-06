from dataclasses import dataclass
from math import isfinite

from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_source import CharacterSource
from app.word_builder.exceptions import InvalidCharacterError


@dataclass(frozen=True)
class CharacterEntry:
    identity: str
    rendered_character: str
    case_mode: CharacterCaseMode
    shift_was_active: bool
    confidence: float | None
    source: CharacterSource
    prediction_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.case_mode, CharacterCaseMode):
            raise ValueError("case_mode must be a CharacterCaseMode")
        if not isinstance(self.shift_was_active, bool):
            raise ValueError("shift_was_active must be a boolean")
        if len(self.identity) != 1 or self.identity not in "abcdefghijklmnopqrstuvwxyz":
            raise InvalidCharacterError(
                "identity must be one canonical lowercase letter from a to z"
            )
        expected_character = (
            self.identity
            if self.case_mode is CharacterCaseMode.LOWERCASE
            else self.identity.upper()
        )
        if self.rendered_character != expected_character:
            raise InvalidCharacterError("rendered_character must match identity and case_mode")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be None or finite and between 0.0 and 1.0")
        if self.prediction_id is not None and not self.prediction_id.strip():
            raise ValueError("prediction_id must not be empty when provided")

    @property
    def character(self) -> str:
        """Compatibility alias for the exact case-preserved character."""
        return self.rendered_character
