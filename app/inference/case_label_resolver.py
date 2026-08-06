from dataclasses import dataclass

from app.inference.character_case_mode import CharacterCaseMode
from app.inference.identity_candidate import IdentityCandidate
from app.inference.prediction_candidate import PredictionCandidate
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


@dataclass(frozen=True)
class ResolvedCharacter:
    identity: str
    rendered_character: str
    class_index: int
    case_mode: CharacterCaseMode


class CaseLabelResolver:
    def __init__(
        self,
        identity_labels: tuple[str, ...],
        lowercase_display_labels: tuple[str, ...],
        uppercase_display_labels: tuple[str, ...],
    ) -> None:
        self.validate_mappings(
            identity_labels,
            lowercase_display_labels,
            uppercase_display_labels,
        )
        self.identity_labels = identity_labels
        self.lowercase_display_labels = lowercase_display_labels
        self.uppercase_display_labels = uppercase_display_labels

    @staticmethod
    def validate_mappings(
        identity_labels: tuple[str, ...],
        lowercase_display_labels: tuple[str, ...],
        uppercase_display_labels: tuple[str, ...],
    ) -> None:
        mappings = {
            "identity": identity_labels,
            "lowercase display": lowercase_display_labels,
            "uppercase display": uppercase_display_labels,
        }
        for name, labels in mappings.items():
            if len(labels) != 26:
                raise ValueError(f"{name} labels must contain exactly 26 items")
            if len(set(labels)) != 26:
                raise ValueError(f"{name} labels must not contain duplicates")
            if any(len(label) != 1 for label in labels):
                raise ValueError(f"Every {name} label must contain exactly one character")
        if identity_labels != LETTER_IDENTITY_LABELS:
            raise ValueError("identity labels must contain canonical a-z order")
        if lowercase_display_labels != LETTER_IDENTITY_LABELS:
            raise ValueError("lowercase display labels must contain canonical a-z order")
        expected_uppercase = tuple(label.upper() for label in LETTER_IDENTITY_LABELS)
        if uppercase_display_labels != expected_uppercase:
            raise ValueError("uppercase display labels must contain canonical A-Z order")
        for index, identity in enumerate(identity_labels):
            if lowercase_display_labels[index] != identity:
                raise ValueError("lowercase display mapping does not match identity mapping")
            if uppercase_display_labels[index].lower() != identity:
                raise ValueError("uppercase display mapping does not match identity mapping")

    def resolve(self, class_index: int, mode: CharacterCaseMode) -> ResolvedCharacter:
        if not 0 <= class_index < len(self.identity_labels):
            raise IndexError(f"class_index must be between 0 and 25, got {class_index}")
        display_labels = (
            self.lowercase_display_labels
            if mode is CharacterCaseMode.LOWERCASE
            else self.uppercase_display_labels
        )
        return ResolvedCharacter(
            identity=self.identity_labels[class_index],
            rendered_character=display_labels[class_index],
            class_index=class_index,
            case_mode=mode,
        )

    def resolve_candidate(
        self,
        candidate: IdentityCandidate,
        mode: CharacterCaseMode,
    ) -> PredictionCandidate:
        resolved = self.resolve(candidate.class_index, mode)
        if resolved.identity != candidate.identity:
            raise ValueError("Identity candidate does not match the configured label mapping")
        return PredictionCandidate(
            label=resolved.rendered_character,
            identity=resolved.identity,
            rendered_character=resolved.rendered_character,
            class_index=resolved.class_index,
            confidence=candidate.confidence,
            rank=candidate.rank,
            case_mode=resolved.case_mode,
        )
