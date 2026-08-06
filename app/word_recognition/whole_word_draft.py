from dataclasses import dataclass, replace
from enum import StrEnum

from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_status import PredictionStatus
from app.word_builder.character_source import CharacterSource
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.whole_word_prediction_result import WholeWordPredictionResult
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_prediction_character import WordPredictionCharacter


class WholeWordDraftStatus(StrEnum):
    REVIEWING = "REVIEWING"
    ACCEPTED = "ACCEPTED"
    CANCELLED = "CANCELLED"


@dataclass
class WholeWordDraft:
    prediction_id: str
    characters: list[WordPredictionCharacter]
    segments: list[CharacterSegment]
    case_policy: WordCasePolicy
    segmentation_status: SegmentationStatus
    selected_position: int = 0
    status: WholeWordDraftStatus = WholeWordDraftStatus.REVIEWING

    @classmethod
    def from_prediction(
        cls,
        result: WholeWordPredictionResult,
        case_policy: WordCasePolicy,
    ) -> "WholeWordDraft":
        if not result.characters:
            raise ValueError("A whole-word draft requires predicted characters")
        return cls(
            prediction_id=result.prediction_id,
            characters=list(result.characters),
            segments=list(result.segments),
            case_policy=case_policy,
            segmentation_status=result.segmentation_status,
        )

    @property
    def current_word(self) -> str:
        return "".join(character.rendered_character for character in self.characters)

    @property
    def selected_character(self) -> WordPredictionCharacter:
        return self.characters[self.selected_position]

    @property
    def unresolved_positions(self) -> tuple[int, ...]:
        return tuple(
            item.position
            for item in self.characters
            if item.status is not PredictionStatus.ACCEPTED
        )

    @property
    def can_accept(self) -> bool:
        return (
            self.status is WholeWordDraftStatus.REVIEWING
            and self.segmentation_status is SegmentationStatus.SUCCESS
            and not self.unresolved_positions
        )

    def move(self, offset: int) -> None:
        if self.status is not WholeWordDraftStatus.REVIEWING:
            return
        self.selected_position = min(
            max(self.selected_position + offset, 0),
            len(self.characters) - 1,
        )

    def move_previous(self) -> None:
        self.move(-1)

    def move_next(self) -> None:
        self.move(1)

    def select_candidate(self, rank: int) -> None:
        selected = self.selected_character
        try:
            candidate = next(item for item in selected.candidates if item.rank == rank)
        except StopIteration as error:
            raise ValueError(f"Candidate rank {rank} is unavailable") from error
        candidate_mode = candidate.case_mode
        if candidate_mode is None:
            raise ValueError("Selected candidate has no case mode")
        self.characters[self.selected_position] = replace(
            selected,
            identity=candidate.identity,
            rendered_character=candidate.rendered_character,
            case_mode=candidate_mode,
            confidence=candidate.confidence,
            status=PredictionStatus.ACCEPTED,
            source=CharacterSource.USER_SELECTED,
        )

    def toggle_selected_case(self) -> None:
        selected = self.selected_character
        mode = (
            CharacterCaseMode.UPPERCASE
            if selected.case_mode is CharacterCaseMode.LOWERCASE
            else CharacterCaseMode.LOWERCASE
        )
        candidates = tuple(
            replace(
                candidate,
                label=(
                    candidate.identity
                    if mode is CharacterCaseMode.LOWERCASE
                    else candidate.identity.upper()
                ),
                rendered_character=(
                    candidate.identity
                    if mode is CharacterCaseMode.LOWERCASE
                    else candidate.identity.upper()
                ),
                case_mode=mode,
            )
            for candidate in selected.candidates
        )
        self.characters[self.selected_position] = replace(
            selected,
            rendered_character=(
                selected.identity
                if mode is CharacterCaseMode.LOWERCASE
                else selected.identity.upper()
            ),
            case_mode=mode,
            candidates=candidates,
            source=CharacterSource.USER_SELECTED,
        )

    def replace_range(
        self,
        start: int,
        remove_count: int,
        segments: tuple[CharacterSegment, ...],
        characters: tuple[WordPredictionCharacter, ...],
    ) -> None:
        self.segments[start : start + remove_count] = segments
        self.characters[start : start + remove_count] = characters
        self._renumber()
        self.selected_position = min(start, len(self.characters) - 1)
        self.segmentation_status = SegmentationStatus.SUCCESS

    def mark_accepted(self) -> None:
        if not self.can_accept:
            raise ValueError("Resolve every uncertain character and segmentation issue first")
        self.status = WholeWordDraftStatus.ACCEPTED

    def cancel(self) -> None:
        self.status = WholeWordDraftStatus.CANCELLED

    def _renumber(self) -> None:
        self.segments = [
            replace(segment, position=position) for position, segment in enumerate(self.segments)
        ]
        self.characters = [
            replace(character, position=position)
            for position, character in enumerate(self.characters)
        ]
