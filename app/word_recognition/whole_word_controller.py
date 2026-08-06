from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_result import WordBuilderResult
from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.isolated_letter_word_recognition_strategy import (
    IsolatedLetterWordRecognitionStrategy,
)
from app.word_recognition.whole_word_draft import WholeWordDraft
from app.word_recognition.whole_word_prediction_status import WholeWordPredictionStatus
from app.word_recognition.whole_word_recognition_service import WholeWordRecognitionService
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot


class WholeWordController:
    def __init__(
        self,
        recognition_service: WholeWordRecognitionService,
        correction_strategy: IsolatedLetterWordRecognitionStrategy,
        word_builder: WordBuilder,
        min_characters: int = 2,
        max_characters: int = 12,
    ) -> None:
        self.recognition_service = recognition_service
        self.correction_strategy = correction_strategy
        self.word_builder = word_builder
        self.min_characters = min_characters
        self.max_characters = max_characters
        self.draft: WholeWordDraft | None = None
        self.last_message = "Whole-word mode ready"

    def recognize(
        self,
        snapshot: WordInputSnapshot,
        case_policy: WordCasePolicy,
    ) -> WholeWordDraft | None:
        if self.draft is not None:
            self.last_message = "Accept or cancel the current whole-word draft first"
            return self.draft
        result = self.recognition_service.recognize(
            snapshot,
            DrawingInputMode.ISOLATED_WORD,
            case_policy,
        )
        if result.status in {
            WholeWordPredictionStatus.EMPTY,
            WholeWordPredictionStatus.SEGMENTATION_FAILED,
            WholeWordPredictionStatus.INFERENCE_FAILED,
        }:
            self.last_message = result.message
            return None
        self.draft = WholeWordDraft.from_prediction(result, case_policy)
        self.last_message = result.message
        return self.draft

    def move_previous(self) -> None:
        if self.draft is not None:
            self.draft.move(-1)

    def move_next(self) -> None:
        if self.draft is not None:
            self.draft.move(1)

    def select_candidate(self, rank: int) -> None:
        self._require_draft().select_candidate(rank)
        self.last_message = f"Candidate {rank} selected"

    def toggle_case(self) -> None:
        self._require_draft().toggle_selected_case()
        self.last_message = "Selected character case toggled"

    def split_selected(self) -> bool:
        draft = self._require_draft()
        selected = draft.selected_position
        if len(draft.characters) >= self.max_characters:
            self.last_message = "Cannot split beyond the whole-word character limit"
            return False
        children = self.correction_strategy.segmenter.split_segment(draft.segments[selected])
        if children is None:
            self.last_message = "No reliable split point was found"
            return False
        modes = self._modes_for_correction(draft, selected, len(children))
        predicted = self.correction_strategy.predict_segments(
            draft.prediction_id,
            children,
            WordCasePolicy.CUSTOM,
            modes,
        )
        draft.replace_range(selected, 1, children, predicted)
        self.last_message = "Selected segment split; review both predictions"
        return True

    def merge_selected_with_next(self) -> bool:
        draft = self._require_draft()
        selected = draft.selected_position
        if selected >= len(draft.segments) - 1 or len(draft.characters) <= self.min_characters:
            self.last_message = "Selected segment cannot be merged"
            return False
        merged = self.correction_strategy.segmenter.merge_segments(
            draft.segments[selected],
            draft.segments[selected + 1],
        )
        mode = draft.characters[selected].case_mode
        predicted = self.correction_strategy.predict_segments(
            draft.prediction_id,
            (merged,),
            WordCasePolicy.CUSTOM,
            {0: mode},
        )
        draft.replace_range(selected, 2, (merged,), predicted)
        self.last_message = "Selected segment merged with the next segment"
        return True

    def accept(self) -> WordBuilderResult:
        draft = self._require_draft()
        if not draft.can_accept:
            self.last_message = "Resolve every uncertain character and segmentation issue first"
            return self.word_builder.snapshot(WordAction.ERROR, self.last_message)
        entries = tuple(
            CharacterEntry(
                identity=character.identity,
                rendered_character=character.rendered_character,
                case_mode=character.case_mode,
                shift_was_active=False,
                confidence=character.confidence,
                source=character.source,
                prediction_id=f"{draft.prediction_id}:{character.position}",
            )
            for character in draft.characters
        )
        result = self.word_builder.commit_entries_atomic(entries, draft.prediction_id)
        if result.action is WordAction.WHOLE_WORD_COMMITTED:
            draft.mark_accepted()
            self.draft = None
        self.last_message = result.message
        return result

    def cancel(self) -> None:
        draft = self._require_draft()
        draft.cancel()
        self.draft = None
        self.last_message = "Whole-word draft cancelled"

    def _require_draft(self) -> WholeWordDraft:
        if self.draft is None:
            raise ValueError("No whole-word draft is active")
        return self.draft

    @staticmethod
    def _modes_for_correction(
        draft: WholeWordDraft,
        start: int,
        count: int,
    ) -> dict[int, CharacterCaseMode]:
        mode = draft.characters[start].case_mode
        return {position: mode for position in range(count)}
