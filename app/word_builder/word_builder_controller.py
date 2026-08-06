from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.pending_character_selection import PendingCharacterSelection
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_result import WordBuilderResult


class WordBuilderController:
    def __init__(
        self,
        word_builder: WordBuilder,
        auto_append_accepted: bool = True,
        require_selection_for_uncertain: bool = True,
    ) -> None:
        self.word_builder = word_builder
        self.auto_append_accepted = auto_append_accepted
        self.require_selection_for_uncertain = require_selection_for_uncertain

    def handle_prediction(self, prediction: PredictionResult) -> WordBuilderResult:
        if prediction.status == PredictionStatus.SKIPPED_EMPTY:
            return self.word_builder.snapshot(WordAction.NONE, "Nothing added to the word")
        if prediction.status == PredictionStatus.FAILED:
            return self.word_builder.snapshot(WordAction.ERROR, "Prediction failed; word unchanged")
        if prediction.prediction_id is None:
            return self.word_builder.snapshot(
                WordAction.ERROR, "Prediction event has no identifier"
            )
        if not prediction.candidates or prediction.top_prediction is None:
            return self.word_builder.snapshot(WordAction.ERROR, "Prediction has no candidates")
        if any(
            candidate.case_mode is not prediction.case_selection.mode
            for candidate in prediction.candidates
        ):
            return self.word_builder.snapshot(
                WordAction.ERROR,
                "Prediction candidates do not match the case snapshot",
            )

        if prediction.status == PredictionStatus.ACCEPTED and self.auto_append_accepted:
            top_prediction = prediction.top_prediction
            case_mode = top_prediction.case_mode
            if case_mode is None:
                return self.word_builder.snapshot(
                    WordAction.ERROR,
                    "Prediction candidate has no case mode",
                )
            return self.word_builder.append_entry(
                CharacterEntry(
                    identity=top_prediction.identity,
                    rendered_character=top_prediction.rendered_character,
                    case_mode=case_mode,
                    shift_was_active=prediction.case_selection.shift_was_active,
                    confidence=top_prediction.confidence,
                    source=CharacterSource.AUTO_ACCEPTED,
                    prediction_id=prediction.prediction_id,
                )
            )

        if (
            prediction.status == PredictionStatus.UNCERTAIN
            and not self.require_selection_for_uncertain
        ):
            return self.word_builder.snapshot(
                WordAction.NONE,
                "Uncertain prediction ignored by configuration",
            )

        return self.word_builder.create_pending_selection(
            PendingCharacterSelection(
                prediction_id=prediction.prediction_id,
                candidates=prediction.candidates,
                case_selection=prediction.case_selection,
            )
        )

    def select_candidate(self, rank: int) -> WordBuilderResult:
        return self.word_builder.select_pending_candidate(rank)

    def cancel_pending(self) -> WordBuilderResult:
        return self.word_builder.cancel_pending_selection()

    def backspace(self) -> WordBuilderResult:
        return self.word_builder.backspace()

    def clear_word(self) -> WordBuilderResult:
        return self.word_builder.clear()

    def confirm(self, timestamp_ms: int) -> WordBuilderResult:
        return self.word_builder.confirm(timestamp_ms)

    def start_new_word(self) -> WordBuilderResult:
        return self.word_builder.start_new_word()

    def append_manual(self, character: str) -> WordBuilderResult:
        supported = self.word_builder.supported_character_set
        try:
            identity = supported.identity_for(character)
            case_mode = supported.case_mode_for(character)
        except ValueError as error:
            return self.word_builder.snapshot(WordAction.ERROR, str(error))
        return self.word_builder.append_entry(
            CharacterEntry(
                identity=identity,
                rendered_character=character,
                case_mode=case_mode,
                shift_was_active=False,
                confidence=None,
                source=CharacterSource.MANUAL_INPUT,
                prediction_id=None,
            )
        )
