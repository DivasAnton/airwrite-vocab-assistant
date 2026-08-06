import pytest

from app.inference.case_input_state import CaseInputState
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.word_builder.case_state_coordinator import update_case_state_after_word_action
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_controller import WordBuilderController
from tests.test_case_preserving_word_builder_controller import prediction


def test_accepted_shift_is_consumed_only_after_character_commit() -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    controller = WordBuilderController(WordBuilder())

    result = controller.handle_prediction(
        prediction("pred_c", ("c",), CharacterCaseMode.UPPERCASE, shift=True)
    )
    update_case_state_after_word_action(result, case_state)

    assert result.current_word == "C"
    assert case_state.shift_next is False
    assert case_state.locked_mode is CharacterCaseMode.LOWERCASE


def test_uncertain_shift_stays_active_until_selection_then_is_consumed() -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    controller = WordBuilderController(WordBuilder())
    pending = controller.handle_prediction(
        prediction(
            "pred_c",
            ("c", "e", "o"),
            CharacterCaseMode.UPPERCASE,
            status=PredictionStatus.UNCERTAIN,
            shift=True,
        )
    )
    update_case_state_after_word_action(pending, case_state)
    assert case_state.shift_next is True

    selected = controller.select_candidate(1)
    update_case_state_after_word_action(selected, case_state)

    assert selected.current_word == "C"
    assert case_state.shift_next is False


def test_cancel_shift_pending_cancels_shift_and_preserves_word() -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    controller = WordBuilderController(WordBuilder())
    controller.handle_prediction(
        prediction(
            "pred_c",
            ("c", "e", "o"),
            CharacterCaseMode.UPPERCASE,
            status=PredictionStatus.UNCERTAIN,
            shift=True,
        )
    )
    cancelled_pending = controller.word_builder.pending_selection

    cancelled = controller.cancel_pending()
    update_case_state_after_word_action(cancelled, case_state, cancelled_pending)

    assert cancelled.action is WordAction.PENDING_SELECTION_CANCELLED
    assert cancelled.current_word == ""
    assert case_state.shift_next is False
    assert case_state.locked_mode is CharacterCaseMode.LOWERCASE


@pytest.mark.parametrize("status", [PredictionStatus.FAILED, PredictionStatus.SKIPPED_EMPTY])
def test_failed_or_empty_prediction_does_not_consume_shift(
    status: PredictionStatus,
) -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    controller = WordBuilderController(WordBuilder())
    failed = controller.handle_prediction(
        PredictionResult(
            status=status,
            top_prediction=None,
            candidates=(),
            confidence_margin=None,
            inference_time_ms=None,
            model_version="1.0.0",
            message=status.value,
            prediction_id=f"pred_{status.value}",
            case_selection=case_state.create_selection(),
        )
    )

    update_case_state_after_word_action(failed, case_state)

    assert case_state.shift_next is True


def test_changing_global_mode_does_not_rewrite_pending_candidates() -> None:
    case_state = CaseInputState(CharacterCaseMode.UPPERCASE)
    controller = WordBuilderController(WordBuilder())
    controller.handle_prediction(
        prediction(
            "pred_d",
            ("o", "d", "g"),
            CharacterCaseMode.UPPERCASE,
            status=PredictionStatus.UNCERTAIN,
        )
    )

    case_state.set_lowercase()
    selected = controller.select_candidate(2)
    update_case_state_after_word_action(selected, case_state)

    assert selected.current_word == "D"
    assert case_state.locked_mode is CharacterCaseMode.LOWERCASE
    assert case_state.get_effective_mode() is CharacterCaseMode.LOWERCASE


def test_clear_and_start_new_word_cancel_shift_but_keep_locked_mode() -> None:
    case_state = CaseInputState(CharacterCaseMode.UPPERCASE, shift_next=True)
    controller = WordBuilderController(WordBuilder())

    cleared = controller.clear_word()
    update_case_state_after_word_action(cleared, case_state)
    assert case_state.shift_next is False
    assert case_state.locked_mode is CharacterCaseMode.UPPERCASE

    controller.handle_prediction(prediction("pred_a", ("a",), CharacterCaseMode.UPPERCASE))
    controller.confirm(1)
    case_state.toggle_shift_next()
    started = controller.start_new_word()
    update_case_state_after_word_action(started, case_state)

    assert case_state.shift_next is False
    assert case_state.locked_mode is CharacterCaseMode.UPPERCASE
