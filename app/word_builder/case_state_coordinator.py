from app.inference.case_input_state import CaseInputState
from app.word_builder.pending_character_selection import PendingCharacterSelection
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder_result import WordBuilderResult


def update_case_state_after_word_action(
    result: WordBuilderResult,
    case_state: CaseInputState,
    cancelled_pending: PendingCharacterSelection | None = None,
) -> None:
    if result.did_append_character:
        if result.committed_entry is not None and result.committed_entry.shift_was_active:
            case_state.consume_shift_after_commit()
        return
    if (
        result.action == WordAction.PENDING_SELECTION_CANCELLED
        and cancelled_pending is not None
        and cancelled_pending.case_selection.shift_was_active
    ):
        case_state.cancel_shift()
    elif result.action == WordAction.WORD_CLEARED:
        case_state.cancel_shift()
