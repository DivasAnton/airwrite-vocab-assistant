from app.inference.case_input_state import CaseInputState
from app.inference.character_case_mode import CharacterCaseMode


def test_locked_modes_and_shift_produce_expected_effective_mode() -> None:
    state = CaseInputState(CharacterCaseMode.LOWERCASE)
    assert state.get_effective_mode() is CharacterCaseMode.LOWERCASE

    state.toggle_shift_next()
    assert state.get_effective_mode() is CharacterCaseMode.UPPERCASE

    state.set_uppercase()
    assert state.shift_next is False
    assert state.get_effective_mode() is CharacterCaseMode.UPPERCASE

    state.toggle_shift_next()
    assert state.get_effective_mode() is CharacterCaseMode.LOWERCASE
    state.consume_shift_after_commit()
    assert state.shift_next is False


def test_case_selection_is_immutable_and_independent_from_global_state() -> None:
    state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    selection = state.create_selection()

    state.set_lowercase()

    assert selection.mode is CharacterCaseMode.UPPERCASE
    assert selection.shift_was_active is True
    assert state.get_effective_mode() is CharacterCaseMode.LOWERCASE
