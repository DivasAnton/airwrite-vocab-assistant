import pytest

from app.inference.case_input_state import CaseInputState
from app.inference.case_mode_controller import CaseAction, CaseModeController
from app.inference.character_case_mode import CharacterCaseMode
from app.utils.config import CaseControlSettings


def test_controller_handles_semantic_actions() -> None:
    state = CaseInputState(CharacterCaseMode.LOWERCASE)
    controller = CaseModeController(state)

    controller.handle(CaseAction.TOGGLE_SHIFT_NEXT)
    assert state.shift_next is True
    controller.handle(CaseAction.SET_UPPERCASE)
    assert state.locked_mode is CharacterCaseMode.UPPERCASE
    assert state.shift_next is False
    controller.handle(CaseAction.TOGGLE_SHIFT_NEXT)
    controller.handle(CaseAction.CANCEL_SHIFT)
    assert state.shift_next is False
    controller.handle(CaseAction.SET_LOWERCASE)
    assert state.locked_mode is CharacterCaseMode.LOWERCASE


def test_case_settings_reject_duplicate_keys_and_auto_mode() -> None:
    duplicate = CaseControlSettings(lowercase_mode_key="l", uppercase_mode_key="l")
    with pytest.raises(ValueError, match="distinct"):
        duplicate.validate()

    automatic = CaseControlSettings(enable_auto_case_mode=True)
    with pytest.raises(ValueError, match="must remain false"):
        automatic.validate()

    assert CaseControlSettings().default_mode is CharacterCaseMode.LOWERCASE
