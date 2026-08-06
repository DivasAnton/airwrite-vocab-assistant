from dataclasses import dataclass
from enum import StrEnum

from app.inference.case_input_state import CaseInputState


class CaseAction(StrEnum):
    SET_LOWERCASE = "SET_LOWERCASE"
    SET_UPPERCASE = "SET_UPPERCASE"
    TOGGLE_SHIFT_NEXT = "TOGGLE_SHIFT_NEXT"
    CANCEL_SHIFT = "CANCEL_SHIFT"


@dataclass(frozen=True)
class CaseModeControllerResult:
    action: CaseAction
    message: str


class CaseModeController:
    def __init__(self, state: CaseInputState) -> None:
        self.state = state

    def handle(self, action: CaseAction) -> CaseModeControllerResult:
        if action is CaseAction.SET_LOWERCASE:
            self.state.set_lowercase()
            message = "Case mode set to lowercase"
        elif action is CaseAction.SET_UPPERCASE:
            self.state.set_uppercase()
            message = "Case mode set to uppercase"
        elif action is CaseAction.TOGGLE_SHIFT_NEXT:
            self.state.toggle_shift_next()
            message = (
                f"Next character: {self.state.get_effective_mode().value}"
                if self.state.shift_next
                else "Shift-next cancelled"
            )
        elif action is CaseAction.CANCEL_SHIFT:
            self.state.cancel_shift()
            message = "Shift-next cancelled"
        else:
            raise ValueError(f"Unsupported case action: {action!r}")
        return CaseModeControllerResult(action=action, message=message)
