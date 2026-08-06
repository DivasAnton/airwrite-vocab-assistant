from dataclasses import dataclass

from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode


@dataclass
class CaseInputState:
    locked_mode: CharacterCaseMode
    shift_next: bool = False

    def set_lowercase(self) -> None:
        self.locked_mode = CharacterCaseMode.LOWERCASE
        self.shift_next = False

    def set_uppercase(self) -> None:
        self.locked_mode = CharacterCaseMode.UPPERCASE
        self.shift_next = False

    def toggle_shift_next(self) -> None:
        self.shift_next = not self.shift_next

    def cancel_shift(self) -> None:
        self.shift_next = False

    def get_effective_mode(self) -> CharacterCaseMode:
        return self.locked_mode.opposite() if self.shift_next else self.locked_mode

    def create_selection(self) -> CaseSelection:
        return CaseSelection(
            mode=self.get_effective_mode(),
            shift_was_active=self.shift_next,
        )

    def consume_shift_after_commit(self) -> None:
        self.shift_next = False
