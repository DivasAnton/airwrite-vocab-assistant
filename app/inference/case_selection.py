from dataclasses import dataclass

from app.inference.character_case_mode import CharacterCaseMode


@dataclass(frozen=True)
class CaseSelection:
    mode: CharacterCaseMode
    shift_was_active: bool
