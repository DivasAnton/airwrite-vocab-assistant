from enum import StrEnum


class CharacterCaseMode(StrEnum):
    LOWERCASE = "LOWERCASE"
    UPPERCASE = "UPPERCASE"

    @classmethod
    def from_string(cls, value: str) -> "CharacterCaseMode":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError("Character case mode must be lowercase or uppercase") from error

    def opposite(self) -> "CharacterCaseMode":
        if self is CharacterCaseMode.LOWERCASE:
            return CharacterCaseMode.UPPERCASE
        return CharacterCaseMode.LOWERCASE
