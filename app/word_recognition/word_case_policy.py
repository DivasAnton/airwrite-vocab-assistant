from enum import StrEnum


class WordCasePolicy(StrEnum):
    LOWERCASE = "LOWERCASE"
    UPPERCASE = "UPPERCASE"
    CAPITALIZE_FIRST = "CAPITALIZE_FIRST"
    CUSTOM = "CUSTOM"

    @classmethod
    def from_string(cls, value: str) -> "WordCasePolicy":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(
                "Word case policy must be lowercase, uppercase, capitalize_first, or custom"
            ) from error
