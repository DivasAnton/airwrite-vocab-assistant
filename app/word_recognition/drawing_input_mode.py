from enum import StrEnum


class DrawingInputMode(StrEnum):
    CHARACTER = "CHARACTER"
    ISOLATED_WORD = "ISOLATED_WORD"

    @classmethod
    def from_string(cls, value: str) -> "DrawingInputMode":
        normalized = value.strip().upper()
        if normalized == "WORD":
            normalized = cls.ISOLATED_WORD.value
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError("Drawing input mode must be character or isolated_word") from error
