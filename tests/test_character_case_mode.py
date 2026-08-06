import pytest

from app.inference.character_case_mode import CharacterCaseMode


def test_case_mode_parses_supported_values_without_auto_mode() -> None:
    assert CharacterCaseMode.from_string("lowercase") is CharacterCaseMode.LOWERCASE
    assert CharacterCaseMode.from_string(" UPPERCASE ") is CharacterCaseMode.UPPERCASE
    assert CharacterCaseMode.LOWERCASE.opposite() is CharacterCaseMode.UPPERCASE


@pytest.mark.parametrize("value", ["", "auto", "mixed"])
def test_case_mode_rejects_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="lowercase or uppercase"):
        CharacterCaseMode.from_string(value)
