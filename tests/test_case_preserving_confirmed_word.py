import pytest

from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.confirmed_word import ConfirmedWord


def entry(character: str) -> CharacterEntry:
    mode = (
        CharacterCaseMode.UPPERCASE
        if character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else CharacterCaseMode.LOWERCASE
    )
    return CharacterEntry(
        identity=character.casefold(),
        rendered_character=character,
        case_mode=mode,
        shift_was_active=False,
        confidence=0.9,
        source=CharacterSource.AUTO_ACCEPTED,
        prediction_id=f"pred_{character}_{mode.value}",
    )


@pytest.mark.parametrize(
    ("word", "canonical"),
    [("cat", "cat"), ("Cat", "cat"), ("CAT", "cat"), ("AirWrite", "airwrite")],
)
def test_confirmed_word_preserves_original_and_exposes_canonical_form(
    word: str,
    canonical: str,
) -> None:
    confirmed = ConfirmedWord(word, tuple(entry(character) for character in word), 100)

    assert confirmed.word == word
    assert confirmed.canonical_word == canonical


def test_confirmed_word_rejects_word_that_does_not_match_entries() -> None:
    with pytest.raises(ValueError, match="match"):
        ConfirmedWord("Cat", (entry("C"), entry("a"), entry("r")), 100)
