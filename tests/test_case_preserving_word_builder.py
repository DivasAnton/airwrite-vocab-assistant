from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder


def entry(character: str, prediction_id: str) -> CharacterEntry:
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
        prediction_id=prediction_id,
    )


def append_word(builder: WordBuilder, word: str) -> None:
    for index, character in enumerate(word):
        builder.append_entry(entry(character, f"pred_{index}"))


def test_builds_lowercase_capitalized_uppercase_and_mixed_case_words() -> None:
    for word in ("cat", "Cat", "CAT", "AirWrite"):
        builder = WordBuilder()
        append_word(builder, word)

        assert builder.current_word == word
        assert builder.canonical_word == word.casefold()


def test_repeated_letters_and_mixed_case_repetitions_are_not_deduplicated() -> None:
    builder = WordBuilder()
    append_word(builder, "apple")
    assert builder.current_word == "apple"

    mixed = WordBuilder()
    mixed.append_entry(entry("A", "pred_upper"))
    result = mixed.append_entry(entry("a", "pred_lower"))
    assert result.current_word == "Aa"


def test_duplicate_prediction_is_ignored_but_same_character_with_new_id_is_kept() -> None:
    builder = WordBuilder()
    first = entry("P", "pred_1")

    builder.append_entry(first)
    duplicate = builder.append_entry(first)
    repeated = builder.append_entry(entry("P", "pred_2"))

    assert duplicate.action is WordAction.DUPLICATE_IGNORED
    assert repeated.current_word == "PP"


def test_backspace_and_confirm_preserve_remaining_case() -> None:
    builder = WordBuilder()
    append_word(builder, "AirW")

    backed = builder.backspace()
    confirmed = builder.confirm(123)

    assert backed.current_word == "Air"
    assert [item.rendered_character for item in builder.entries] == ["A", "i", "r"]
    assert confirmed.confirmed_word is not None
    assert confirmed.confirmed_word.word == "Air"
    assert confirmed.confirmed_word.canonical_word == "air"
