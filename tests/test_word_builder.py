from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_state import WordBuilderState


def entry(character: str, prediction_id: str) -> CharacterEntry:
    return CharacterEntry(
        identity=character.casefold(),
        rendered_character=character,
        case_mode=(
            CharacterCaseMode.UPPERCASE
            if character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            else CharacterCaseMode.LOWERCASE
        ),
        shift_was_active=False,
        confidence=0.90,
        source=CharacterSource.AUTO_ACCEPTED,
        prediction_id=prediction_id,
    )


def test_appends_characters_and_preserves_metadata() -> None:
    builder = WordBuilder()

    builder.append_character(entry("C", "pred_c"))
    builder.append_character(entry("A", "pred_a"))
    result = builder.append_character(entry("T", "pred_t"))

    assert result.current_word == "CAT"
    assert result.action == WordAction.CHARACTER_APPENDED
    assert builder.entries[-1].source == CharacterSource.AUTO_ACCEPTED
    assert builder.entries[-1].confidence == 0.90


def test_duplicate_id_is_ignored_but_repeated_character_with_new_id_is_valid() -> None:
    builder = WordBuilder()

    builder.append_character(entry("P", "pred_1"))
    duplicate = builder.append_character(entry("P", "pred_1"))
    repeated = builder.append_character(entry("P", "pred_2"))

    assert duplicate.action == WordAction.DUPLICATE_IGNORED
    assert repeated.current_word == "PP"


def test_backspace_removes_to_empty_without_crashing() -> None:
    builder = WordBuilder()
    for character in "CAT":
        builder.append_character(entry(character, f"pred_{character}"))

    assert builder.backspace().current_word == "CA"
    assert builder.backspace().current_word == "C"
    assert builder.backspace().state == WordBuilderState.EMPTY
    assert builder.backspace().action == WordAction.NONE


def test_limit_does_not_process_rejected_event_and_allows_retry() -> None:
    builder = WordBuilder(max_length=3)
    for index, character in enumerate("CAT"):
        builder.append_character(entry(character, f"pred_{index}"))

    limited = builder.append_character(entry("S", "pred_s"))
    builder.backspace()
    retried = builder.append_character(entry("S", "pred_s"))

    assert limited.action == WordAction.LIMIT_REACHED
    assert limited.current_word == "CAT"
    assert "pred_s" not in builder.processed_prediction_ids or retried.current_word == "CAS"
    assert retried.current_word == "CAS"


def test_confirm_rejects_empty_and_freezes_valid_word() -> None:
    builder = WordBuilder()
    assert builder.confirm(10).action == WordAction.ERROR
    for index, character in enumerate("CAT"):
        builder.append_character(entry(character, f"pred_{index}"))

    confirmed = builder.confirm(1234)
    edit = builder.backspace()

    assert confirmed.action == WordAction.WORD_CONFIRMED
    assert confirmed.state == WordBuilderState.CONFIRMED
    assert confirmed.confirmed_word is not None
    assert confirmed.confirmed_word.word == "CAT"
    assert confirmed.confirmed_word.confirmed_at_ms == 1234
    assert len(confirmed.confirmed_word.entries) == 3
    assert edit.action == WordAction.ERROR


def test_clear_and_start_new_word_reset_all_state() -> None:
    builder = WordBuilder()
    builder.append_character(entry("C", "pred_c"))
    cleared = builder.clear()
    builder.append_character(entry("D", "pred_d"))
    builder.confirm(1)
    started = builder.start_new_word()

    assert cleared.state == WordBuilderState.EMPTY
    assert started.state == WordBuilderState.EMPTY
    assert started.current_word == ""
    assert builder.processed_prediction_ids == frozenset()
