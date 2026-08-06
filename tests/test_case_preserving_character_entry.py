import pytest

from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.exceptions import InvalidCharacterError
from app.word_builder.supported_character_set import SupportedCharacterSet


def make_entry(
    identity: str,
    rendered: str,
    mode: CharacterCaseMode,
) -> CharacterEntry:
    return CharacterEntry(
        identity=identity,
        rendered_character=rendered,
        case_mode=mode,
        shift_was_active=False,
        confidence=0.8,
        source=CharacterSource.AUTO_ACCEPTED,
        prediction_id="pred_1",
    )


def test_supported_character_set_has_aligned_lowercase_and_uppercase_mappings() -> None:
    supported = SupportedCharacterSet.english_letters()

    assert supported.identity_count == 26
    assert len(supported.all_characters) == 52
    assert supported.lowercase_characters[0] == "a"
    assert supported.uppercase_characters[0] == "A"
    assert supported.lowercase_characters[-1] == "z"
    assert supported.uppercase_characters[-1] == "Z"


def test_supported_character_set_rejects_an_invalid_index_contract() -> None:
    lowercase = tuple("abcdefghijklmnopqrstuvwxyz")
    uppercase = tuple("BACDEFGHIJKLMNOPQRSTUVWXYZ")

    with pytest.raises(ValueError, match="index contract"):
        SupportedCharacterSet(lowercase, lowercase, uppercase)


@pytest.mark.parametrize(
    ("identity", "rendered", "mode"),
    [
        ("a", "a", CharacterCaseMode.LOWERCASE),
        ("a", "A", CharacterCaseMode.UPPERCASE),
    ],
)
def test_character_entry_preserves_valid_identity_and_case(
    identity: str,
    rendered: str,
    mode: CharacterCaseMode,
) -> None:
    entry = make_entry(identity, rendered, mode)

    assert entry.identity == identity
    assert entry.rendered_character == rendered
    assert entry.character == rendered


@pytest.mark.parametrize(
    ("identity", "rendered", "mode"),
    [
        ("a", "B", CharacterCaseMode.UPPERCASE),
        ("d", "d", CharacterCaseMode.UPPERCASE),
    ],
)
def test_character_entry_rejects_identity_or_case_mismatch(
    identity: str,
    rendered: str,
    mode: CharacterCaseMode,
) -> None:
    with pytest.raises(InvalidCharacterError, match="match"):
        make_entry(identity, rendered, mode)


def test_character_entry_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="confidence"):
        CharacterEntry(
            identity="a",
            rendered_character="a",
            case_mode=CharacterCaseMode.LOWERCASE,
            shift_was_active=False,
            confidence=1.1,
            source=CharacterSource.AUTO_ACCEPTED,
            prediction_id="pred_1",
        )
    with pytest.raises(ValueError, match="prediction_id"):
        CharacterEntry(
            identity="a",
            rendered_character="a",
            case_mode=CharacterCaseMode.LOWERCASE,
            shift_was_active=False,
            confidence=0.8,
            source=CharacterSource.AUTO_ACCEPTED,
            prediction_id=" ",
        )


def test_character_entry_requires_a_boolean_shift_snapshot() -> None:
    with pytest.raises(ValueError, match="boolean"):
        CharacterEntry(
            identity="a",
            rendered_character="a",
            case_mode=CharacterCaseMode.LOWERCASE,
            shift_was_active=1,  # type: ignore[arg-type]
            confidence=0.8,
            source=CharacterSource.AUTO_ACCEPTED,
            prediction_id="pred_1",
        )
