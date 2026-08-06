import pytest

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.identity_candidate import IdentityCandidate
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


def resolver() -> CaseLabelResolver:
    return CaseLabelResolver(
        LETTER_IDENTITY_LABELS,
        LETTER_IDENTITY_LABELS,
        tuple(label.upper() for label in LETTER_IDENTITY_LABELS),
    )


@pytest.mark.parametrize(
    ("class_index", "mode", "identity", "character"),
    [
        (0, CharacterCaseMode.LOWERCASE, "a", "a"),
        (0, CharacterCaseMode.UPPERCASE, "a", "A"),
        (25, CharacterCaseMode.LOWERCASE, "z", "z"),
        (25, CharacterCaseMode.UPPERCASE, "z", "Z"),
    ],
)
def test_resolver_maps_identity_to_selected_case(
    class_index: int,
    mode: CharacterCaseMode,
    identity: str,
    character: str,
) -> None:
    resolved = resolver().resolve(class_index, mode)
    assert resolved.identity == identity
    assert resolved.rendered_character == character


@pytest.mark.parametrize("class_index", [-1, 26])
def test_resolver_rejects_out_of_range_index(class_index: int) -> None:
    with pytest.raises(IndexError):
        resolver().resolve(class_index, CharacterCaseMode.LOWERCASE)


def test_resolver_rejects_misaligned_and_duplicate_mappings() -> None:
    uppercase = tuple(label.upper() for label in LETTER_IDENTITY_LABELS)
    with pytest.raises(ValueError):
        CaseLabelResolver(LETTER_IDENTITY_LABELS[:-1], LETTER_IDENTITY_LABELS, uppercase)
    with pytest.raises(ValueError):
        CaseLabelResolver(
            LETTER_IDENTITY_LABELS,
            (*LETTER_IDENTITY_LABELS[:-1], "a"),
            uppercase,
        )


def test_resolved_candidate_preserves_identity_confidence_and_rank() -> None:
    raw = IdentityCandidate("d", 3, 0.72, 1)
    candidate = resolver().resolve_candidate(raw, CharacterCaseMode.UPPERCASE)

    assert candidate.label == candidate.rendered_character == "D"
    assert candidate.identity == "d"
    assert candidate.confidence == 0.72
    assert candidate.rank == 1
