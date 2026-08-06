import pytest

from app.ml.letter_identity_labels import (
    LETTER_IDENTITY_LABELS,
    display_labels_payload,
    identity_to_index,
    index_to_identity,
)


def test_identity_labels_are_lowercase_a_to_z() -> None:
    assert LETTER_IDENTITY_LABELS == tuple("abcdefghijklmnopqrstuvwxyz")
    assert identity_to_index("a") == 0
    assert identity_to_index("z") == 25
    assert index_to_identity(25) == "z"


def test_identity_mapping_rejects_case_and_invalid_indices() -> None:
    with pytest.raises(ValueError):
        identity_to_index("A")
    with pytest.raises(ValueError):
        index_to_identity(26)


def test_display_labels_keep_identity_index_order() -> None:
    assert display_labels_payload(uppercase=True)["labels"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert display_labels_payload(uppercase=False)["labels"] == list(LETTER_IDENTITY_LABELS)
