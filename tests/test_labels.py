import pytest

from app.ml.labels import CHARACTER_LABELS, index_to_label, label_to_index, normalize_label


def test_character_labels_are_fixed_uppercase_a_to_z() -> None:
    assert CHARACTER_LABELS == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert label_to_index("A") == 0
    assert label_to_index("z") == 25
    assert index_to_label(25) == "Z"


def test_normalize_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        normalize_label("AA")
