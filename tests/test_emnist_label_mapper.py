import numpy as np
import pytest

from app.ml.emnist_label_mapper import EMNISTLabelMapper
from app.ml.exceptions import InvalidEMNISTLabelError


def test_emnist_label_mapper_maps_once_without_mutating_input() -> None:
    raw = np.array([1, 2, 26], dtype=np.uint8)
    original = raw.copy()
    mapped = EMNISTLabelMapper.map_raw_labels(raw)
    assert mapped.tolist() == [0, 1, 25]
    assert mapped.dtype == np.int64
    assert np.array_equal(raw, original)


@pytest.mark.parametrize("label", [0, 27])
def test_emnist_label_mapper_rejects_out_of_range_values(label: int) -> None:
    with pytest.raises(InvalidEMNISTLabelError):
        EMNISTLabelMapper.map_raw_label(label)
