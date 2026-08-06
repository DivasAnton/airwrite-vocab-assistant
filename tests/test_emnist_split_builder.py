from pathlib import Path

import numpy as np

from app.ml.emnist_split_builder import EMNISTSplitBuilder, EMNISTSplitIndices


def test_split_is_stratified_deterministic_disjoint_and_persistable(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(26, dtype=np.int64), 20)
    builder = EMNISTSplitBuilder(validation_ratio=0.2, random_seed=7)
    first = builder.build(labels)
    second = builder.build(labels)
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.intersect1d(first.train_indices, first.validation_indices).size == 0
    assert len(first.validation_indices) == 104
    assert set(labels[first.train_indices]) == set(range(26))
    assert set(labels[first.validation_indices]) == set(range(26))
    builder.validate(first, len(labels))
    path = tmp_path / "split_indices.npz"
    first.save(path)
    loaded = EMNISTSplitIndices.load(path)
    assert np.array_equal(first.validation_indices, loaded.validation_indices)
