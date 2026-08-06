import numpy as np
import pytest

from app.ml.emnist_dataset_auditor import EMNISTDatasetAuditor
from app.ml.emnist_letters_dataset import EMNISTDatasetSplit, EMNISTLettersDataset
from app.ml.exceptions import InvalidEMNISTDatasetError


def make_dataset(labels: np.ndarray | None = None) -> EMNISTLettersDataset:
    mapped = np.arange(26, dtype=np.int64) if labels is None else labels
    images = np.ones((len(mapped), 28, 28), dtype=np.uint8)
    images[0] = 0
    raw = (mapped + 1).astype(np.uint8)
    return EMNISTLettersDataset(
        official_train=EMNISTDatasetSplit(images, mapped, "official_train"),
        official_test=EMNISTDatasetSplit(images.copy(), mapped.copy(), "official_test"),
        raw_train_labels=raw,
        raw_test_labels=raw.copy(),
    )


def test_auditor_reports_counts_ranges_and_empty_images() -> None:
    report = EMNISTDatasetAuditor().audit(make_dataset())
    assert report.observed_class_count == 26
    assert report.raw_label_range == (1, 26)
    assert report.mapped_label_range == (0, 25)
    assert report.splits["official_train"]["empty_image_count"] == 1
    assert report.splits["official_train"]["class_counts"]["25"] == 1


def test_auditor_rejects_missing_class() -> None:
    with pytest.raises(InvalidEMNISTDatasetError):
        EMNISTDatasetAuditor().audit(make_dataset(np.arange(25, dtype=np.int64)))


def test_dataset_split_rejects_shape_and_count_mismatch() -> None:
    with pytest.raises(InvalidEMNISTDatasetError):
        EMNISTDatasetSplit(
            np.zeros((2, 27, 28), dtype=np.uint8),
            np.array([0, 1], dtype=np.int64),
            "bad",
        )
    with pytest.raises(InvalidEMNISTDatasetError):
        EMNISTDatasetSplit(
            np.zeros((2, 28, 28), dtype=np.uint8),
            np.array([0], dtype=np.int64),
            "bad",
        )
