from pathlib import Path

import cv2
import numpy as np

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.dataset_validator import DatasetValidator
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor


def make_entry(path: Path, label: str = "A", label_index: int = 0) -> DatasetManifestEntry:
    return DatasetManifestEntry(path, label, label_index, "airwrite")


def write_character(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[5:23, 12:16] = 255
    assert cv2.imwrite(str(path), image)


def test_validator_accepts_valid_grayscale_28_by_28_png(tmp_path: Path) -> None:
    path = tmp_path / "A" / "valid.png"
    write_character(path)

    report = DatasetValidator(HandwritingPreprocessor()).validate([make_entry(path)])

    assert report.valid_entries == [make_entry(path)]
    assert report.issues == []


def test_validator_reports_wrong_shape_and_exact_duplicate(tmp_path: Path) -> None:
    wrong_shape = tmp_path / "A" / "wrong.png"
    wrong_shape.parent.mkdir(parents=True)
    assert cv2.imwrite(str(wrong_shape), np.full((20, 20), 255, dtype=np.uint8))
    first = tmp_path / "A" / "first.png"
    second = tmp_path / "A" / "second.png"
    write_character(first)
    second.write_bytes(first.read_bytes())

    report = DatasetValidator(HandwritingPreprocessor()).validate(
        [make_entry(wrong_shape), make_entry(first), make_entry(second)]
    )

    assert [issue.error_type for issue in report.issues] == [
        "wrong_image_shape",
        "exact_duplicate",
    ]
    assert report.valid_entries == [make_entry(first)]


def test_validator_reports_label_path_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "B" / "sample.png"
    write_character(path)

    report = DatasetValidator(HandwritingPreprocessor()).validate([make_entry(path)])

    assert report.issues[0].error_type == "label_path_mismatch"
