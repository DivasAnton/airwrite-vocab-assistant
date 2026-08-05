from pathlib import Path

import numpy as np
import pytest

from app.storage.character_dataset_saver import CharacterDatasetImageSaver
from app.storage.exceptions import InvalidDrawingImageError


def make_processed_image() -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[8:20, 10:18] = 255
    return image


def test_dataset_saver_writes_image_under_label_directory(tmp_path: Path) -> None:
    saver = CharacterDatasetImageSaver(output_dir=tmp_path, filename_prefix="sample")

    result = saver.save("a", make_processed_image())

    assert result.label == "A"
    assert result.file_path.parent == tmp_path / "A"
    assert result.file_path.exists()
    assert result.file_path.name.startswith("sample_A_")
    assert result.file_path.suffix == ".png"


def test_dataset_saver_rejects_invalid_label(tmp_path: Path) -> None:
    saver = CharacterDatasetImageSaver(output_dir=tmp_path)

    with pytest.raises(ValueError):
        saver.save("AA", make_processed_image())


def test_dataset_saver_rejects_color_image(tmp_path: Path) -> None:
    saver = CharacterDatasetImageSaver(output_dir=tmp_path)

    with pytest.raises(InvalidDrawingImageError):
        saver.save("A", np.zeros((28, 28, 3), dtype=np.uint8))


def test_dataset_label_navigation_wraps() -> None:
    assert CharacterDatasetImageSaver.next_label("Z") == "A"
    assert CharacterDatasetImageSaver.previous_label("A") == "Z"
