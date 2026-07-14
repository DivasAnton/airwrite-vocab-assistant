from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from app.storage.drawing_image_saver import DrawingImageSaver
from app.storage.exceptions import DrawingImageSaveError, InvalidDrawingImageError
from app.storage.save_result import SaveStatus


def make_image() -> NDArray[np.uint8]:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[4:20, 6:26] = (255, 255, 255)
    return image


def test_save_writes_png_to_output_dir(tmp_path: Path) -> None:
    saver = DrawingImageSaver(output_dir=tmp_path)

    result = saver.save(make_image())

    assert result.status == SaveStatus.SAVED
    assert result.file_path is not None
    assert result.file_path.exists()
    assert result.file_path.suffix == ".png"
    assert result.file_path.parent == tmp_path


def test_save_creates_output_dir_when_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "drawings"
    saver = DrawingImageSaver(output_dir=output_dir)

    result = saver.save(make_image())

    assert output_dir.exists()
    assert result.file_path is not None
    assert result.file_path.exists()


def test_save_uses_unique_filenames(tmp_path: Path) -> None:
    saver = DrawingImageSaver(output_dir=tmp_path, filename_prefix="airwrite")

    first = saver.save(make_image())
    second = saver.save(make_image())

    assert first.file_path is not None
    assert second.file_path is not None
    assert first.file_path != second.file_path
    assert first.file_path.name.startswith("airwrite_")
    assert second.file_path.name.startswith("airwrite_")


def test_save_does_not_mutate_input_image(tmp_path: Path) -> None:
    image = make_image()
    original = image.copy()
    saver = DrawingImageSaver(output_dir=tmp_path)

    saver.save(image)

    assert np.array_equal(image, original)


def test_saved_png_can_be_read_back(tmp_path: Path) -> None:
    image = make_image()
    saver = DrawingImageSaver(output_dir=tmp_path)

    result = saver.save(image)

    assert result.file_path is not None
    saved_image = cv2.imread(str(result.file_path), cv2.IMREAD_UNCHANGED)
    assert saved_image is not None
    assert saved_image.shape == image.shape


@pytest.mark.parametrize(
    "image",
    [
        np.array([], dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10,), dtype=np.uint8),
        object(),
    ],
)
def test_save_rejects_invalid_images(tmp_path: Path, image: object) -> None:
    saver = DrawingImageSaver(output_dir=tmp_path)

    with pytest.raises(InvalidDrawingImageError):
        saver.save(image)  # type: ignore[arg-type]


def test_save_raises_when_writer_returns_false(tmp_path: Path) -> None:
    saver = DrawingImageSaver(output_dir=tmp_path, writer=lambda _path, _image: False)

    with pytest.raises(DrawingImageSaveError):
        saver.save(make_image())


def test_save_raises_when_writer_fails(tmp_path: Path) -> None:
    def failing_writer(_path: str, _image: NDArray[np.uint8]) -> bool:
        raise OSError("disk is unavailable")

    saver = DrawingImageSaver(output_dir=tmp_path, writer=failing_writer)

    with pytest.raises(DrawingImageSaveError, match="disk is unavailable"):
        saver.save(make_image())


def test_constructor_rejects_non_png_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_format"):
        DrawingImageSaver(output_dir=tmp_path, image_format="jpg")
