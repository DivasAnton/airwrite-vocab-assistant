import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from app.ml.exceptions import InvalidIDXFileError
from app.ml.idx_image_reader import IDXImageReader


def write_images(
    path: Path,
    images: np.ndarray,
    *,
    magic: int = 2051,
    count: int | None = None,
    rows: int = 28,
    columns: int = 28,
) -> None:
    with gzip.open(path, "wb") as output:
        output.write(
            struct.pack(">IIII", magic, len(images) if count is None else count, rows, columns)
        )
        output.write(images.tobytes())


def test_idx_image_reader_preserves_raw_orientation(tmp_path: Path) -> None:
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 2, 7] = 255
    path = tmp_path / "images.gz"
    write_images(path, images)
    result = IDXImageReader().read(path)
    assert np.array_equal(result, images)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("magic,rows,columns", [(999, 28, 28), (2051, 27, 28), (2051, 28, 27)])
def test_idx_image_reader_rejects_invalid_header(
    tmp_path: Path, magic: int, rows: int, columns: int
) -> None:
    path = tmp_path / "images.gz"
    write_images(
        path, np.zeros((1, rows, columns), dtype=np.uint8), magic=magic, rows=rows, columns=columns
    )
    with pytest.raises(InvalidIDXFileError):
        IDXImageReader().read(path)


def test_idx_image_reader_rejects_count_payload_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "images.gz"
    write_images(path, np.zeros((1, 28, 28), dtype=np.uint8), count=2)
    with pytest.raises(InvalidIDXFileError):
        IDXImageReader().read(path)
