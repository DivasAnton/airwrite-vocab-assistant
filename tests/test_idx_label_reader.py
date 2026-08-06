import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from app.ml.exceptions import InvalidIDXFileError
from app.ml.idx_label_reader import IDXLabelReader


def write_labels(
    path: Path, labels: np.ndarray, *, magic: int = 2049, count: int | None = None
) -> None:
    with gzip.open(path, "wb") as output:
        output.write(struct.pack(">II", magic, len(labels) if count is None else count))
        output.write(labels.tobytes())


def test_idx_label_reader_reads_uint8_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.gz"
    write_labels(path, np.array([1, 2, 26], dtype=np.uint8))
    assert IDXLabelReader().read(path).tolist() == [1, 2, 26]


@pytest.mark.parametrize("magic,count", [(999, 3), (2049, 4)])
def test_idx_label_reader_rejects_invalid_header_or_count(
    tmp_path: Path, magic: int, count: int
) -> None:
    path = tmp_path / "labels.gz"
    write_labels(path, np.array([1, 2, 3], dtype=np.uint8), magic=magic, count=count)
    with pytest.raises(InvalidIDXFileError):
        IDXLabelReader().read(path)
