import gzip
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.ml.exceptions import InvalidIDXFileError


class IDXImageReader:
    MAGIC_NUMBER = 2051

    def __init__(self, expected_rows: int = 28, expected_columns: int = 28) -> None:
        if expected_rows <= 0 or expected_columns <= 0:
            raise ValueError("Expected IDX image dimensions must be greater than 0")
        self.expected_rows = expected_rows
        self.expected_columns = expected_columns

    def read(self, path: Path) -> NDArray[np.uint8]:
        if not path.is_file():
            raise InvalidIDXFileError(f"IDX image file does not exist: {path}")
        try:
            with gzip.open(path, "rb") as input_file:
                header = input_file.read(16)
                if len(header) != 16:
                    raise InvalidIDXFileError("IDX image header must contain 16 bytes")
                magic, count, rows, columns = struct.unpack(">IIII", header)
                payload = input_file.read()
        except OSError as error:
            raise InvalidIDXFileError(f"Could not read gzip IDX image file: {path}") from error

        if magic != self.MAGIC_NUMBER:
            raise InvalidIDXFileError(
                f"Invalid IDX image magic number: expected {self.MAGIC_NUMBER}, got {magic}"
            )
        if count <= 0:
            raise InvalidIDXFileError("IDX image count must be greater than 0")
        if (rows, columns) != (self.expected_rows, self.expected_columns):
            raise InvalidIDXFileError(
                "IDX image shape must be "
                f"{self.expected_rows}x{self.expected_columns}, got {rows}x{columns}"
            )
        expected_bytes = count * rows * columns
        if len(payload) != expected_bytes:
            raise InvalidIDXFileError(
                f"IDX image payload must contain {expected_bytes} bytes, got {len(payload)}"
            )
        return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns).copy()
