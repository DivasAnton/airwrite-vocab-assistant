import gzip
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.ml.exceptions import InvalidIDXFileError


class IDXLabelReader:
    MAGIC_NUMBER = 2049

    def read(self, path: Path) -> NDArray[np.uint8]:
        if not path.is_file():
            raise InvalidIDXFileError(f"IDX label file does not exist: {path}")
        try:
            with gzip.open(path, "rb") as input_file:
                header = input_file.read(8)
                if len(header) != 8:
                    raise InvalidIDXFileError("IDX label header must contain 8 bytes")
                magic, count = struct.unpack(">II", header)
                payload = input_file.read()
        except OSError as error:
            raise InvalidIDXFileError(f"Could not read gzip IDX label file: {path}") from error

        if magic != self.MAGIC_NUMBER:
            raise InvalidIDXFileError(
                f"Invalid IDX label magic number: expected {self.MAGIC_NUMBER}, got {magic}"
            )
        if count <= 0:
            raise InvalidIDXFileError("IDX label count must be greater than 0")
        if len(payload) != count:
            raise InvalidIDXFileError(
                f"IDX label payload must contain {count} bytes, got {len(payload)}"
            )
        return np.frombuffer(payload, dtype=np.uint8).copy()
