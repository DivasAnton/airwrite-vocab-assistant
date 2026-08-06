from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CharacterBatch:
    segment_ids: tuple[str, ...]
    tensor: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not self.segment_ids:
            raise ValueError("Character batch must not be empty")
        if self.tensor.dtype != np.float32 or self.tensor.ndim != 4:
            raise ValueError("Character batch tensor must be a 4D float32 array")
        if self.tensor.shape[1:] != (28, 28, 1):
            raise ValueError("Character batch tensor must have shape (N, 28, 28, 1)")
        if self.tensor.shape[0] != len(self.segment_ids):
            raise ValueError("Character batch rows must match segment IDs")
        if not np.isfinite(self.tensor).all() or self.tensor.min() < 0.0 or self.tensor.max() > 1.0:
            raise ValueError("Character batch values must be finite and between 0 and 1")
