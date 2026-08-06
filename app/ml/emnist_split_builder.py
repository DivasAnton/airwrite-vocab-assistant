from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.ml.exceptions import EMNISTSplitError


@dataclass(frozen=True)
class EMNISTSplitIndices:
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    random_seed: int
    validation_ratio: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_indices=self.train_indices,
            validation_indices=self.validation_indices,
            random_seed=np.array(self.random_seed, dtype=np.int64),
            validation_ratio=np.array(self.validation_ratio, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "EMNISTSplitIndices":
        if not path.is_file():
            raise EMNISTSplitError(f"Split index file does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                train_indices=payload["train_indices"].astype(np.int64),
                validation_indices=payload["validation_indices"].astype(np.int64),
                random_seed=int(payload["random_seed"]),
                validation_ratio=float(payload["validation_ratio"]),
            )


class EMNISTSplitBuilder:
    def __init__(self, validation_ratio: float = 0.10, random_seed: int = 42) -> None:
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed

    def build(self, official_train_labels: NDArray[np.int64]) -> EMNISTSplitIndices:
        if official_train_labels.ndim != 1 or len(official_train_labels) == 0:
            raise EMNISTSplitError("Official training labels must be a non-empty 1D array")
        if not np.array_equal(np.unique(official_train_labels), np.arange(26)):
            raise EMNISTSplitError("Official training labels must contain every class 0-25")
        rng = np.random.default_rng(self.random_seed)
        train_parts: list[NDArray[np.int64]] = []
        validation_parts: list[NDArray[np.int64]] = []
        for class_index in range(26):
            indices = np.flatnonzero(official_train_labels == class_index).astype(np.int64)
            if len(indices) < 2:
                raise EMNISTSplitError(f"Class {class_index} needs at least two samples")
            shuffled = rng.permutation(indices)
            validation_count = max(
                1, min(len(indices) - 1, round(len(indices) * self.validation_ratio))
            )
            validation_parts.append(shuffled[:validation_count])
            train_parts.append(shuffled[validation_count:])
        return EMNISTSplitIndices(
            train_indices=np.sort(np.concatenate(train_parts)),
            validation_indices=np.sort(np.concatenate(validation_parts)),
            random_seed=self.random_seed,
            validation_ratio=self.validation_ratio,
        )

    @staticmethod
    def validate(indices: EMNISTSplitIndices, sample_count: int) -> None:
        train = indices.train_indices
        validation = indices.validation_indices
        if np.intersect1d(train, validation).size:
            raise EMNISTSplitError("Train and validation indices overlap")
        combined = np.sort(np.concatenate((train, validation)))
        if not np.array_equal(combined, np.arange(sample_count)):
            raise EMNISTSplitError("Split indices do not cover official training samples exactly")
