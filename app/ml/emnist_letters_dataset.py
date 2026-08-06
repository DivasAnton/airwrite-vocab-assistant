from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.ml.emnist_label_mapper import EMNISTLabelMapper
from app.ml.exceptions import InvalidEMNISTDatasetError
from app.ml.idx_image_reader import IDXImageReader
from app.ml.idx_label_reader import IDXLabelReader


@dataclass(frozen=True)
class EMNISTDatasetSplit:
    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    split_name: str

    def __post_init__(self) -> None:
        if self.images.ndim != 3 or self.images.shape[1:] != (28, 28):
            raise InvalidEMNISTDatasetError(
                f"{self.split_name} images must have shape (N,28,28), got {self.images.shape}"
            )
        if self.images.dtype != np.uint8:
            raise InvalidEMNISTDatasetError(f"{self.split_name} images must use uint8 dtype")
        if self.labels.ndim != 1 or self.labels.dtype != np.int64:
            raise InvalidEMNISTDatasetError(f"{self.split_name} labels must be 1D int64")
        if len(self.images) != len(self.labels):
            raise InvalidEMNISTDatasetError(
                f"{self.split_name} image/label count mismatch: "
                f"{len(self.images)} != {len(self.labels)}"
            )
        if len(self.labels) == 0 or np.any(self.labels < 0) or np.any(self.labels > 25):
            raise InvalidEMNISTDatasetError(
                f"{self.split_name} mapped labels must be non-empty and stay in 0-25"
            )


@dataclass(frozen=True)
class EMNISTLettersDataset:
    official_train: EMNISTDatasetSplit
    official_test: EMNISTDatasetSplit
    raw_train_labels: NDArray[np.uint8]
    raw_test_labels: NDArray[np.uint8]

    @classmethod
    def load(
        cls,
        *,
        train_images_path: Path,
        train_labels_path: Path,
        test_images_path: Path,
        test_labels_path: Path,
        image_reader: IDXImageReader | None = None,
        label_reader: IDXLabelReader | None = None,
        label_mapper: EMNISTLabelMapper | None = None,
    ) -> "EMNISTLettersDataset":
        selected_image_reader = image_reader or IDXImageReader()
        selected_label_reader = label_reader or IDXLabelReader()
        selected_mapper = label_mapper or EMNISTLabelMapper()
        train_images = selected_image_reader.read(train_images_path)
        raw_train_labels = selected_label_reader.read(train_labels_path)
        test_images = selected_image_reader.read(test_images_path)
        raw_test_labels = selected_label_reader.read(test_labels_path)
        cls._validate_pair("official_train", train_images, raw_train_labels)
        cls._validate_pair("official_test", test_images, raw_test_labels)
        train_labels = selected_mapper.map_raw_labels(raw_train_labels)
        test_labels = selected_mapper.map_raw_labels(raw_test_labels)
        return cls(
            official_train=EMNISTDatasetSplit(train_images, train_labels, "official_train"),
            official_test=EMNISTDatasetSplit(test_images, test_labels, "official_test"),
            raw_train_labels=raw_train_labels,
            raw_test_labels=raw_test_labels,
        )

    @staticmethod
    def _validate_pair(
        split_name: str,
        images: NDArray[np.uint8],
        labels: NDArray[np.uint8],
    ) -> None:
        if len(images) != len(labels):
            raise InvalidEMNISTDatasetError(
                f"{split_name} image/label count mismatch: {len(images)} != {len(labels)}"
            )
