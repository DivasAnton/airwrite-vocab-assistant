import importlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.ml.dataset_manifest import DatasetManifestEntry, read_manifest
from app.ml.exceptions import DatasetLoadError, TrainingDependencyError
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor

BatchArrays = tuple[NDArray[np.float32], NDArray[np.int64]]
EntryBatch = tuple[list[DatasetManifestEntry], NDArray[np.float32], NDArray[np.int64]]


class TrainingDataLoader:
    def __init__(
        self,
        preprocessor: HandwritingPreprocessor | None = None,
        batch_size: int = 64,
        random_seed: int = 42,
        project_root: Path | None = None,
        images_are_preprocessed: bool = True,
        expected_shape: tuple[int, int] = (28, 28),
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
        self.preprocessor = preprocessor or HandwritingPreprocessor()
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.project_root = project_root.resolve() if project_root is not None else None
        self.images_are_preprocessed = images_are_preprocessed
        self.expected_shape = expected_shape

    def read_split(self, manifest_path: Path, split: str) -> list[DatasetManifestEntry]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        entries = [entry for entry in read_manifest(manifest_path) if entry.split == split]
        if not entries:
            raise DatasetLoadError(f"No entries found for split {split!r} in {manifest_path}")
        return entries

    def iter_batches(
        self, entries: list[DatasetManifestEntry], shuffle: bool = False
    ) -> Iterator[BatchArrays]:
        for _, images, labels in self.iter_entry_batches(entries, shuffle=shuffle):
            yield images, labels

    def iter_entry_batches(
        self, entries: list[DatasetManifestEntry], shuffle: bool = False
    ) -> Iterator[EntryBatch]:
        ordered_entries = list(entries)
        if shuffle:
            np.random.default_rng(self.random_seed).shuffle(ordered_entries)
        for start in range(0, len(ordered_entries), self.batch_size):
            batch_entries = ordered_entries[start : start + self.batch_size]
            images, labels = self.load_batch(batch_entries)
            yield batch_entries, images, labels

    def load_batch(self, entries: list[DatasetManifestEntry]) -> BatchArrays:
        if not entries:
            return (
                np.empty((0, *self.expected_shape, 1), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        images = np.stack([self._load_image(entry) for entry in entries]).astype(
            np.float32, copy=False
        )
        labels = np.asarray([entry.label_index for entry in entries], dtype=np.int64)
        return cast(NDArray[np.float32], images), cast(NDArray[np.int64], labels)

    def load_all(self, entries: list[DatasetManifestEntry]) -> BatchArrays:
        return self.load_batch(entries)

    def as_tensorflow_dataset(
        self,
        entries: list[DatasetManifestEntry],
        shuffle: bool = False,
    ) -> Any:
        tensorflow = self._require_tensorflow()
        output_signature = (
            tensorflow.TensorSpec(shape=(None, *self.expected_shape, 1), dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=(None,), dtype=tensorflow.int64),
        )
        dataset = tensorflow.data.Dataset.from_generator(
            lambda: self.iter_batches(entries, shuffle=False),
            output_signature=output_signature,
        )
        if shuffle:
            dataset = (
                dataset.unbatch()
                .shuffle(
                    buffer_size=len(entries),
                    seed=self.random_seed,
                    reshuffle_each_iteration=True,
                )
                .batch(self.batch_size)
            )
        steps = math.ceil(len(entries) / self.batch_size)

        dataset = dataset.apply(tensorflow.data.experimental.assert_cardinality(steps))
        return dataset.prefetch(tensorflow.data.AUTOTUNE)

    def _load_image(self, entry: DatasetManifestEntry) -> NDArray[np.float32]:
        image_path = self._resolve_path(entry.image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise DatasetLoadError(f"Could not read training image: {image_path}")

        if self.images_are_preprocessed:
            if image.ndim != 2:
                raise DatasetLoadError(
                    "Expected a grayscale preprocessed image, "
                    f"got shape {image.shape}: {image_path}"
                )
            if image.shape != self.expected_shape:
                raise DatasetLoadError(
                    f"Expected image shape {self.expected_shape}, got {image.shape}: {image_path}"
                )
            normalized = image.astype(np.float32) / 255.0
        else:
            result = self.preprocessor.process(image)
            normalized = result.normalized_image
            if normalized.shape != self.expected_shape:
                raise DatasetLoadError(
                    f"Preprocessor returned {normalized.shape}, expected {self.expected_shape}"
                )

        if not np.isfinite(normalized).all() or normalized.min() < 0.0 or normalized.max() > 1.0:
            raise DatasetLoadError(
                f"Normalized image is outside the finite 0-1 range: {image_path}"
            )
        return cast(NDArray[np.float32], normalized[..., np.newaxis])

    def _resolve_path(self, image_path: Path) -> Path:
        if image_path.is_absolute():
            return image_path
        if self.project_root is None:
            return image_path.resolve()
        return (self.project_root / image_path).resolve()

    @staticmethod
    def _require_tensorflow() -> Any:
        try:
            return importlib.import_module("tensorflow")
        except ImportError as error:
            raise TrainingDependencyError(
                "TensorFlow is required to build tf.data datasets. "
                "Install training dependencies before training."
            ) from error
