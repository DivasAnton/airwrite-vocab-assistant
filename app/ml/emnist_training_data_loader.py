import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.ml.emnist_letters_dataset import EMNISTDatasetSplit
from app.ml.exceptions import TrainingDependencyError
from app.preprocessing.emnist_source_adapter import EMNISTSourceAdapter
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor


@dataclass(frozen=True)
class EMNISTDatasetBundle:
    train: Any
    validation: Any
    test: Any


class EMNISTTrainingDataLoader:
    def __init__(
        self,
        batch_size: int = 128,
        shuffle_buffer: int = 10000,
        random_seed: int = 42,
        cache: bool = False,
        prefetch: bool = True,
        adapter: EMNISTSourceAdapter | None = None,
        preprocessor: ModelInputPreprocessor | None = None,
    ) -> None:
        if batch_size <= 0 or shuffle_buffer <= 0:
            raise ValueError("batch_size and shuffle_buffer must be greater than 0")
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.random_seed = random_seed
        self.cache = cache
        self.prefetch = prefetch
        self.adapter = adapter or EMNISTSourceAdapter()
        self.preprocessor = preprocessor or ModelInputPreprocessor(validate_dark_background=False)

    def build_bundle(
        self,
        official_train: EMNISTDatasetSplit,
        official_test: EMNISTDatasetSplit,
        train_indices: NDArray[np.int64],
        validation_indices: NDArray[np.int64],
    ) -> EMNISTDatasetBundle:
        return EMNISTDatasetBundle(
            train=self.build_dataset(
                official_train.images[train_indices],
                official_train.labels[train_indices],
                training=True,
            ),
            validation=self.build_dataset(
                official_train.images[validation_indices],
                official_train.labels[validation_indices],
                training=False,
            ),
            test=self.build_dataset(official_test.images, official_test.labels, training=False),
        )

    def build_dataset(
        self,
        raw_images: NDArray[np.uint8],
        labels: NDArray[np.int64],
        *,
        training: bool,
    ) -> Any:
        if len(raw_images) != len(labels) or len(labels) == 0:
            raise ValueError("Images and labels must have the same non-zero sample count")
        tensorflow = self._require_tensorflow()
        prepared = self.adapter.adapt_batch(raw_images)
        normalized = self.preprocessor.normalize_batch(prepared)[..., np.newaxis]
        dataset = tensorflow.data.Dataset.from_tensor_slices((normalized, labels))
        if self.cache:
            dataset = dataset.cache()
        if training:
            dataset = dataset.shuffle(
                min(self.shuffle_buffer, len(labels)),
                seed=self.random_seed,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        if self.prefetch:
            dataset = dataset.prefetch(tensorflow.data.AUTOTUNE)
        return dataset

    @staticmethod
    def _require_tensorflow() -> Any:
        try:
            return importlib.import_module("tensorflow")
        except ImportError as error:
            raise TrainingDependencyError(
                "TensorFlow is required to build EMNIST training datasets"
            ) from error
