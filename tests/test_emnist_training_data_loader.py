import importlib.util

import numpy as np
import pytest

from app.ml.emnist_training_data_loader import EMNISTTrainingDataLoader


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed"
)
def test_training_loader_outputs_normalized_finite_batches() -> None:
    images = np.zeros((5, 28, 28), dtype=np.uint8)
    images[:, 3, 9] = 255
    labels = np.arange(5, dtype=np.int64)
    loader = EMNISTTrainingDataLoader(batch_size=3, prefetch=False)
    dataset = loader.build_dataset(images, labels, training=False)
    batches = list(dataset.as_numpy_iterator())
    assert sum(len(batch_labels) for _, batch_labels in batches) == 5
    batch_images, batch_labels = batches[0]
    assert batch_images.shape == (3, 28, 28, 1)
    assert batch_images.dtype == np.float32
    assert batch_labels.shape == (3,)
    assert float(batch_images.min()) == 0.0
    assert float(batch_images.max()) == 1.0
    assert batch_images[0, 9, 3, 0] == 1.0
