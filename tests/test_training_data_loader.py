from pathlib import Path

import cv2
import numpy as np

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.training_data_loader import TrainingDataLoader


class FailingPreprocessor:
    def process(self, image: object) -> object:
        raise AssertionError("Already-preprocessed custom data must not be processed twice")


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[6:22, 10:18] = value
    assert cv2.imwrite(str(path), image)


def test_loader_returns_model_ready_batch_without_double_preprocessing(tmp_path: Path) -> None:
    first_path = tmp_path / "A" / "a.png"
    second_path = tmp_path / "B" / "b.png"
    write_image(first_path, 255)
    write_image(second_path, 128)
    entries = [
        DatasetManifestEntry(first_path, "A", 0, "airwrite", split="train"),
        DatasetManifestEntry(second_path, "B", 1, "airwrite", split="train"),
    ]
    loader = TrainingDataLoader(
        preprocessor=FailingPreprocessor(),  # type: ignore[arg-type]
        batch_size=2,
        images_are_preprocessed=True,
    )

    images, labels = loader.load_batch(entries)

    assert images.shape == (2, 28, 28, 1)
    assert labels.shape == (2,)
    assert images.dtype == np.float32
    assert labels.dtype == np.int64
    assert float(images.min()) == 0.0
    assert float(images.max()) == 1.0
    assert labels.tolist() == [0, 1]


def test_loader_shuffle_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    entries: list[DatasetManifestEntry] = []
    for index in range(6):
        path = tmp_path / "A" / f"{index}.png"
        write_image(path, 255)
        entries.append(DatasetManifestEntry(path, "A", index, "airwrite"))

    first_loader = TrainingDataLoader(batch_size=6, random_seed=7)
    second_loader = TrainingDataLoader(batch_size=6, random_seed=7)
    _, first_labels = next(first_loader.iter_batches(entries, shuffle=True))
    _, second_labels = next(second_loader.iter_batches(entries, shuffle=True))

    assert first_labels.tolist() == second_labels.tolist()
