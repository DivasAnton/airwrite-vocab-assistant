from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.ml.airwrite_identity_evaluator import (
    AirWriteIdentityEvaluator,
    AirWriteIdentitySample,
)


class StubPreprocessor:
    def process_file(self, path: Path) -> object:
        index = int(path.stem)
        image = np.zeros((28, 28), dtype=np.float32)
        image[0, 0] = index
        return SimpleNamespace(normalized_image=image)


class StubModel:
    def predict_on_batch(self, images: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((len(images), 26), dtype=np.float32)
        for row, image in enumerate(images):
            probabilities[row, int(image[0, 0, 0])] = 1.0
        return probabilities


def test_airwrite_evaluator_reports_accuracy_by_writing_style(tmp_path: Path) -> None:
    samples = [
        AirWriteIdentitySample(tmp_path / "0.png", "a", "uppercase", "s1"),
        AirWriteIdentitySample(tmp_path / "1.png", "b", "lowercase", "s2"),
    ]
    result = AirWriteIdentityEvaluator(
        preprocessor=StubPreprocessor(),
        images_preprocessed=False,
    ).evaluate(StubModel(), samples)
    assert result.test_accuracy == 1.0
    assert result.source_metrics["uppercase"]["accuracy"] == 1.0
    assert result.source_metrics["lowercase"]["accuracy"] == 1.0
