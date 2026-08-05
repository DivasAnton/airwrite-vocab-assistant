from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.inference.exceptions import CharacterPredictionError, InvalidModelInputError
from app.inference.model_bundle import ModelBundle
from app.inference.prediction_candidate import PredictionCandidate


@dataclass(frozen=True)
class RawPrediction:
    candidates: tuple[PredictionCandidate, ...]
    inference_time_ms: float


class CharacterPredictor:
    def __init__(self, bundle: ModelBundle, top_k: int = 3) -> None:
        if not 0 < top_k <= len(bundle.labels):
            raise ValueError("top_k must be between 1 and the number of labels")
        self._model = bundle.model
        self.labels = bundle.labels
        self.model_version = bundle.model_version
        self.expected_image_shape = bundle.expected_input_shape[:2]
        self.top_k = top_k

    def predict(self, normalized_image: NDArray[np.float32]) -> RawPrediction:
        self._validate_input(normalized_image)
        model_input = normalized_image[np.newaxis, ..., np.newaxis]
        expected_batch_shape = (1, *self.expected_image_shape, 1)
        if model_input.shape != expected_batch_shape:
            raise InvalidModelInputError(
                f"Model input batch must have shape {expected_batch_shape}, got {model_input.shape}"
            )

        started_at = perf_counter()
        try:
            output = self._model.predict(model_input, verbose=0)
        except Exception as error:
            raise CharacterPredictionError("Character model prediction failed") from error
        elapsed_ms = (perf_counter() - started_at) * 1000.0

        probabilities = self._validate_output(output)
        indices = np.argsort(probabilities)[::-1][: self.top_k]
        candidates = tuple(
            PredictionCandidate(
                label=self.labels[int(index)],
                class_index=int(index),
                confidence=float(probabilities[index]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        )
        return RawPrediction(candidates=candidates, inference_time_ms=elapsed_ms)

    def _validate_input(self, image: object) -> None:
        if not isinstance(image, np.ndarray):
            raise InvalidModelInputError("Normalized image must be a NumPy array")
        if image.shape != self.expected_image_shape:
            raise InvalidModelInputError(
                f"Normalized image must have shape {self.expected_image_shape}, got {image.shape}"
            )
        if image.dtype != np.float32:
            raise InvalidModelInputError("Normalized image must use float32 dtype")
        if not np.all(np.isfinite(image)):
            raise InvalidModelInputError("Normalized image must contain only finite values")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise InvalidModelInputError("Normalized image values must be between 0.0 and 1.0")

    def _validate_output(self, output: object) -> NDArray[np.float32]:
        probabilities = np.asarray(output)
        expected_shape = (1, len(self.labels))
        if probabilities.shape != expected_shape:
            raise CharacterPredictionError(
                f"Model output must have shape {expected_shape}, got {probabilities.shape}"
            )
        sample = probabilities[0]
        if not np.all(np.isfinite(sample)):
            raise CharacterPredictionError("Model output contains NaN or infinity")
        if np.any(sample < 0.0) or np.any(sample > 1.0):
            raise CharacterPredictionError("Model probabilities must be between 0.0 and 1.0")
        if not np.isclose(float(sample.sum()), 1.0, rtol=1e-4, atol=1e-5):
            raise CharacterPredictionError("Model softmax probabilities must sum to 1.0")
        return cast(NDArray[np.float32], sample.astype(np.float32, copy=False))
