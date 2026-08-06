from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.inference.exceptions import CharacterPredictionError, InvalidModelInputError
from app.inference.identity_candidate import IdentityCandidate
from app.inference.model_bundle import ModelBundle


@dataclass(frozen=True)
class RawPrediction:
    candidates: tuple[IdentityCandidate, ...]
    inference_time_ms: float


class CharacterPredictor:
    def __init__(self, bundle: ModelBundle, top_k: int = 3) -> None:
        if not 0 < top_k <= len(bundle.identity_labels):
            raise ValueError("top_k must be between 1 and the number of labels")
        self._model = bundle.model
        self.identity_labels = bundle.identity_labels
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
            IdentityCandidate(
                identity=self.identity_labels[int(index)],
                class_index=int(index),
                confidence=float(probabilities[index]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        )
        return RawPrediction(candidates=candidates, inference_time_ms=elapsed_ms)

    def predict_batch(
        self,
        images: NDArray[np.float32],
        *,
        top_k: int | None = None,
        max_batch_size: int = 26,
    ) -> tuple[RawPrediction, ...]:
        if not isinstance(images, np.ndarray) or images.dtype != np.float32:
            raise InvalidModelInputError("Batch images must be a float32 NumPy array")
        expected_tail = (*self.expected_image_shape, 1)
        if images.ndim != 4 or images.shape[1:] != expected_tail:
            raise InvalidModelInputError(
                f"Model input batch must have shape (N, {expected_tail}), got {images.shape}"
            )
        if not 1 <= images.shape[0] <= max_batch_size:
            raise InvalidModelInputError(
                f"Batch size must be between 1 and {max_batch_size}, got {images.shape[0]}"
            )
        if (
            not np.all(np.isfinite(images))
            or float(images.min()) < 0.0
            or float(images.max()) > 1.0
        ):
            raise InvalidModelInputError("Batch values must be finite and between 0.0 and 1.0")
        selected_top_k = self.top_k if top_k is None else top_k
        if not 1 <= selected_top_k <= len(self.identity_labels):
            raise ValueError("top_k must be between 1 and the number of labels")

        started_at = perf_counter()
        try:
            output = self._model.predict(images, verbose=0)
        except Exception as error:
            raise CharacterPredictionError("Character model batch prediction failed") from error
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        probabilities = self._validate_batch_output(output, images.shape[0])
        predictions: list[RawPrediction] = []
        for row in probabilities:
            indices = np.argsort(row)[::-1][:selected_top_k]
            predictions.append(
                RawPrediction(
                    candidates=tuple(
                        IdentityCandidate(
                            identity=self.identity_labels[int(index)],
                            class_index=int(index),
                            confidence=float(row[index]),
                            rank=rank,
                        )
                        for rank, index in enumerate(indices, start=1)
                    ),
                    inference_time_ms=elapsed_ms,
                )
            )
        return tuple(predictions)

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
        expected_shape = (1, len(self.identity_labels))
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

    def _validate_batch_output(
        self,
        output: object,
        batch_size: int,
    ) -> NDArray[np.float32]:
        probabilities = np.asarray(output)
        expected_shape = (batch_size, len(self.identity_labels))
        if probabilities.shape != expected_shape:
            raise CharacterPredictionError(
                f"Model output must have shape {expected_shape}, got {probabilities.shape}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise CharacterPredictionError("Model output contains NaN or infinity")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise CharacterPredictionError("Model probabilities must be between 0.0 and 1.0")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-5):
            raise CharacterPredictionError("Every model output row must sum to 1.0")
        return cast(NDArray[np.float32], probabilities.astype(np.float32, copy=False))
