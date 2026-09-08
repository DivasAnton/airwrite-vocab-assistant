from time import perf_counter

from app.inference.character_predictor import CharacterPredictor, RawPrediction
from app.inference.identity_candidate import IdentityCandidate


class EnsembleCharacterPredictor:
    """Blend custom AirWrite probabilities with EMNIST as a weak fallback signal."""

    def __init__(
        self,
        custom: CharacterPredictor,
        emnist: CharacterPredictor,
        custom_weight: float = 0.80,
        top_k: int = 3,
    ) -> None:
        if not 0.0 < custom_weight < 1.0:
            raise ValueError("custom_weight must be between 0 and 1")
        if custom.identity_labels != emnist.identity_labels:
            raise ValueError("Ensemble models must use the same identity label order")
        if not 0 < top_k <= len(custom.identity_labels):
            raise ValueError("top_k must be between 1 and the number of labels")
        self.custom = custom
        self.emnist = emnist
        self.identity_labels = custom.identity_labels
        self.expected_image_shape = custom.expected_image_shape
        self.model_version = f"custom:{custom.model_version}+emnist:{emnist.model_version}"
        self.custom_weight = custom_weight
        self.top_k = top_k

    def predict(self, normalized_image):
        started = perf_counter()
        custom = self.custom.predict(normalized_image)
        emnist = self.emnist.predict(normalized_image)
        return self._blend(custom, emnist, (perf_counter() - started) * 1000.0)

    def predict_batch(self, images, *, top_k=None, max_batch_size=26):
        custom_predictions = self.custom.predict_batch(
            images, top_k=len(self.identity_labels), max_batch_size=max_batch_size
        )
        emnist_predictions = self.emnist.predict_batch(
            images, top_k=len(self.identity_labels), max_batch_size=max_batch_size
        )
        selected_top_k = self.top_k if top_k is None else top_k
        return tuple(
            self._blend(custom, emnist, custom.inference_time_ms + emnist.inference_time_ms, selected_top_k)
            for custom, emnist in zip(custom_predictions, emnist_predictions, strict=True)
        )

    def _blend(self, custom, emnist, elapsed_ms, top_k=None):
        selected_top_k = self.top_k if top_k is None else top_k
        custom_scores = {candidate.class_index: candidate.confidence for candidate in custom.candidates}
        emnist_scores = {candidate.class_index: candidate.confidence for candidate in emnist.candidates}
        scores = {
            index: self.custom_weight * custom_scores.get(index, 0.0)
            + (1.0 - self.custom_weight) * emnist_scores.get(index, 0.0)
            for index in range(len(self.identity_labels))
        }
        indices = sorted(scores, key=scores.get, reverse=True)[:selected_top_k]
        return RawPrediction(
            candidates=tuple(
                IdentityCandidate(
                    identity=self.identity_labels[index],
                    class_index=index,
                    confidence=float(scores[index]),
                    rank=rank,
                )
                for rank, index in enumerate(indices, start=1)
            ),
            inference_time_ms=elapsed_ms,
        )
