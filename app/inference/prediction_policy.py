from math import isclose

from app.inference.character_predictor import RawPrediction
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus


class PredictionPolicy:
    def __init__(self, min_confidence: float = 0.60, min_margin: float = 0.15) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0.0 and 1.0")
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def evaluate(
        self, candidates: tuple[PredictionCandidate, ...]
    ) -> tuple[PredictionStatus, float]:
        if not candidates:
            raise ValueError("At least one prediction candidate is required")
        top_confidence = candidates[0].confidence
        margin = (
            top_confidence - candidates[1].confidence if len(candidates) > 1 else top_confidence
        )
        status = (
            PredictionStatus.ACCEPTED
            if top_confidence >= self.min_confidence
            and (margin >= self.min_margin or isclose(margin, self.min_margin, abs_tol=1e-12))
            else PredictionStatus.UNCERTAIN
        )
        return status, margin

    def apply(self, prediction: RawPrediction, model_version: str) -> PredictionResult:
        status, margin = self.evaluate(prediction.candidates)
        top_prediction = prediction.candidates[0]
        message = (
            f"Prediction accepted: {top_prediction.label}"
            if status == PredictionStatus.ACCEPTED
            else "Prediction is uncertain"
        )
        return PredictionResult(
            status=status,
            top_prediction=top_prediction,
            candidates=prediction.candidates,
            confidence_margin=margin,
            inference_time_ms=prediction.inference_time_ms,
            model_version=model_version,
            message=message,
        )
