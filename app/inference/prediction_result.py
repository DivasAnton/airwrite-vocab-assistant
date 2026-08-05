from dataclasses import dataclass

from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_status import PredictionStatus


@dataclass(frozen=True)
class PredictionResult:
    status: PredictionStatus
    top_prediction: PredictionCandidate | None
    candidates: tuple[PredictionCandidate, ...]
    confidence_margin: float | None
    inference_time_ms: float | None
    model_version: str | None
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("message must not be empty")
        if self.candidates and self.top_prediction != self.candidates[0]:
            raise ValueError("top_prediction must be the first candidate")
        if self.status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN}:
            if self.top_prediction is None or not self.candidates:
                raise ValueError("prediction candidates are required for a prediction result")
            if self.confidence_margin is None:
                raise ValueError("confidence_margin is required for a prediction result")
        if self.inference_time_ms is not None and self.inference_time_ms < 0.0:
            raise ValueError("inference_time_ms must be greater than or equal to 0")
