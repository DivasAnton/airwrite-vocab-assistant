import pytest

from app.inference.character_predictor import RawPrediction
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_status import PredictionStatus


def prediction(top_one: float, top_two: float) -> RawPrediction:
    return RawPrediction(
        candidates=(
            PredictionCandidate("A", 0, top_one, 1),
            PredictionCandidate("B", 1, top_two, 2),
        ),
        inference_time_ms=10.0,
    )


def test_policy_accepts_high_confidence_and_margin() -> None:
    result = PredictionPolicy(0.60, 0.15).apply(prediction(0.80, 0.10), "0.1.0")

    assert result.status == PredictionStatus.ACCEPTED
    assert result.confidence_margin == pytest.approx(0.70)


def test_policy_rejects_low_confidence() -> None:
    result = PredictionPolicy(0.60, 0.15).apply(prediction(0.49, 0.20), "0.1.0")
    assert result.status == PredictionStatus.UNCERTAIN


def test_policy_rejects_low_margin() -> None:
    result = PredictionPolicy(0.60, 0.15).apply(prediction(0.65, 0.61), "0.1.0")
    assert result.status == PredictionStatus.UNCERTAIN


def test_policy_accepts_values_equal_to_thresholds() -> None:
    result = PredictionPolicy(0.60, 0.15).apply(prediction(0.60, 0.45), "0.1.0")
    assert result.status == PredictionStatus.ACCEPTED
