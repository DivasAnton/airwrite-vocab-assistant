import pytest

from app.inference.identity_candidate import IdentityCandidate
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_status import PredictionStatus


def candidates(top_one: float, top_two: float) -> tuple[IdentityCandidate, ...]:
    return (
        IdentityCandidate("a", 0, top_one, 1),
        IdentityCandidate("b", 1, top_two, 2),
    )


def test_policy_accepts_high_confidence_and_margin() -> None:
    status, margin = PredictionPolicy(0.60, 0.15).evaluate(candidates(0.80, 0.10))

    assert status == PredictionStatus.ACCEPTED
    assert margin == pytest.approx(0.70)


def test_policy_rejects_low_confidence() -> None:
    status, _margin = PredictionPolicy(0.60, 0.15).evaluate(candidates(0.49, 0.20))
    assert status == PredictionStatus.UNCERTAIN


def test_policy_rejects_low_margin() -> None:
    status, _margin = PredictionPolicy(0.60, 0.15).evaluate(candidates(0.65, 0.61))
    assert status == PredictionStatus.UNCERTAIN


def test_policy_accepts_values_equal_to_thresholds() -> None:
    status, _margin = PredictionPolicy(0.60, 0.15).evaluate(candidates(0.60, 0.45))
    assert status == PredictionStatus.ACCEPTED
