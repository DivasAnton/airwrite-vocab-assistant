import numpy as np

from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.inference.prediction_status_renderer import PredictionStatusRenderer


def result(status: PredictionStatus) -> PredictionResult:
    candidates = (
        PredictionCandidate("F", 5, 0.42, 1),
        PredictionCandidate("P", 15, 0.39, 2),
        PredictionCandidate("E", 4, 0.19, 3),
    )
    return PredictionResult(
        status,
        candidates[0]
        if status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN}
        else None,
        candidates if status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN} else (),
        0.03 if status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN} else None,
        12.0 if status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN} else None,
        "0.1.0",
        "Model unavailable" if status == PredictionStatus.FAILED else status.value,
    )


def test_renderer_formats_accepted_prediction() -> None:
    lines = PredictionStatusRenderer().format_lines(result(PredictionStatus.ACCEPTED))
    assert lines[:2] == ["Prediction: F", "Confidence: 42.0%"]
    assert lines[-1] == "Model: v0.1.0"


def test_renderer_formats_uncertain_top_three() -> None:
    lines = PredictionStatusRenderer().format_lines(result(PredictionStatus.UNCERTAIN))
    assert lines[:4] == [
        "Uncertain prediction",
        "1. F - 42.0%",
        "2. P - 39.0%",
        "3. E - 19.0%",
    ]


def test_renderer_hides_result_after_timeout_without_mutating_frame() -> None:
    renderer = PredictionStatusRenderer(status_display_ms=4000)
    frame = np.zeros((200, 500, 3), dtype=np.uint8)

    output = renderer.render(frame, result(PredictionStatus.ACCEPTED), 1000, 5001)

    assert np.array_equal(output, frame)
    assert output is not frame


def test_renderer_uses_safe_model_unavailable_message() -> None:
    lines = PredictionStatusRenderer().format_lines(result(PredictionStatus.FAILED))
    assert lines[0] == "Model unavailable"
