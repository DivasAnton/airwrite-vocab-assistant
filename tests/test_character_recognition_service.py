import numpy as np

from app.inference.character_predictor import CharacterPredictor
from app.inference.character_recognition_service import CharacterRecognitionService
from app.inference.exceptions import CharacterPredictionError
from app.inference.model_bundle import ModelBundle
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_status import PredictionStatus
from app.ml.labels import CHARACTER_LABELS
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor


class FakeModel:
    input_shape = (None, 28, 28, 1)
    output_shape = (None, 26)

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, model_input: np.ndarray, verbose: int) -> np.ndarray:
        self.calls += 1
        probabilities = np.full((1, 26), 0.01, dtype=np.float32)
        probabilities[0, 0] = 0.75
        return probabilities


class FailingPredictor:
    model_version = "0.1.0"

    def predict(self, _image: np.ndarray) -> None:
        raise CharacterPredictionError("failed")


def make_service(model: FakeModel | None = None) -> tuple[CharacterRecognitionService, FakeModel]:
    fake_model = model or FakeModel()
    bundle = ModelBundle(
        model=fake_model,
        labels=CHARACTER_LABELS,
        model_version="0.1.0",
        expected_input_shape=(28, 28, 1),
        num_classes=26,
        preprocessing_contract={},
    )
    service = CharacterRecognitionService(
        HandwritingPreprocessor(),
        CharacterPredictor(bundle),
        PredictionPolicy(),
        log_latency=False,
    )
    return service, fake_model


def test_service_preprocesses_snapshot_without_mutating_it() -> None:
    service, model = make_service()
    snapshot = np.zeros((100, 100, 3), dtype=np.uint8)
    snapshot[20:80, 45:55] = 255
    original = snapshot.copy()

    result = service.recognize(snapshot)

    assert result.status == PredictionStatus.ACCEPTED
    assert result.top_prediction is not None
    assert result.top_prediction.label == "A"
    assert len(result.candidates) == 3
    assert model.calls == 1
    assert np.array_equal(snapshot, original)


def test_service_skips_empty_canvas_without_calling_model() -> None:
    service, model = make_service()

    result = service.recognize(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.status == PredictionStatus.SKIPPED_EMPTY
    assert model.calls == 0


def test_service_converts_predictor_error_to_failed_result() -> None:
    service = CharacterRecognitionService(
        HandwritingPreprocessor(),
        FailingPredictor(),  # type: ignore[arg-type]
        PredictionPolicy(),
        log_latency=False,
    )
    snapshot = np.zeros((100, 100, 3), dtype=np.uint8)
    snapshot[20:80, 45:55] = 255

    result = service.recognize(snapshot)

    assert result.status == PredictionStatus.FAILED
    assert result.message == "Prediction failed"
