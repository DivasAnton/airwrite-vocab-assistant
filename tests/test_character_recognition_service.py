import numpy as np

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.character_predictor import CharacterPredictor
from app.inference.character_recognition_service import CharacterRecognitionService
from app.inference.exceptions import CharacterPredictionError
from app.inference.model_bundle import ModelBundle
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_status import PredictionStatus
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS
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
        identity_labels=LETTER_IDENTITY_LABELS,
        lowercase_display_labels=LETTER_IDENTITY_LABELS,
        uppercase_display_labels=tuple(label.upper() for label in LETTER_IDENTITY_LABELS),
        model_version="1.0.0",
        task_type="letter_identity_classification",
        case_sensitive=False,
        case_source="user_selected_mode",
        expected_input_shape=(28, 28, 1),
        num_classes=26,
        preprocessing_contract={},
    )
    service = CharacterRecognitionService(
        HandwritingPreprocessor(),
        CharacterPredictor(bundle),
        PredictionPolicy(),
        CaseLabelResolver(
            bundle.identity_labels,
            bundle.lowercase_display_labels,
            bundle.uppercase_display_labels,
        ),
        log_latency=False,
    )
    return service, fake_model


def test_service_preprocesses_snapshot_without_mutating_it() -> None:
    service, model = make_service()
    snapshot = np.zeros((100, 100, 3), dtype=np.uint8)
    snapshot[20:80, 45:55] = 255
    original = snapshot.copy()

    result = service.recognize(
        snapshot,
        case_selection=CaseSelection(CharacterCaseMode.UPPERCASE, False),
    )

    assert result.status == PredictionStatus.ACCEPTED
    assert result.top_prediction is not None
    assert result.top_prediction.label == "A"
    assert result.top_prediction.identity == "a"
    assert result.case_was_inferred is False
    assert len(result.candidates) == 3
    assert model.calls == 1
    assert np.array_equal(snapshot, original)


def test_service_skips_empty_canvas_without_calling_model() -> None:
    service, model = make_service()

    result = service.recognize(
        np.zeros((100, 100, 3), dtype=np.uint8),
        case_selection=CaseSelection(CharacterCaseMode.LOWERCASE, False),
    )

    assert result.status == PredictionStatus.SKIPPED_EMPTY
    assert model.calls == 0


def test_service_converts_predictor_error_to_failed_result() -> None:
    service = CharacterRecognitionService(
        HandwritingPreprocessor(),
        FailingPredictor(),  # type: ignore[arg-type]
        PredictionPolicy(),
        CaseLabelResolver(
            LETTER_IDENTITY_LABELS,
            LETTER_IDENTITY_LABELS,
            tuple(label.upper() for label in LETTER_IDENTITY_LABELS),
        ),
        log_latency=False,
    )
    snapshot = np.zeros((100, 100, 3), dtype=np.uint8)
    snapshot[20:80, 45:55] = 255

    result = service.recognize(
        snapshot,
        case_selection=CaseSelection(CharacterCaseMode.LOWERCASE, False),
    )

    assert result.status == PredictionStatus.FAILED
    assert result.message == "Prediction failed"


def test_same_identity_is_rendered_without_changing_confidence_between_modes() -> None:
    service, model = make_service()
    snapshot = np.zeros((100, 100, 3), dtype=np.uint8)
    snapshot[20:80, 45:55] = 255

    lowercase = service.recognize(
        snapshot,
        case_selection=CaseSelection(CharacterCaseMode.LOWERCASE, False),
    )
    uppercase = service.recognize(
        snapshot,
        case_selection=CaseSelection(CharacterCaseMode.UPPERCASE, False),
    )

    assert lowercase.top_prediction is not None
    assert uppercase.top_prediction is not None
    assert lowercase.top_prediction.identity == uppercase.top_prediction.identity == "a"
    assert lowercase.top_prediction.rendered_character == "a"
    assert uppercase.top_prediction.rendered_character == "A"
    assert lowercase.top_prediction.confidence == uppercase.top_prediction.confidence
    assert lowercase.confidence_margin == uppercase.confidence_margin
    assert model.calls == 2
