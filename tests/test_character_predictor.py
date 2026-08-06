import numpy as np
import pytest

from app.inference.character_predictor import CharacterPredictor
from app.inference.exceptions import CharacterPredictionError, InvalidModelInputError
from app.inference.model_bundle import ModelBundle
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


class FakeModel:
    input_shape = (None, 28, 28, 1)
    output_shape = (None, 26)

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.received_input: np.ndarray | None = None

    def predict(self, model_input: np.ndarray, verbose: int) -> np.ndarray:
        assert verbose == 0
        self.received_input = model_input.copy()
        return self.output


def make_predictor(output: np.ndarray) -> tuple[CharacterPredictor, FakeModel]:
    model = FakeModel(output)
    model_bundle = ModelBundle(
        model=model,
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
    return CharacterPredictor(model_bundle, top_k=3), model


def probabilities() -> np.ndarray:
    values = np.full(26, 0.01, dtype=np.float32)
    values[0] = 0.50
    values[3] = 0.16
    values[5] = 0.11
    return values[np.newaxis, :]


def test_predictor_builds_batch_and_returns_sorted_top_three() -> None:
    predictor, model = make_predictor(probabilities())
    image = np.zeros((28, 28), dtype=np.float32)
    original = image.copy()

    prediction = predictor.predict(image)

    assert [candidate.identity for candidate in prediction.candidates] == ["a", "d", "f"]
    assert [candidate.rank for candidate in prediction.candidates] == [1, 2, 3]
    assert prediction.candidates[0].confidence == pytest.approx(0.50)
    assert model.received_input is not None
    assert model.received_input.shape == (1, 28, 28, 1)
    assert model.received_input.dtype == np.float32
    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((28, 28), dtype=np.uint8),
        np.full((28, 28), np.nan, dtype=np.float32),
        np.full((28, 28), 1.1, dtype=np.float32),
    ],
)
def test_predictor_rejects_invalid_input(image: np.ndarray) -> None:
    predictor, _model = make_predictor(probabilities())

    with pytest.raises(InvalidModelInputError):
        predictor.predict(image)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((26,), dtype=np.float32),
        np.full((1, 26), np.nan, dtype=np.float32),
        np.full((1, 26), np.inf, dtype=np.float32),
        np.full((1, 26), 0.01, dtype=np.float32),
    ],
)
def test_predictor_rejects_invalid_model_output(output: np.ndarray) -> None:
    predictor, _model = make_predictor(output)

    with pytest.raises(CharacterPredictionError):
        predictor.predict(np.zeros((28, 28), dtype=np.float32))
