import importlib.util

import numpy as np
import pytest

from app.inference.character_predictor import CharacterPredictor
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import identity_model_settings, settings

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None


@pytest.mark.skipif(not KERAS_AVAILABLE, reason="Keras is an optional runtime dependency")
def test_real_model_pipeline_is_deterministic() -> None:
    model_path = identity_model_settings.model_path
    if not model_path.is_file():
        pytest.skip("Trained Sprint 9 model artifact is not available")

    bundle = ModelBundleLoader(
        model_path=model_path,
        identity_labels_path=identity_model_settings.identity_labels_path,
        lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
        uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
        metadata_path=identity_model_settings.metadata_path,
        preprocessing_config_path=identity_model_settings.preprocessing_config_path,
    ).load()
    ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())
    predictor = CharacterPredictor(bundle, top_k=3)
    normalized = (
        HandwritingPreprocessor().process(synthetic_letter_like_snapshot()).normalized_image
    )

    first = predictor.predict(normalized)
    second = predictor.predict(normalized)

    assert len(first.candidates) == 3
    assert [item.identity for item in first.candidates] == [
        item.identity for item in second.candidates
    ]
    assert [item.confidence for item in first.candidates] == pytest.approx(
        [item.confidence for item in second.candidates],
        abs=1e-7,
    )
    status, margin = PredictionPolicy().evaluate(first.candidates)
    assert status.value in {"ACCEPTED", "UNCERTAIN"}
    assert 0.0 <= margin <= 1.0
    assert all(0.0 <= candidate.confidence <= 1.0 for candidate in first.candidates)


def synthetic_letter_like_snapshot() -> np.ndarray:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:100, 52:60] = 255
    image[20:28, 52:90] = 255
    image[55:63, 52:82] = 255
    return image
