import importlib.util

import numpy as np
import pytest

from app.inference.character_predictor import CharacterPredictor
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import PROJECT_ROOT, settings

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None


@pytest.mark.skipif(not KERAS_AVAILABLE, reason="Keras is an optional runtime dependency")
def test_real_model_pipeline_is_deterministic() -> None:
    model_path = PROJECT_ROOT / "artifacts/models/character_recognizer.keras"
    if not model_path.is_file():
        pytest.skip("Trained Sprint 9 model artifact is not available")

    bundle = ModelBundleLoader(
        model_path=model_path,
        labels_path=PROJECT_ROOT / "artifacts/labels/labels.json",
        metadata_path=PROJECT_ROOT / "artifacts/metadata/model_metadata.json",
        preprocessing_config_path=PROJECT_ROOT / "artifacts/metadata/preprocessing_config.json",
    ).load()
    ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())
    predictor = CharacterPredictor(bundle, top_k=3)
    normalized = (
        HandwritingPreprocessor().process(synthetic_letter_like_snapshot()).normalized_image
    )

    first = predictor.predict(normalized)
    second = predictor.predict(normalized)

    assert len(first.candidates) == 3
    assert [item.label for item in first.candidates] == [item.label for item in second.candidates]
    assert [item.confidence for item in first.candidates] == pytest.approx(
        [item.confidence for item in second.candidates],
        abs=1e-7,
    )
    result = PredictionPolicy().apply(first, bundle.model_version)
    assert result.model_version == "0.1.0"
    assert all(0.0 <= candidate.confidence <= 1.0 for candidate in result.candidates)


def synthetic_letter_like_snapshot() -> np.ndarray:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:100, 52:60] = 255
    image[20:28, 52:90] = 255
    image[55:63, 52:82] = 255
    return image
