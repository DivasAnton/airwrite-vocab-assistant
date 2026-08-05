import importlib.util

import pytest

from app.ml.character_model_builder import AugmentationConfig, CharacterModelBuilder
from app.ml.exceptions import TrainingDependencyError


def test_augmentation_config_rejects_invalid_factor() -> None:
    with pytest.raises(ValueError):
        AugmentationConfig(rotation_factor=1.0).validate()


def test_builder_reports_missing_tensorflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_tensorflow(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(
        "app.ml.character_model_builder.importlib.import_module", missing_tensorflow
    )

    with pytest.raises(TrainingDependencyError):
        CharacterModelBuilder().build()


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed"
)
def test_builder_creates_expected_input_and_output_shapes() -> None:
    model = CharacterModelBuilder(augmentation_config=AugmentationConfig(enabled=False)).build()

    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 26)
    assert model.output_shape[-1] == 26
    assert model.layers[-1].activation.__name__ == "softmax"
