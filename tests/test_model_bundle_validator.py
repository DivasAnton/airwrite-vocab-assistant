import pytest

from app.inference.exceptions import InvalidModelBundleError
from app.inference.model_bundle import ModelBundle
from app.inference.model_bundle_validator import ModelBundleValidator
from app.ml.labels import CHARACTER_LABELS


class FakeModel:
    def __init__(
        self,
        input_shape: tuple[object, ...] = (None, 28, 28, 1),
        output_shape: tuple[object, ...] = (None, 26),
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape


def contract() -> dict[str, object]:
    return {
        "output_width": 28,
        "output_height": 28,
        "channels": 1,
        "background": "black",
        "foreground": "light",
        "normalized_min": 0.0,
        "normalized_max": 1.0,
        "content_width": 20,
        "content_height": 20,
        "binary_threshold": 20,
        "crop_padding": 8,
        "min_foreground_pixels": 10,
        "invert_input": False,
        "center_of_mass": False,
    }


def bundle(model: FakeModel | None = None) -> ModelBundle:
    return ModelBundle(
        model=model or FakeModel(),
        labels=CHARACTER_LABELS,
        model_version="0.1.0",
        expected_input_shape=(28, 28, 1),
        num_classes=26,
        preprocessing_contract=contract(),
    )


def test_validator_accepts_matching_bundle() -> None:
    ModelBundleValidator().validate(bundle(), contract())


def test_validator_rejects_model_input_mismatch() -> None:
    with pytest.raises(InvalidModelBundleError, match="input shape"):
        ModelBundleValidator().validate(
            bundle(FakeModel(input_shape=(None, 32, 32, 1))), contract()
        )


def test_validator_rejects_model_output_mismatch() -> None:
    with pytest.raises(InvalidModelBundleError, match="output shape"):
        ModelBundleValidator().validate(bundle(FakeModel(output_shape=(None, 27))), contract())


def test_validator_rejects_preprocessing_mismatch() -> None:
    runtime = contract()
    runtime["background"] = "white"

    with pytest.raises(InvalidModelBundleError, match="incompatible"):
        ModelBundleValidator().validate(bundle(), runtime)
