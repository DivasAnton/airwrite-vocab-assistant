import pytest

from app.inference.exceptions import InvalidModelBundleError
from app.inference.model_bundle import ModelBundle
from app.inference.model_bundle_validator import ModelBundleValidator
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


class FakeModel:
    def __init__(
        self,
        input_shape: tuple[object, ...] = (None, 28, 28, 1),
        output_shape: tuple[object, ...] = (None, 26),
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape


def runtime_contract() -> dict[str, object]:
    return {
        "output_width": 28,
        "output_height": 28,
        "channels": 1,
        "background": "black",
        "foreground": "light",
        "normalization_divisor": 255.0,
        "orientation_transform": "none",
        "invert_input": False,
    }


def artifact_contract() -> dict[str, object]:
    return {
        "input_width": 28,
        "input_height": 28,
        "channels": 1,
        "orientation_transform": "transpose",
        "intensity_inversion": False,
        "normalization_divisor": 255.0,
        "background": "black",
        "foreground": "light",
    }


def bundle(model: FakeModel | None = None) -> ModelBundle:
    return ModelBundle(
        model=model or FakeModel(),
        identity_labels=LETTER_IDENTITY_LABELS,
        lowercase_display_labels=LETTER_IDENTITY_LABELS,
        uppercase_display_labels=tuple(label.upper() for label in LETTER_IDENTITY_LABELS),
        model_version="1.0.0",
        task_type="letter_identity_classification",
        case_sensitive=False,
        case_source="user_selected_mode",
        expected_input_shape=(28, 28, 1),
        num_classes=26,
        preprocessing_contract=artifact_contract(),
    )


def test_validator_accepts_matching_bundle() -> None:
    ModelBundleValidator().validate(bundle(), runtime_contract())


def test_validator_rejects_model_input_mismatch() -> None:
    with pytest.raises(InvalidModelBundleError, match="input shape"):
        ModelBundleValidator().validate(
            bundle(FakeModel(input_shape=(None, 32, 32, 1))), runtime_contract()
        )


def test_validator_rejects_model_output_mismatch() -> None:
    with pytest.raises(InvalidModelBundleError, match="output shape"):
        ModelBundleValidator().validate(
            bundle(FakeModel(output_shape=(None, 27))), runtime_contract()
        )


def test_validator_rejects_preprocessing_mismatch() -> None:
    runtime = runtime_contract()
    runtime["background"] = "white"

    with pytest.raises(InvalidModelBundleError, match="incompatible"):
        ModelBundleValidator().validate(bundle(), runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "uppercase_character_classification"),
        ("case_sensitive", True),
        ("case_source", "model_inferred"),
    ],
)
def test_validator_rejects_identity_metadata_mismatch(field: str, value: object) -> None:
    invalid = bundle()
    object.__setattr__(invalid, field, value)

    with pytest.raises(InvalidModelBundleError):
        ModelBundleValidator().validate(invalid, runtime_contract())


def test_validator_rejects_misaligned_uppercase_mapping() -> None:
    invalid = bundle()
    object.__setattr__(
        invalid,
        "uppercase_display_labels",
        tuple(reversed(invalid.uppercase_display_labels)),
    )

    with pytest.raises(InvalidModelBundleError, match="uppercase display"):
        ModelBundleValidator().validate(invalid, runtime_contract())
