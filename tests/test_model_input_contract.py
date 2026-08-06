import pytest

from app.preprocessing.model_input_contract import ModelInputContract


def test_contract_exposes_image_and_sample_shapes() -> None:
    contract = ModelInputContract(width=28, height=28, channels=1)

    assert contract.image_shape == (28, 28)
    assert contract.model_sample_shape == (28, 28, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("height", -1),
        ("channels", 3),
        ("background_value", -1),
        ("background_value", 256),
        ("normalization_divisor", 0.0),
    ],
)
def test_contract_rejects_invalid_values(field: str, value: int | float) -> None:
    arguments: dict[str, int | float] = {
        "width": 28,
        "height": 28,
        "channels": 1,
        "background_value": 0,
        "normalization_divisor": 255.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        ModelInputContract(**arguments)  # type: ignore[arg-type]
