import numpy as np
import pytest

from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.image_validator import ImageValidator


def test_validate_accepts_grayscale_uint8() -> None:
    image = np.zeros((20, 30), dtype=np.uint8)

    assert ImageValidator.validate(image) is image


def test_validate_accepts_bgr_uint8() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert ImageValidator.validate(image) is image


def test_validate_accepts_bgra_uint8() -> None:
    image = np.zeros((20, 30, 4), dtype=np.uint8)

    assert ImageValidator.validate(image) is image


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.array([], dtype=np.uint8),
        np.zeros((20, 30), dtype=np.float32),
        np.zeros((20,), dtype=np.uint8),
        np.zeros((20, 30, 5), dtype=np.uint8),
        np.array([object()], dtype=object),
    ],
)
def test_validate_rejects_invalid_images(image: object) -> None:
    with pytest.raises(InvalidImageError):
        ImageValidator.validate(image)
