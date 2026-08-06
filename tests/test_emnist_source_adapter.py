import numpy as np
import pytest

from app.preprocessing.emnist_source_adapter import EMNISTSourceAdapter
from app.preprocessing.exceptions import InvalidImageError


def asymmetric_sample() -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[2, 5] = 255
    image[6:20, 24] = 180
    return image


def test_no_transpose_preserves_asymmetric_coordinates_and_input() -> None:
    image = asymmetric_sample()
    original = image.copy()

    result = EMNISTSourceAdapter(transpose_images=False).adapt(image)

    assert np.array_equal(result, original)
    assert np.array_equal(image, original)
    assert result is not image


def test_transpose_swaps_axes_once() -> None:
    image = asymmetric_sample()

    result = EMNISTSourceAdapter(transpose_images=True).adapt(image)

    assert result[5, 2] == 255
    assert result[24, 6] == 180
    assert result[2, 5] == 0


def test_single_channel_input_is_squeezed() -> None:
    image = asymmetric_sample()[..., np.newaxis]

    result = EMNISTSourceAdapter(transpose_images=False).adapt(image)

    assert result.shape == (28, 28)
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((27, 28), dtype=np.uint8),
        np.zeros((28, 28, 3), dtype=np.uint8),
        np.zeros((28, 28), dtype=np.float32),
    ],
)
def test_adapter_rejects_invalid_shape_or_dtype(image: np.ndarray) -> None:
    with pytest.raises(InvalidImageError):
        EMNISTSourceAdapter().adapt(image)


def test_adapter_does_not_offer_binary_emnist_mode() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        EMNISTSourceAdapter(preserve_grayscale=False)
