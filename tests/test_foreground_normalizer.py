import numpy as np

from app.preprocessing.foreground_normalizer import ForegroundNormalizer


def test_normalize_keeps_grayscale_shape() -> None:
    image = np.zeros((20, 30), dtype=np.uint8)
    image[5:10, 8:16] = 255
    normalizer = ForegroundNormalizer(binary_threshold=20)

    result = normalizer.normalize(image)

    assert result.grayscale.shape == (20, 30)
    assert result.binary.shape == (20, 30)
    assert result.binary[6, 9] == 255
    assert result.binary[0, 0] == 0


def test_normalize_converts_bgr_to_grayscale() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:10, 8:16] = (255, 255, 255)
    normalizer = ForegroundNormalizer(binary_threshold=20)

    result = normalizer.normalize(image)

    assert result.grayscale.ndim == 2
    assert result.binary[6, 9] == 255


def test_normalize_converts_bgra_to_grayscale() -> None:
    image = np.zeros((20, 30, 4), dtype=np.uint8)
    image[5:10, 8:16] = (255, 255, 255, 255)
    normalizer = ForegroundNormalizer(binary_threshold=20)

    result = normalizer.normalize(image)

    assert result.grayscale.ndim == 2
    assert result.binary[6, 9] == 255


def test_normalize_can_invert_white_background_input() -> None:
    image = np.full((20, 30), 255, dtype=np.uint8)
    image[5:10, 8:16] = 0
    normalizer = ForegroundNormalizer(binary_threshold=20, invert_input=True)

    result = normalizer.normalize(image)

    assert result.binary[6, 9] == 255
    assert result.binary[0, 0] == 0


def test_threshold_255_produces_empty_binary() -> None:
    image = np.full((20, 30), 255, dtype=np.uint8)
    normalizer = ForegroundNormalizer(binary_threshold=255)

    result = normalizer.normalize(image)

    assert np.count_nonzero(result.binary) == 0
