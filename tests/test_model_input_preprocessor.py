import numpy as np
import pytest

from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor
from app.preprocessing.preprocessing_source import PreprocessingSource


def test_common_preprocessor_normalizes_without_mutating_input() -> None:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[5:20, 8:18] = 200
    original = image.copy()

    result = ModelInputPreprocessor().process(
        image,
        source=PreprocessingSource.EMNIST_LETTERS,
        orientation_transform="transpose",
    )

    assert np.array_equal(image, original)
    assert result.processed_image.dtype == np.uint8
    assert result.normalized_image.dtype == np.float32
    assert result.normalized_image.shape == (28, 28)
    assert result.normalized_image.min() >= 0.0
    assert result.normalized_image.max() <= 1.0
    assert np.isfinite(result.normalized_image).all()
    assert result.source == PreprocessingSource.EMNIST_LETTERS
    assert result.orientation_transform == "transpose"


def test_common_preprocessor_accepts_empty_contract_image() -> None:
    result = ModelInputPreprocessor().process(
        np.zeros((28, 28), dtype=np.uint8),
        source=PreprocessingSource.EMNIST_LETTERS,
    )

    assert result.foreground_pixel_count == 0
    assert result.foreground_ratio == 0.0
    assert result.bounding_box is None


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((27, 28), dtype=np.uint8),
        np.zeros((28, 28, 1), dtype=np.uint8),
        np.zeros((28, 28), dtype=np.float32),
    ],
)
def test_common_preprocessor_rejects_wrong_shape_or_dtype(image: np.ndarray) -> None:
    with pytest.raises(InvalidImageError):
        ModelInputPreprocessor().process(
            image,
            source=PreprocessingSource.EMNIST_LETTERS,
        )


def test_common_preprocessor_rejects_light_background() -> None:
    image = np.full((28, 28), 255, dtype=np.uint8)
    image[8:20, 10:18] = 0

    with pytest.raises(InvalidImageError, match="light background"):
        ModelInputPreprocessor(ModelInputContract()).process(
            image,
            source=PreprocessingSource.EMNIST_LETTERS,
        )
