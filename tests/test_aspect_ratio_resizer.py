import numpy as np
import pytest

from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.exceptions import InvalidPreprocessingConfigError


def foreground_box(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(image > 0)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def test_square_image_is_resized_to_content_size() -> None:
    cropped = np.full((10, 10), 255, dtype=np.uint8)
    resizer = AspectRatioResizer(28, 28, 20, 20)

    output = resizer.resize_and_center(cropped)

    assert output.shape == (28, 28)
    assert foreground_box(output) == (4, 4, 24, 24)


def test_tall_image_keeps_aspect_ratio() -> None:
    cropped = np.full((20, 10), 255, dtype=np.uint8)
    resizer = AspectRatioResizer(28, 28, 20, 20)

    output = resizer.resize_and_center(cropped)

    x_start, y_start, x_end, y_end = foreground_box(output)
    assert x_end - x_start == 10
    assert y_end - y_start == 20
    assert y_start == 4


def test_wide_image_keeps_aspect_ratio() -> None:
    cropped = np.full((10, 20), 255, dtype=np.uint8)
    resizer = AspectRatioResizer(28, 28, 20, 20)

    output = resizer.resize_and_center(cropped)

    x_start, y_start, x_end, y_end = foreground_box(output)
    assert x_end - x_start == 20
    assert y_end - y_start == 10
    assert x_start == 4


def test_output_shape_is_stable_for_small_input() -> None:
    cropped = np.full((1, 2), 255, dtype=np.uint8)
    resizer = AspectRatioResizer(28, 28, 20, 20)

    output = resizer.resize_and_center(cropped)

    assert output.shape == (28, 28)
    assert np.count_nonzero(output) > 0


def test_center_of_mass_option_keeps_output_shape() -> None:
    cropped = np.zeros((20, 20), dtype=np.uint8)
    cropped[12:20, 0:10] = 255
    resizer = AspectRatioResizer(28, 28, 20, 20, center_of_mass=True)

    output = resizer.resize_and_center(cropped)

    assert output.shape == (28, 28)
    assert np.count_nonzero(output) > 0


def test_invalid_content_size_raises() -> None:
    with pytest.raises(InvalidPreprocessingConfigError):
        AspectRatioResizer(28, 28, 30, 20)
