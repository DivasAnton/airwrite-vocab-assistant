from pathlib import Path

import cv2
import numpy as np
import pytest

from app.preprocessing.exceptions import EmptyDrawingError
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.storage.drawing_image_saver import DrawingImageSaver


def draw_block(
    x: int,
    y: int,
    width: int = 20,
    height: int = 30,
    canvas_size: tuple[int, int] = (100, 100),
) -> np.ndarray:
    image = np.zeros(canvas_size, dtype=np.uint8)
    image[y : y + height, x : x + width] = 255
    return image


def foreground_center(image: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(image > 0)
    return float(xs.mean()), float(ys.mean())


def make_preprocessor() -> HandwritingPreprocessor:
    return HandwritingPreprocessor()


def test_process_bgr_input_returns_model_ready_images() -> None:
    gray = draw_block(30, 20)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    preprocessor = make_preprocessor()

    result = preprocessor.process(bgr)

    assert result.processed_image.shape == (28, 28)
    assert result.processed_image.dtype == np.uint8
    assert result.normalized_image.shape == (28, 28)
    assert result.normalized_image.dtype == np.float32
    assert result.normalized_image.min() >= 0.0
    assert result.normalized_image.max() <= 1.0


def test_process_grayscale_input_uses_same_pipeline() -> None:
    image = draw_block(30, 20)
    preprocessor = make_preprocessor()

    result = preprocessor.process(image)

    assert result.processed_image.shape == (28, 28)
    assert result.bounding_box.x == 22
    assert result.bounding_box.y == 12


def test_empty_canvas_raises() -> None:
    preprocessor = make_preprocessor()

    with pytest.raises(EmptyDrawingError):
        preprocessor.process(np.zeros((100, 100), dtype=np.uint8))


def test_input_is_not_mutated() -> None:
    image = draw_block(30, 20)
    original = image.copy()
    preprocessor = make_preprocessor()

    preprocessor.process(image)

    assert np.array_equal(image, original)


def test_left_and_right_drawings_are_centered_similarly() -> None:
    preprocessor = make_preprocessor()
    left = preprocessor.process(draw_block(20, 30)).processed_image
    right = preprocessor.process(draw_block(60, 30)).processed_image

    assert np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))) < 1.0


def test_output_foreground_is_near_center() -> None:
    preprocessor = make_preprocessor()

    result = preprocessor.process(draw_block(5, 5))
    center_x, center_y = foreground_center(result.processed_image)

    assert center_x == pytest.approx(13.5, abs=1.5)
    assert center_y == pytest.approx(13.5, abs=1.5)


def test_tall_character_is_not_stretched_wide() -> None:
    preprocessor = make_preprocessor()

    result = preprocessor.process(draw_block(30, 10, width=8, height=60))
    ys, xs = np.where(result.processed_image > 0)

    assert (xs.max() - xs.min() + 1) < (ys.max() - ys.min() + 1)


def test_wide_character_is_not_stretched_tall() -> None:
    preprocessor = make_preprocessor()

    result = preprocessor.process(draw_block(10, 40, width=60, height=8))
    ys, xs = np.where(result.processed_image > 0)

    assert (xs.max() - xs.min() + 1) > (ys.max() - ys.min() + 1)


def test_debug_images_are_returned_when_enabled() -> None:
    preprocessor = HandwritingPreprocessor(include_debug_images=True)

    result = preprocessor.process(draw_block(30, 20))

    assert set(result.debug_images) == {
        "01_grayscale",
        "02_binary",
        "03_cropped",
        "04_processed",
    }


def test_saved_sprint_7_png_can_be_processed(tmp_path: Path) -> None:
    image = cv2.cvtColor(draw_block(30, 20), cv2.COLOR_GRAY2BGR)
    saver = DrawingImageSaver(output_dir=tmp_path)
    save_result = saver.save(image)
    assert save_result.file_path is not None

    result = make_preprocessor().process_file(save_result.file_path)

    assert result.processed_image.shape == (28, 28)
    assert result.normalized_image.dtype == np.float32
