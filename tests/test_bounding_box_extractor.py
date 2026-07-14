import numpy as np
import pytest

from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.exceptions import EmptyDrawingError


def test_find_bounding_box_for_center_rectangle() -> None:
    binary = np.zeros((100, 120), dtype=np.uint8)
    binary[30:60, 20:40] = 255
    extractor = BoundingBoxExtractor(crop_padding=0, min_foreground_pixels=10)

    bbox = extractor.find(binary)

    assert bbox == BoundingBox(x=20, y=30, width=20, height=30)


def test_padding_is_clamped_at_left_edge() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10:20, 0:5] = 255
    extractor = BoundingBoxExtractor(crop_padding=8, min_foreground_pixels=10)

    bbox = extractor.find(binary)

    assert bbox.x == 0
    assert bbox.y == 2
    assert bbox.right == 13


def test_padding_is_clamped_at_right_edge() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10:20, 45:50] = 255
    extractor = BoundingBoxExtractor(crop_padding=8, min_foreground_pixels=10)

    bbox = extractor.find(binary)

    assert bbox.x == 37
    assert bbox.right == 50


def test_empty_canvas_raises() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    extractor = BoundingBoxExtractor(crop_padding=0, min_foreground_pixels=10)

    with pytest.raises(EmptyDrawingError):
        extractor.find(binary)


def test_tiny_noise_below_minimum_raises() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10, 10] = 255
    extractor = BoundingBoxExtractor(crop_padding=0, min_foreground_pixels=2)

    with pytest.raises(EmptyDrawingError):
        extractor.find(binary)


def test_multiple_strokes_are_covered_by_one_bounding_box() -> None:
    binary = np.zeros((80, 80), dtype=np.uint8)
    binary[10:20, 10:20] = 255
    binary[50:60, 55:65] = 255
    extractor = BoundingBoxExtractor(crop_padding=0, min_foreground_pixels=10)

    bbox = extractor.find(binary)

    assert bbox == BoundingBox(x=10, y=10, width=55, height=50)


def test_crop_uses_bounding_box() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    bbox = BoundingBox(x=2, y=3, width=4, height=5)

    crop = BoundingBoxExtractor.crop(image, bbox)

    assert crop.shape == (5, 4)
    assert crop[0, 0] == image[3, 2]
