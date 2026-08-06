import cv2
import numpy as np
import pytest

from app.preprocessing.airwrite_canvas_adapter import AirWriteCanvasAdapter
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.exceptions import EmptyDrawingError


def adapter() -> AirWriteCanvasAdapter:
    return AirWriteCanvasAdapter(
        extractor=BoundingBoxExtractor(crop_padding=2, min_foreground_pixels=5),
        resizer=AspectRatioResizer(content_width=20, content_height=20),
    )


def component_count(image: np.ndarray) -> int:
    mask = (image > 20).astype(np.uint8)
    return int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)


def test_centered_and_corner_drawings_share_contract_shape() -> None:
    centered = np.zeros((100, 100), dtype=np.uint8)
    centered[30:70, 45:55] = 255
    corner = np.zeros((100, 100), dtype=np.uint8)
    corner[3:43, 4:14] = 255

    centered_output = adapter().adapt(centered)
    corner_output = adapter().adapt(corner)

    assert centered_output.shape == (28, 28)
    assert corner_output.shape == (28, 28)
    assert np.mean(np.abs(centered_output.astype(int) - corner_output.astype(int))) < 1.0


def test_tall_and_wide_drawings_keep_aspect_ratio() -> None:
    tall = np.zeros((100, 100), dtype=np.uint8)
    tall[10:90, 45:53] = 255
    wide = np.zeros((100, 100), dtype=np.uint8)
    wide[45:53, 10:90] = 255

    tall_output = adapter().adapt(tall)
    wide_output = adapter().adapt(wide)
    tall_y, tall_x = np.where(tall_output > 20)
    wide_y, wide_x = np.where(wide_output > 20)

    assert np.ptp(tall_y) > np.ptp(tall_x)
    assert np.ptp(wide_x) > np.ptp(wide_y)


@pytest.mark.parametrize("kind", ["i", "j", "t"])
def test_disconnected_or_crossbar_components_are_preserved(kind: str) -> None:
    image = np.zeros((120, 100), dtype=np.uint8)
    if kind in {"i", "j"}:
        image[18:25, 47:54] = 255
        image[42:100, 47:54] = 255
        if kind == "j":
            image[92:100, 39:54] = 255
    else:
        image[20:105, 47:54] = 255
        image[38:45, 30:70] = 255

    output = adapter().adapt(image)

    assert np.count_nonzero(output) > 0
    if kind in {"i", "j"}:
        assert component_count(output) == 2
    else:
        assert component_count(output) == 1


def test_adapter_returns_metadata_for_legacy_facade() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:60, 30:50] = 255

    result = adapter().adapt_with_metadata(image)

    assert result.original_shape == (100, 100, 3)
    assert result.cropped_shape[0] > 0
    assert result.foreground_pixel_count == 800
    assert result.bounding_box.x == 28


def test_empty_canvas_is_rejected() -> None:
    with pytest.raises(EmptyDrawingError):
        adapter().adapt(np.zeros((100, 100), dtype=np.uint8))
