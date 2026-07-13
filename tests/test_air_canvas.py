import numpy as np
import pytest

from app.drawing.air_canvas import AirCanvas


def test_canvas_initializes_with_expected_shape() -> None:
    canvas = AirCanvas(width=640, height=480)

    assert canvas.get_image().shape == (480, 640, 3)


def test_canvas_starts_empty() -> None:
    canvas = AirCanvas(width=100, height=80)

    assert canvas.is_empty() is True


def test_draw_line_marks_canvas_not_empty() -> None:
    canvas = AirCanvas(width=120, height=100)

    canvas.draw_line((10, 10), (100, 80))

    assert canvas.is_empty() is False
    assert np.any(canvas.get_image() != canvas.background_color)


def test_clear_resets_canvas() -> None:
    canvas = AirCanvas(width=120, height=100)
    canvas.draw_line((10, 10), (100, 80))

    canvas.clear()

    assert canvas.is_empty() is True
    assert np.all(canvas.get_image() == canvas.background_color)


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_invalid_size_raises(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="width and height"):
        AirCanvas(width=width, height=height)


def test_invalid_thickness_raises() -> None:
    with pytest.raises(ValueError, match="stroke_thickness"):
        AirCanvas(width=100, height=80, stroke_thickness=0)


def test_get_image_returns_copy_by_default() -> None:
    canvas = AirCanvas(width=20, height=20)
    image = canvas.get_image()

    image[:] = 255

    assert np.all(canvas.get_image() == canvas.background_color)


def test_reset_size_recreates_empty_canvas() -> None:
    canvas = AirCanvas(width=20, height=20)
    canvas.draw_line((0, 0), (19, 19))

    canvas.reset_size(width=40, height=30)

    assert canvas.get_image().shape == (30, 40, 3)
    assert canvas.is_empty() is True
