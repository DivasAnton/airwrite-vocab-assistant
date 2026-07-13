import numpy as np
import pytest

from app.drawing.canvas_overlay_renderer import CanvasOverlayRenderer


def test_empty_canvas_returns_frame_copy_with_same_shape() -> None:
    renderer = CanvasOverlayRenderer()
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas = np.zeros((50, 60, 3), dtype=np.uint8)

    output = renderer.render(frame, canvas)

    assert output.shape == frame.shape
    assert np.array_equal(output, frame)
    assert output is not frame


def test_canvas_stroke_changes_output() -> None:
    renderer = CanvasOverlayRenderer()
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas[10, 10] = (255, 255, 255)

    output = renderer.render(frame, canvas)

    assert not np.array_equal(output, frame)
    assert np.array_equal(output[10, 10], canvas[10, 10])


def test_render_does_not_mutate_inputs() -> None:
    renderer = CanvasOverlayRenderer()
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas[10, 10] = (255, 255, 255)
    original_frame = frame.copy()
    original_canvas = canvas.copy()

    renderer.render(frame, canvas)

    assert np.array_equal(frame, original_frame)
    assert np.array_equal(canvas, original_canvas)


def test_shape_mismatch_raises() -> None:
    renderer = CanvasOverlayRenderer()
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    canvas = np.zeros((40, 60, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        renderer.render(frame, canvas)


def test_invalid_dtype_raises() -> None:
    renderer = CanvasOverlayRenderer()
    frame = np.zeros((50, 60, 3), dtype=np.float32)
    canvas = np.zeros((50, 60, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="uint8"):
        renderer.render(frame, canvas)


def test_opacity_blends_only_canvas_pixels() -> None:
    renderer = CanvasOverlayRenderer(opacity=0.5)
    frame = np.full((20, 20, 3), 10, dtype=np.uint8)
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    canvas[5, 5] = (250, 250, 250)

    output = renderer.render(frame, canvas)

    assert np.array_equal(output[0, 0], frame[0, 0])
    assert not np.array_equal(output[5, 5], frame[5, 5])
