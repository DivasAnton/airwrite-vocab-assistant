import numpy as np

from app.camera.frame_processor import FrameProcessor


def test_validate_frame_accepts_valid_frame() -> None:
    processor = FrameProcessor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert processor.validate_frame(frame) is True


def test_validate_frame_rejects_none() -> None:
    processor = FrameProcessor()

    assert processor.validate_frame(None) is False


def test_validate_frame_rejects_empty_array() -> None:
    processor = FrameProcessor()

    assert processor.validate_frame(np.array([])) is False


def test_mirror_frame_flips_pixels_horizontally() -> None:
    processor = FrameProcessor()
    frame = np.array([[[1, 0, 0], [2, 0, 0]]], dtype=np.uint8)

    mirrored = processor.mirror_frame(frame)

    assert mirrored[0, 0, 0] == 2
    assert mirrored[0, 1, 0] == 1


def test_draw_debug_info_keeps_frame_shape() -> None:
    processor = FrameProcessor()
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    output = processor.draw_debug_info(frame, fps=29.6)

    assert isinstance(output, np.ndarray)
    assert output.shape == frame.shape
