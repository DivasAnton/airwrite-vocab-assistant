from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.camera.camera_stream import CameraStream


def test_constructor_stores_config() -> None:
    camera = CameraStream(camera_index=1, width=640, height=480, fps=24)

    assert camera.camera_index == 1
    assert camera.width == 640
    assert camera.height == 480
    assert camera.fps == 24


def test_read_returns_false_when_camera_is_not_open() -> None:
    camera = CameraStream(camera_index=0, width=640, height=480, fps=30)

    success, frame = camera.read()

    assert success is False
    assert frame is None


def test_release_is_safe_when_camera_was_never_opened() -> None:
    camera = CameraStream(camera_index=0, width=640, height=480, fps=30)

    camera.release()
    camera.release()


def test_open_failure_releases_camera_and_raises() -> None:
    capture = Mock()
    capture.isOpened.return_value = False

    with patch("app.camera.camera_stream.cv2.VideoCapture", return_value=capture):
        camera = CameraStream(camera_index=0, width=640, height=480, fps=30)

        with pytest.raises(RuntimeError, match="Unable to open camera index 0"):
            camera.open()

    capture.release.assert_called_once()
    assert camera.capture is None


def test_read_rejects_empty_frame() -> None:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.array([]))
    camera = CameraStream(camera_index=0, width=640, height=480, fps=30)
    camera.capture = capture

    success, frame = camera.read()

    assert success is False
    assert frame is None


def test_open_sets_camera_properties() -> None:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [640, 480, 30.0]

    with patch("app.camera.camera_stream.cv2.VideoCapture", return_value=capture):
        camera = CameraStream(camera_index=0, width=640, height=480, fps=30)
        camera.open()

    assert capture.set.call_count == 3
