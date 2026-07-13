import logging
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraStream:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.capture.isOpened():
            self.logger.error("Unable to open camera index %s", self.camera_index)
            self.release()
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")

        actual_width, actual_height, actual_fps = self.get_actual_settings()
        self.logger.info("Requested camera: %sx%s at %s FPS", self.width, self.height, self.fps)
        self.logger.info(
            "Actual camera: %sx%s at %.1f FPS", actual_width, actual_height, actual_fps
        )

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self.capture is None or not self.capture.isOpened():
            self.logger.error("Cannot read frame because camera is not open")
            return False, None

        success, frame = self.capture.read()
        if not success or frame is None or frame.size == 0:
            self.logger.warning("Failed to read a valid frame from camera")
            return False, None

        return True, cast(NDArray[np.uint8], frame)

    def get_actual_settings(self) -> tuple[int, int, float]:
        if self.capture is None:
            return 0, 0, 0.0

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return actual_width, actual_height, actual_fps

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
