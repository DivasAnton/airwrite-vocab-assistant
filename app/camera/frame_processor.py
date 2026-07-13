from time import perf_counter
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class FrameProcessor:
    def __init__(self) -> None:
        self.previous_time: float | None = None

    def validate_frame(self, frame: object) -> bool:
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0 or frame.ndim != 3:
            return False

        height, width, channels = frame.shape
        return height > 0 and width > 0 and channels in {1, 3, 4}

    def mirror_frame(self, frame: NDArray[np.uint8], enabled: bool = True) -> NDArray[np.uint8]:
        if not enabled:
            return frame
        return cast(NDArray[np.uint8], cv2.flip(frame, 1))

    def calculate_fps(self) -> float:
        current_time = perf_counter()
        if self.previous_time is None:
            self.previous_time = current_time
            return 0.0

        elapsed = current_time - self.previous_time
        self.previous_time = current_time
        if elapsed <= 0:
            return 0.0

        return 1.0 / elapsed

    def draw_debug_info(
        self,
        frame: NDArray[np.uint8],
        fps: float,
        show_fps: bool = True,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        height, width = output.shape[:2]

        lines = []
        if show_fps:
            lines.append(f"FPS: {fps:.1f}")
        lines.append(f"Resolution: {width}x{height}")
        lines.append("Press Q or ESC to quit")

        y = 28
        for line in lines:
            cv2.putText(
                output,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 30

        return output
