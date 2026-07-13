import cv2
import numpy as np
from numpy.typing import NDArray

from app.vision.finger_tracking_result import FingerTrackingResult


class FingerTrackingRenderer:
    def __init__(
        self,
        draw_raw_point: bool = False,
        draw_smoothed_point: bool = True,
        point_radius: int = 8,
    ) -> None:
        if point_radius <= 0:
            raise ValueError(f"point_radius must be greater than 0, got {point_radius}")

        self.draw_raw_point = draw_raw_point
        self.draw_smoothed_point = draw_smoothed_point
        self.point_radius = point_radius

    def draw(
        self,
        frame: NDArray[np.uint8],
        result: FingerTrackingResult,
    ) -> NDArray[np.uint8]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a non-empty color image")
        if frame.size == 0:
            raise ValueError("frame must be a non-empty color image")

        output = frame.copy()
        if not result.is_tracked:
            return output

        if self.draw_raw_point and result.raw_point is not None:
            cv2.circle(
                output,
                result.raw_point,
                max(2, self.point_radius // 2),
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

        if self.draw_smoothed_point and result.smoothed_point is not None:
            cv2.circle(
                output, result.smoothed_point, self.point_radius, (255, 0, 255), 2, cv2.LINE_AA
            )
            self._draw_coordinates(output, result.smoothed_point)

        return output

    @staticmethod
    def _draw_coordinates(frame: NDArray[np.uint8], point: tuple[int, int]) -> None:
        x, y = point
        cv2.putText(
            frame,
            f"Point: {x}, {y}",
            (x + 12, max(y - 12, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
