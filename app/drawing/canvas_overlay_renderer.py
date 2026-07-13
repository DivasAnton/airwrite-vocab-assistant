import cv2
import numpy as np
from numpy.typing import NDArray


class CanvasOverlayRenderer:
    def __init__(
        self,
        background_color: tuple[int, int, int] = (0, 0, 0),
        opacity: float = 1.0,
    ) -> None:
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be between 0.0 and 1.0, got {opacity}")

        self.background_color = np.array(background_color, dtype=np.uint8)
        self.opacity = opacity

    def render(
        self,
        frame: NDArray[np.uint8],
        canvas: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        self._validate_images(frame, canvas)

        output = frame.copy()
        mask = np.any(canvas != self.background_color, axis=2)
        if not np.any(mask):
            return output

        if self.opacity >= 1.0:
            output[mask] = canvas[mask]
            return output

        blended = cv2.addWeighted(frame, 1.0, canvas, self.opacity, 0.0)
        output[mask] = blended[mask]
        return output

    @staticmethod
    def _validate_images(frame: NDArray[np.uint8], canvas: NDArray[np.uint8]) -> None:
        if frame.shape != canvas.shape:
            raise ValueError(
                f"frame and canvas must have the same shape, got {frame.shape} and {canvas.shape}"
            )
        if frame.ndim != 3 or canvas.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame and canvas must be color images with 3 channels")
        if frame.dtype != np.uint8 or canvas.dtype != np.uint8:
            raise ValueError("frame and canvas must use uint8 dtype")
