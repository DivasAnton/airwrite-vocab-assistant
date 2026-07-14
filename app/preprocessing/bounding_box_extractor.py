import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.exceptions import EmptyDrawingError, InvalidImageError


class BoundingBoxExtractor:
    def __init__(self, crop_padding: int = 8, min_foreground_pixels: int = 10) -> None:
        if crop_padding < 0:
            raise ValueError(f"crop_padding must be greater than or equal to 0, got {crop_padding}")
        if min_foreground_pixels <= 0:
            raise ValueError(
                f"min_foreground_pixels must be greater than 0, got {min_foreground_pixels}"
            )

        self.crop_padding = crop_padding
        self.min_foreground_pixels = min_foreground_pixels

    def find(self, binary: NDArray[np.uint8]) -> BoundingBox:
        self._validate_binary(binary)
        foreground_count = self.count_foreground(binary)
        if foreground_count < self.min_foreground_pixels:
            raise EmptyDrawingError(
                "Not enough foreground pixels to preprocess drawing: "
                f"{foreground_count}/{self.min_foreground_pixels}"
            )

        ys, xs = np.where(binary > 0)
        height, width = binary.shape
        x_start = max(int(xs.min()) - self.crop_padding, 0)
        y_start = max(int(ys.min()) - self.crop_padding, 0)
        x_end = min(int(xs.max()) + self.crop_padding + 1, width)
        y_end = min(int(ys.max()) + self.crop_padding + 1, height)

        return BoundingBox(
            x=x_start,
            y=y_start,
            width=x_end - x_start,
            height=y_end - y_start,
        )

    @staticmethod
    def crop(image: NDArray[np.uint8], bounding_box: BoundingBox) -> NDArray[np.uint8]:
        return image[
            bounding_box.y : bounding_box.bottom,
            bounding_box.x : bounding_box.right,
        ].copy()

    @staticmethod
    def count_foreground(binary: NDArray[np.uint8]) -> int:
        return int(np.count_nonzero(binary))

    @staticmethod
    def _validate_binary(binary: NDArray[np.uint8]) -> None:
        if not isinstance(binary, np.ndarray):
            raise InvalidImageError("Binary image must be a NumPy array")
        if binary.dtype != np.uint8:
            raise InvalidImageError("Binary image must use uint8 dtype")
        if binary.ndim != 2:
            raise InvalidImageError("Binary image must be 2D")
        if binary.size == 0 or binary.shape[0] <= 0 or binary.shape[1] <= 0:
            raise InvalidImageError("Binary image width and height must be positive")
