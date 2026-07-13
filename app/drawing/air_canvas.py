from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class AirCanvas:
    def __init__(
        self,
        width: int,
        height: int,
        background_color: tuple[int, int, int] = (0, 0, 0),
        stroke_color: tuple[int, int, int] = (255, 255, 255),
        stroke_thickness: int = 8,
    ) -> None:
        self._validate_size(width, height)
        self._validate_color("background_color", background_color)
        self._validate_color("stroke_color", stroke_color)
        if stroke_thickness <= 0:
            raise ValueError(f"stroke_thickness must be greater than 0, got {stroke_thickness}")

        self.width = width
        self.height = height
        self.background_color = background_color
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness
        self._has_content = False
        self._image = self._create_image()

    def draw_line(self, start_point: tuple[int, int], end_point: tuple[int, int]) -> None:
        start = self._clamp_point(start_point)
        end = self._clamp_point(end_point)
        cv2.line(
            self._image,
            start,
            end,
            self.stroke_color,
            self.stroke_thickness,
            cv2.LINE_AA,
        )
        self._has_content = True

    def clear(self) -> None:
        self._image[...] = self.background_color
        self._has_content = False

    def is_empty(self) -> bool:
        return not self._has_content

    def get_image(self, copy: bool = True) -> NDArray[np.uint8]:
        if copy:
            return self._image.copy()
        return self._image

    def reset_size(self, width: int, height: int) -> None:
        self._validate_size(width, height)
        self.width = width
        self.height = height
        self._image = self._create_image()
        self._has_content = False

    def matches_size(self, width: int, height: int) -> bool:
        return self.width == width and self.height == height

    def _create_image(self) -> NDArray[np.uint8]:
        return cast(
            NDArray[np.uint8],
            np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8),
        )

    @staticmethod
    def _validate_size(width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be greater than 0, got {width}x{height}")

    @staticmethod
    def _validate_color(name: str, color: tuple[int, int, int]) -> None:
        if any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"{name} channels must be between 0 and 255, got {color}")

    def _clamp_point(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)
