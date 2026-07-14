from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.exceptions import InvalidImageError, InvalidPreprocessingConfigError


class AspectRatioResizer:
    def __init__(
        self,
        output_width: int = 28,
        output_height: int = 28,
        content_width: int = 20,
        content_height: int = 20,
        center_of_mass: bool = False,
    ) -> None:
        self._validate_config(output_width, output_height, content_width, content_height)
        self.output_width = output_width
        self.output_height = output_height
        self.content_width = content_width
        self.content_height = content_height
        self.center_of_mass = center_of_mass

    def resize_and_center(self, cropped: NDArray[np.uint8]) -> NDArray[np.uint8]:
        self._validate_cropped(cropped)
        crop_height, crop_width = cropped.shape
        scale = min(self.content_width / crop_width, self.content_height / crop_height)
        resized_width = max(1, round(crop_width * scale))
        resized_height = max(1, round(crop_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        resized = cv2.resize(
            cropped,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        output = np.zeros((self.output_height, self.output_width), dtype=np.uint8)
        offset_x = (self.output_width - resized_width) // 2
        offset_y = (self.output_height - resized_height) // 2
        output[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized

        if self.center_of_mass:
            output = self._center_by_moments(output)

        return cast(NDArray[np.uint8], output)

    def _center_by_moments(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        moments = cv2.moments(image)
        if moments["m00"] == 0:
            return image

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        target_x = (self.output_width - 1) / 2.0
        target_y = (self.output_height - 1) / 2.0
        matrix = cast(
            NDArray[np.float32],
            np.array([[1, 0, target_x - center_x], [0, 1, target_y - center_y]], dtype=np.float32),
        )
        return cast(
            NDArray[np.uint8],
            cv2.warpAffine(
                image,
                matrix,
                (self.output_width, self.output_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
        )

    @staticmethod
    def _validate_cropped(cropped: NDArray[np.uint8]) -> None:
        if not isinstance(cropped, np.ndarray):
            raise InvalidImageError("Cropped image must be a NumPy array")
        if cropped.dtype != np.uint8:
            raise InvalidImageError("Cropped image must use uint8 dtype")
        if cropped.ndim != 2:
            raise InvalidImageError("Cropped image must be 2D grayscale")
        if cropped.size == 0 or cropped.shape[0] <= 0 or cropped.shape[1] <= 0:
            raise InvalidImageError("Cropped image width and height must be positive")

    @staticmethod
    def _validate_config(
        output_width: int,
        output_height: int,
        content_width: int,
        content_height: int,
    ) -> None:
        if output_width <= 0 or output_height <= 0:
            raise InvalidPreprocessingConfigError(
                f"Output size must be positive, got {output_width}x{output_height}"
            )
        if content_width <= 0 or content_height <= 0:
            raise InvalidPreprocessingConfigError(
                f"Content size must be positive, got {content_width}x{content_height}"
            )
        if content_width > output_width or content_height > output_height:
            raise InvalidPreprocessingConfigError(
                "Content size must fit inside output size, "
                f"got content {content_width}x{content_height}, "
                f"output {output_width}x{output_height}"
            )
