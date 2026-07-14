from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.exceptions import InvalidImageError


class ImageValidator:
    @staticmethod
    def validate(image: object) -> NDArray[np.uint8]:
        if image is None:
            raise InvalidImageError("Input image must not be None")
        if not isinstance(image, np.ndarray):
            raise InvalidImageError("Input image must be a NumPy array")
        if image.size == 0:
            raise InvalidImageError("Input image must not be empty")
        if image.dtype == object:
            raise InvalidImageError("Object arrays are not supported")
        if image.dtype != np.uint8:
            raise InvalidImageError("Input image must use uint8 dtype")
        if image.ndim not in {2, 3}:
            raise InvalidImageError("Input image must be 2D grayscale or 3D BGR/BGRA")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise InvalidImageError("Input image width and height must be positive")
        if image.ndim == 3 and image.shape[2] not in {3, 4}:
            raise InvalidImageError("3D input image must have 3 BGR or 4 BGRA channels")

        return cast(NDArray[np.uint8], image)
