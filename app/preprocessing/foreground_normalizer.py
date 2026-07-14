from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ForegroundResult:
    grayscale: NDArray[np.uint8]
    binary: NDArray[np.uint8]


class ForegroundNormalizer:
    def __init__(self, binary_threshold: int = 20, invert_input: bool = False) -> None:
        if not 0 <= binary_threshold <= 255:
            raise ValueError(f"binary_threshold must be between 0 and 255, got {binary_threshold}")

        self.binary_threshold = binary_threshold
        self.invert_input = invert_input

    def normalize(self, image: NDArray[np.uint8]) -> ForegroundResult:
        grayscale = self.to_grayscale(image)
        if self.invert_input:
            grayscale = cast(NDArray[np.uint8], cv2.bitwise_not(grayscale))

        _, binary = cv2.threshold(
            grayscale,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        return ForegroundResult(
            grayscale=cast(NDArray[np.uint8], grayscale),
            binary=cast(NDArray[np.uint8], binary),
        )

    @staticmethod
    def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.ndim == 2:
            return image.copy()
        if image.shape[2] == 3:
            return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if image.shape[2] == 4:
            return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
        raise ValueError(f"Expected grayscale, BGR, or BGRA image, got shape {image.shape}")
