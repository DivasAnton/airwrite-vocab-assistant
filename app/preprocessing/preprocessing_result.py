from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.preprocessing_source import PreprocessingSource


@dataclass(frozen=True)
class PreprocessingResult:
    processed_image: NDArray[np.uint8]
    normalized_image: NDArray[np.float32]
    bounding_box: BoundingBox | None
    original_shape: tuple[int, ...]
    cropped_shape: tuple[int, int]
    foreground_pixel_count: int
    debug_images: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    source: PreprocessingSource = PreprocessingSource.AIRWRITE_CANVAS
    orientation_transform: str = "none"
    foreground_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.processed_image.dtype != np.uint8 or self.processed_image.ndim != 2:
            raise ValueError("processed_image must be a 2D uint8 array")
        if self.normalized_image.dtype != np.float32:
            raise ValueError("normalized_image must use float32 dtype")
        if self.normalized_image.shape != self.processed_image.shape:
            raise ValueError("normalized_image shape must match processed_image")
        if not np.isfinite(self.normalized_image).all():
            raise ValueError("normalized_image must not contain NaN or infinite values")
        if self.normalized_image.size and (
            self.normalized_image.min() < 0.0 or self.normalized_image.max() > 1.0
        ):
            raise ValueError("normalized_image values must stay in range 0.0 to 1.0")
        if self.foreground_pixel_count < 0:
            raise ValueError("foreground_pixel_count must be greater than or equal to 0")
        if not 0.0 <= self.foreground_ratio <= 1.0:
            raise ValueError("foreground_ratio must be between 0.0 and 1.0")
        if not self.orientation_transform.strip():
            raise ValueError("orientation_transform must not be empty")
