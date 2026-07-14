from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox


@dataclass(frozen=True)
class PreprocessingResult:
    processed_image: NDArray[np.uint8]
    normalized_image: NDArray[np.float32]
    bounding_box: BoundingBox
    original_shape: tuple[int, ...]
    cropped_shape: tuple[int, int]
    foreground_pixel_count: int
    debug_images: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
