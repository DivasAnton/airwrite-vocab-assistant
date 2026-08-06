from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.image_validator import ImageValidator


@dataclass(frozen=True)
class AirWriteAdaptationResult:
    prepared_image: NDArray[np.uint8]
    bounding_box: BoundingBox
    original_shape: tuple[int, ...]
    cropped_shape: tuple[int, int]
    foreground_pixel_count: int
    debug_images: dict[str, NDArray[np.uint8]]


class AirWriteCanvasAdapter:
    def __init__(
        self,
        validator: ImageValidator | None = None,
        normalizer: ForegroundNormalizer | None = None,
        extractor: BoundingBoxExtractor | None = None,
        resizer: AspectRatioResizer | None = None,
        include_debug_images: bool = False,
    ) -> None:
        self.validator = validator or ImageValidator()
        self.normalizer = normalizer or ForegroundNormalizer()
        self.extractor = extractor or BoundingBoxExtractor()
        self.resizer = resizer or AspectRatioResizer()
        self.include_debug_images = include_debug_images

    def adapt(self, image: object) -> NDArray[np.uint8]:
        return self.adapt_with_metadata(image).prepared_image

    def adapt_with_metadata(self, image: object) -> AirWriteAdaptationResult:
        valid_image = self.validator.validate(image)
        foreground = self.normalizer.normalize(valid_image)
        bounding_box = self.extractor.find(foreground.binary)
        cropped = self.extractor.crop(foreground.grayscale, bounding_box)
        prepared = self.resizer.resize_and_center(cropped)
        foreground_pixel_count = self.extractor.count_foreground(foreground.binary)

        debug_images: dict[str, NDArray[np.uint8]] = {}
        if self.include_debug_images:
            debug_images = {
                "01_grayscale": foreground.grayscale,
                "02_binary": foreground.binary,
                "03_cropped": cropped,
                "04_processed": prepared,
            }

        return AirWriteAdaptationResult(
            prepared_image=prepared,
            bounding_box=bounding_box,
            original_shape=tuple(valid_image.shape),
            cropped_shape=tuple(cropped.shape),
            foreground_pixel_count=foreground_pixel_count,
            debug_images=debug_images,
        )
