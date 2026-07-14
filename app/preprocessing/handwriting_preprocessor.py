from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.image_validator import ImageValidator
from app.preprocessing.preprocessing_result import PreprocessingResult


class HandwritingPreprocessor:
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

    def process(self, image: object) -> PreprocessingResult:
        valid_image = self.validator.validate(image)
        original_shape = tuple(valid_image.shape)
        foreground = self.normalizer.normalize(valid_image)
        bounding_box = self.extractor.find(foreground.binary)
        cropped = self.extractor.crop(foreground.grayscale, bounding_box)
        processed = self.resizer.resize_and_center(cropped)
        normalized = cast(NDArray[np.float32], processed.astype(np.float32) / 255.0)
        foreground_pixel_count = self.extractor.count_foreground(foreground.binary)

        debug_images: dict[str, NDArray[np.uint8]] = {}
        if self.include_debug_images:
            debug_images = {
                "01_grayscale": foreground.grayscale,
                "02_binary": foreground.binary,
                "03_cropped": cropped,
                "04_processed": processed,
            }

        return PreprocessingResult(
            processed_image=processed,
            normalized_image=normalized,
            bounding_box=bounding_box,
            original_shape=original_shape,
            cropped_shape=tuple(cropped.shape),
            foreground_pixel_count=foreground_pixel_count,
            debug_images=debug_images,
        )

    def process_file(self, path: Path) -> PreprocessingResult:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError(f"Could not read image file: {path}")
        return self.process(image)
