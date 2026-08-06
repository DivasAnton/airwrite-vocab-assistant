from pathlib import Path

import cv2

from app.preprocessing.airwrite_canvas_adapter import AirWriteCanvasAdapter
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.image_validator import ImageValidator
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor
from app.preprocessing.preprocessing_result import PreprocessingResult
from app.preprocessing.preprocessing_source import PreprocessingSource


class HandwritingPreprocessor:
    def __init__(
        self,
        validator: ImageValidator | None = None,
        normalizer: ForegroundNormalizer | None = None,
        extractor: BoundingBoxExtractor | None = None,
        resizer: AspectRatioResizer | None = None,
        include_debug_images: bool = False,
    ) -> None:
        selected_resizer = resizer or AspectRatioResizer()
        self.airwrite_adapter = AirWriteCanvasAdapter(
            validator=validator,
            normalizer=normalizer,
            extractor=extractor,
            resizer=selected_resizer,
            include_debug_images=include_debug_images,
        )
        self.model_input_preprocessor = ModelInputPreprocessor(
            ModelInputContract(
                width=selected_resizer.output_width,
                height=selected_resizer.output_height,
            )
        )

    def process(self, image: object) -> PreprocessingResult:
        adaptation = self.airwrite_adapter.adapt_with_metadata(image)
        return self.model_input_preprocessor.process(
            adaptation.prepared_image,
            source=PreprocessingSource.AIRWRITE_CANVAS,
            orientation_transform="none",
            original_shape=adaptation.original_shape,
            bounding_box=adaptation.bounding_box,
            cropped_shape=adaptation.cropped_shape,
            foreground_pixel_count=adaptation.foreground_pixel_count,
            debug_images=adaptation.debug_images,
        )

    def process_file(self, path: Path) -> PreprocessingResult:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError(f"Could not read image file: {path}")
        return self.process(image)
