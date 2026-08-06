"""Handwriting image preprocessing pipeline."""

from app.preprocessing.airwrite_canvas_adapter import (
    AirWriteAdaptationResult,
    AirWriteCanvasAdapter,
)
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.emnist_source_adapter import EMNISTSourceAdapter
from app.preprocessing.exceptions import EmptyDrawingError, InvalidImageError, PreprocessingError
from app.preprocessing.foreground_normalizer import ForegroundNormalizer, ForegroundResult
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.preprocessing.image_validator import ImageValidator
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor
from app.preprocessing.preprocessing_auditor import (
    PreprocessingAuditor,
    PreprocessingAuditRecord,
)
from app.preprocessing.preprocessing_result import PreprocessingResult
from app.preprocessing.preprocessing_source import PreprocessingSource

__all__ = [
    "AirWriteAdaptationResult",
    "AirWriteCanvasAdapter",
    "AspectRatioResizer",
    "BoundingBox",
    "BoundingBoxExtractor",
    "EMNISTSourceAdapter",
    "EmptyDrawingError",
    "ForegroundNormalizer",
    "ForegroundResult",
    "HandwritingPreprocessor",
    "ImageValidator",
    "InvalidImageError",
    "ModelInputContract",
    "ModelInputPreprocessor",
    "PreprocessingAuditRecord",
    "PreprocessingAuditor",
    "PreprocessingError",
    "PreprocessingResult",
    "PreprocessingSource",
]
