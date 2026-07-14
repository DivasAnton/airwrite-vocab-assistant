from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from app.storage.exceptions import DrawingImageSaveError, InvalidDrawingImageError
from app.storage.save_result import SaveResult, SaveStatus

ImageWriter = Callable[[str, NDArray[np.uint8]], bool]


class DrawingImageSaver:
    def __init__(
        self,
        output_dir: Path,
        image_format: str = "png",
        filename_prefix: str = "drawing",
        writer: ImageWriter | None = None,
    ) -> None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        normalized_format = image_format.lower().strip().lstrip(".")
        if normalized_format != "png":
            raise ValueError(f"image_format must be png, got {image_format!r}")
        if not filename_prefix.strip():
            raise ValueError("filename_prefix must not be empty")

        self.output_dir = output_dir
        self.image_format = normalized_format
        self.filename_prefix = filename_prefix.strip()
        self.writer = writer or cv2.imwrite

    def save(self, image: NDArray[np.uint8]) -> SaveResult:
        self._validate_image(image)
        saved_at = datetime.now()
        output_path = self._build_output_path(saved_at)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            success = self.writer(str(output_path), image)
        except OSError as error:
            raise DrawingImageSaveError(
                f"Failed to save drawing image to {output_path}: {error}"
            ) from error

        if not success:
            raise DrawingImageSaveError(f"Failed to save drawing image to {output_path}")

        return SaveResult(
            status=SaveStatus.SAVED,
            file_path=output_path,
            saved_at=saved_at,
            message=f"Saved drawing to {output_path.name}",
        )

    @staticmethod
    def _validate_image(image: object) -> None:
        if not isinstance(image, np.ndarray):
            raise InvalidDrawingImageError("Drawing image must be a NumPy array")
        if image.size == 0:
            raise InvalidDrawingImageError("Drawing image must not be empty")
        if image.dtype != np.uint8:
            raise InvalidDrawingImageError("Drawing image must use uint8 dtype")
        if image.ndim not in {2, 3}:
            raise InvalidDrawingImageError("Drawing image must have 2 or 3 dimensions")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise InvalidDrawingImageError("Drawing image width and height must be positive")

    def _build_output_path(self, saved_at: datetime) -> Path:
        timestamp = saved_at.strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid4().hex[:6]
        filename = f"{self.filename_prefix}_{timestamp}_{suffix}.{self.image_format}"
        return self.output_dir / filename
