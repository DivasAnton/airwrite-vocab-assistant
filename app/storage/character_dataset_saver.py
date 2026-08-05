from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from app.storage.exceptions import DrawingImageSaveError, InvalidDrawingImageError

CHARACTER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class CharacterDatasetSaveResult:
    label: str
    file_path: Path
    saved_at: datetime
    message: str


class CharacterDatasetImageSaver:
    def __init__(
        self,
        output_dir: Path,
        image_format: str = "png",
        filename_prefix: str = "airwrite",
    ) -> None:
        normalized_format = image_format.lower().strip().lstrip(".")
        if normalized_format != "png":
            raise ValueError(f"image_format must be png, got {image_format!r}")
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if not filename_prefix.strip():
            raise ValueError("filename_prefix must not be empty")

        self.output_dir = output_dir
        self.image_format = normalized_format
        self.filename_prefix = filename_prefix.strip()

    def save(self, label: str, image: NDArray[np.uint8]) -> CharacterDatasetSaveResult:
        normalized_label = self.normalize_label(label)
        self._validate_image(image)
        saved_at = datetime.now()
        label_dir = self.output_dir / normalized_label
        label_dir.mkdir(parents=True, exist_ok=True)
        output_path = label_dir / self._build_filename(normalized_label, saved_at)

        try:
            success = cv2.imwrite(str(output_path), image)
        except OSError as error:
            raise DrawingImageSaveError(
                f"Failed to save dataset image to {output_path}: {error}"
            ) from error

        if not success:
            raise DrawingImageSaveError(f"Failed to save dataset image to {output_path}")

        return CharacterDatasetSaveResult(
            label=normalized_label,
            file_path=output_path,
            saved_at=saved_at,
            message=f"Saved dataset sample {output_path.name}",
        )

    @staticmethod
    def normalize_label(label: str) -> str:
        normalized = label.strip().upper()
        if normalized not in CHARACTER_LABELS:
            raise ValueError(f"label must be A-Z, got {label!r}")
        return normalized

    @staticmethod
    def next_label(label: str) -> str:
        normalized = CharacterDatasetImageSaver.normalize_label(label)
        index = CHARACTER_LABELS.index(normalized)
        return CHARACTER_LABELS[(index + 1) % len(CHARACTER_LABELS)]

    @staticmethod
    def previous_label(label: str) -> str:
        normalized = CharacterDatasetImageSaver.normalize_label(label)
        index = CHARACTER_LABELS.index(normalized)
        return CHARACTER_LABELS[(index - 1) % len(CHARACTER_LABELS)]

    def _build_filename(self, label: str, saved_at: datetime) -> str:
        timestamp = saved_at.strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid4().hex[:6]
        return f"{self.filename_prefix}_{label}_{timestamp}_{suffix}.{self.image_format}"

    @staticmethod
    def _validate_image(image: object) -> None:
        if not isinstance(image, np.ndarray):
            raise InvalidDrawingImageError("Dataset image must be a NumPy array")
        if image.size == 0:
            raise InvalidDrawingImageError("Dataset image must not be empty")
        if image.dtype != np.uint8:
            raise InvalidDrawingImageError("Dataset image must use uint8 dtype")
        if image.ndim != 2:
            raise InvalidDrawingImageError("Dataset image must be 2D grayscale")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise InvalidDrawingImageError("Dataset image width and height must be positive")
