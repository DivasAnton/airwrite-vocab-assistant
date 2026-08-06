from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from app.storage.exceptions import DrawingImageSaveError, InvalidDrawingImageError

CHARACTER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE_CHARACTER_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")
DATASET_WRITING_STYLES = frozenset({"uppercase", "lowercase"})


@dataclass(frozen=True)
class CharacterDatasetSaveResult:
    label: str
    writing_style: str
    file_path: Path
    saved_at: datetime
    message: str


class CharacterDatasetImageSaver:
    def __init__(
        self,
        output_dir: Path,
        image_format: str = "png",
        filename_prefix: str = "airwrite",
        writing_style: str = "uppercase",
    ) -> None:
        normalized_format = image_format.lower().strip().lstrip(".")
        if normalized_format != "png":
            raise ValueError(f"image_format must be png, got {image_format!r}")
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if not filename_prefix.strip():
            raise ValueError("filename_prefix must not be empty")
        normalized_style = writing_style.strip().lower()
        if normalized_style not in DATASET_WRITING_STYLES:
            raise ValueError("writing_style must be uppercase or lowercase")

        self.output_dir = output_dir
        self.image_format = normalized_format
        self.filename_prefix = filename_prefix.strip()
        self.writing_style = normalized_style

    def save(self, label: str, image: NDArray[np.uint8]) -> CharacterDatasetSaveResult:
        normalized_label = self.normalize_capture_label(label)
        self._validate_image(image)
        saved_at = datetime.now()
        label_dir = self.label_directory(normalized_label)
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
            writing_style=self.writing_style,
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

    def normalize_capture_label(self, label: str) -> str:
        normalized = label.strip()
        if len(normalized) != 1 or not normalized.isalpha() or not normalized.isascii():
            raise ValueError(f"label must be one ASCII letter, got {label!r}")
        normalized = normalized.upper() if self.writing_style == "uppercase" else normalized.lower()
        if normalized not in self.capture_labels:
            raise ValueError(f"label is invalid for {self.writing_style}: {label!r}")
        return normalized

    @property
    def capture_labels(self) -> tuple[str, ...]:
        return CHARACTER_LABELS if self.writing_style == "uppercase" else LOWERCASE_CHARACTER_LABELS

    def label_directory(self, label: str) -> Path:
        normalized = self.normalize_capture_label(label)
        if self.writing_style == "lowercase":
            return self.output_dir / "lowercase" / normalized
        return self.output_dir / normalized

    def next_capture_label(self, label: str) -> str:
        normalized = self.normalize_capture_label(label)
        index = self.capture_labels.index(normalized)
        return self.capture_labels[(index + 1) % len(self.capture_labels)]

    def previous_capture_label(self, label: str) -> str:
        normalized = self.normalize_capture_label(label)
        index = self.capture_labels.index(normalized)
        return self.capture_labels[(index - 1) % len(self.capture_labels)]

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
