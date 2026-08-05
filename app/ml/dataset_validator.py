import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.labels import CHARACTER_LABELS, label_to_index
from app.preprocessing.exceptions import PreprocessingError
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor


@dataclass(frozen=True)
class DatasetValidationIssue:
    image_path: Path
    error_type: str
    message: str


@dataclass(frozen=True)
class DatasetValidationReport:
    valid_entries: list[DatasetManifestEntry]
    issues: list[DatasetValidationIssue]
    sha256_by_path: dict[Path, str]

    @property
    def duplicate_count(self) -> int:
        return sum(issue.error_type == "exact_duplicate" for issue in self.issues)


class DatasetValidator:
    def __init__(
        self,
        preprocessor: HandwritingPreprocessor,
        project_root: Path | None = None,
        expected_shape: tuple[int, int] = (28, 28),
        require_grayscale: bool = True,
        required_extension: str | None = ".png",
    ) -> None:
        self.preprocessor = preprocessor
        self.project_root = project_root.resolve() if project_root is not None else None
        self.expected_shape = expected_shape
        self.require_grayscale = require_grayscale
        self.required_extension = required_extension.lower() if required_extension else None

    def validate(self, entries: list[DatasetManifestEntry]) -> DatasetValidationReport:
        valid_entries: list[DatasetManifestEntry] = []
        issues: list[DatasetValidationIssue] = []
        sha256_by_path: dict[Path, str] = {}
        first_path_by_hash: dict[str, Path] = {}

        for entry in entries:
            issue = self._validate_entry_metadata(entry)
            if issue is not None:
                issues.append(issue)
                continue

            resolved_path = self._resolve_path(entry.image_path)
            if not resolved_path.is_file():
                issues.append(self._issue(entry, "missing_file", "Image file does not exist"))
                continue
            if self.required_extension and resolved_path.suffix.lower() != self.required_extension:
                issues.append(
                    self._issue(
                        entry,
                        "wrong_format",
                        f"Expected {self.required_extension} image, got {resolved_path.suffix}",
                    )
                )
                continue

            file_hash = self._sha256(resolved_path)
            first_path = first_path_by_hash.get(file_hash)
            if first_path is not None:
                issues.append(
                    self._issue(
                        entry,
                        "exact_duplicate",
                        f"Exact duplicate of {first_path.as_posix()}",
                    )
                )
                continue

            image = cv2.imread(str(resolved_path), cv2.IMREAD_UNCHANGED)
            image_issue = self._validate_decoded_image(entry, image)
            if image_issue is not None:
                issues.append(image_issue)
                continue

            assert image is not None
            try:
                result = self.preprocessor.process(image)
            except (PreprocessingError, ValueError) as error:
                issues.append(self._issue(entry, "preprocessing_error", str(error)))
                continue

            normalized = result.normalized_image
            if normalized.shape != self.expected_shape:
                issues.append(
                    self._issue(
                        entry,
                        "wrong_processed_shape",
                        f"Expected processed shape {self.expected_shape}, got {normalized.shape}",
                    )
                )
                continue
            if normalized.dtype != np.float32:
                issues.append(
                    self._issue(
                        entry,
                        "wrong_normalized_dtype",
                        f"Expected float32 normalized image, got {normalized.dtype}",
                    )
                )
                continue
            if not np.isfinite(normalized).all():
                issues.append(
                    self._issue(entry, "non_finite_values", "Normalized image contains NaN or Inf")
                )
                continue
            if normalized.min() < 0.0 or normalized.max() > 1.0:
                issues.append(
                    self._issue(entry, "wrong_normalized_range", "Normalized values must be in 0-1")
                )
                continue

            first_path_by_hash[file_hash] = entry.image_path
            sha256_by_path[entry.image_path] = file_hash
            valid_entries.append(entry)

        return DatasetValidationReport(valid_entries, issues, sha256_by_path)

    def _validate_entry_metadata(
        self, entry: DatasetManifestEntry
    ) -> DatasetValidationIssue | None:
        if entry.label not in CHARACTER_LABELS:
            return self._issue(entry, "invalid_label", f"Label {entry.label!r} is not A-Z")
        if entry.label_index != label_to_index(entry.label):
            return self._issue(
                entry,
                "wrong_label_index",
                f"Label {entry.label} must use index {label_to_index(entry.label)}",
            )
        if entry.image_path.parent.name != entry.label:
            return self._issue(
                entry,
                "label_path_mismatch",
                "Label "
                f"{entry.label} does not match parent directory {entry.image_path.parent.name}",
            )
        return None

    def _validate_decoded_image(
        self, entry: DatasetManifestEntry, image: np.ndarray | None
    ) -> DatasetValidationIssue | None:
        if image is None or image.size == 0:
            return self._issue(entry, "unreadable_image", "OpenCV could not decode the image")
        if self.require_grayscale and image.ndim != 2:
            return self._issue(
                entry,
                "not_grayscale",
                f"Expected a 2D grayscale image, got shape {image.shape}",
            )
        if image.shape[:2] != self.expected_shape:
            return self._issue(
                entry,
                "wrong_image_shape",
                f"Expected image shape {self.expected_shape}, got {image.shape[:2]}",
            )
        if image.dtype != np.uint8:
            return self._issue(
                entry,
                "wrong_image_dtype",
                f"Expected uint8 image, got {image.dtype}",
            )
        return None

    def _resolve_path(self, image_path: Path) -> Path:
        if image_path.is_absolute():
            return image_path
        if self.project_root is None:
            return image_path.resolve()
        return (self.project_root / image_path).resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _issue(
        entry: DatasetManifestEntry, error_type: str, message: str
    ) -> DatasetValidationIssue:
        return DatasetValidationIssue(entry.image_path, error_type, message)
