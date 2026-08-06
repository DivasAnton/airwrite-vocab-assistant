from dataclasses import dataclass
from pathlib import Path

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.exceptions import DatasetStructureError
from app.ml.labels import CHARACTER_LABELS, label_to_index

ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
IGNORED_DATASET_DIRECTORIES = frozenset({"lowercase"})


@dataclass(frozen=True)
class DatasetScanIssue:
    image_path: Path
    error_type: str
    message: str


class DatasetScanner:
    def __init__(
        self,
        dataset_root: Path,
        source: str = "airwrite",
        project_root: Path | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.source = source
        self.project_root = project_root.resolve() if project_root is not None else None
        self.issues: list[DatasetScanIssue] = []

    def scan(self) -> list[DatasetManifestEntry]:
        if not self.dataset_root.is_dir():
            raise DatasetStructureError(f"Dataset root does not exist: {self.dataset_root}")

        self.issues = []
        self._check_label_directories()
        entries: list[DatasetManifestEntry] = []
        for label in CHARACTER_LABELS:
            label_dir = self.dataset_root / label
            if not label_dir.is_dir():
                self.issues.append(
                    DatasetScanIssue(
                        image_path=self._portable_path(label_dir),
                        error_type="missing_label_directory",
                        message=f"Missing required label directory: {label}",
                    )
                )
                continue

            for image_path in sorted(label_dir.iterdir(), key=lambda path: path.name.lower()):
                if image_path.name.startswith(".") or not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    continue
                entries.append(
                    DatasetManifestEntry(
                        image_path=self._portable_path(image_path),
                        label=label,
                        label_index=label_to_index(label),
                        source=self.source,
                    )
                )
        return entries

    def resolve_path(self, image_path: Path) -> Path:
        if image_path.is_absolute():
            return image_path
        if self.project_root is None:
            return image_path.resolve()
        return (self.project_root / image_path).resolve()

    def _check_label_directories(self) -> None:
        for child in sorted(self.dataset_root.iterdir(), key=lambda path: path.name.lower()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            if child.name in IGNORED_DATASET_DIRECTORIES:
                continue
            if child.name not in CHARACTER_LABELS:
                self.issues.append(
                    DatasetScanIssue(
                        image_path=self._portable_path(child),
                        error_type="invalid_label_directory",
                        message=f"Directory name {child.name!r} is not an uppercase A-Z label",
                    )
                )

    def _portable_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if self.project_root is not None:
            try:
                return resolved.relative_to(self.project_root)
            except ValueError:
                pass
        return resolved
