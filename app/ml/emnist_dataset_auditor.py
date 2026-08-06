import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ml.emnist_letters_dataset import EMNISTDatasetSplit, EMNISTLettersDataset
from app.ml.exceptions import InvalidEMNISTDatasetError


@dataclass(frozen=True)
class EMNISTDatasetAudit:
    files: dict[str, dict[str, object]]
    splits: dict[str, dict[str, object]]
    raw_label_range: tuple[int, int]
    mapped_label_range: tuple[int, int]
    observed_class_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "status": "passed",
            "files": self.files,
            "splits": self.splits,
            "raw_label_range": list(self.raw_label_range),
            "mapped_label_range": list(self.mapped_label_range),
            "observed_class_count": self.observed_class_count,
        }


class EMNISTDatasetAuditor:
    def audit(
        self,
        dataset: EMNISTLettersDataset,
        paths: dict[str, Path] | None = None,
    ) -> EMNISTDatasetAudit:
        all_raw = np.concatenate((dataset.raw_train_labels, dataset.raw_test_labels))
        all_mapped = np.concatenate((dataset.official_train.labels, dataset.official_test.labels))
        unique = np.unique(all_mapped)
        if not np.array_equal(unique, np.arange(26)):
            raise InvalidEMNISTDatasetError(
                f"Expected exactly mapped labels 0-25, got {unique.astype(int).tolist()}"
            )
        split_reports = {
            "official_train": self._audit_split(dataset.official_train),
            "official_test": self._audit_split(dataset.official_test),
        }
        file_reports = {name: self._file_report(path) for name, path in (paths or {}).items()}
        return EMNISTDatasetAudit(
            files=file_reports,
            splits=split_reports,
            raw_label_range=(int(all_raw.min()), int(all_raw.max())),
            mapped_label_range=(int(all_mapped.min()), int(all_mapped.max())),
            observed_class_count=len(unique),
        )

    @staticmethod
    def _audit_split(split: EMNISTDatasetSplit) -> dict[str, object]:
        counts = np.bincount(split.labels, minlength=26)
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).astype(int).tolist()
            raise InvalidEMNISTDatasetError(
                f"{split.split_name} is missing mapped classes: {missing}"
            )
        flat = split.images.reshape(len(split.images), -1)
        return {
            "sample_count": len(split.images),
            "image_shape": [28, 28],
            "image_dtype": str(split.images.dtype),
            "label_dtype": str(split.labels.dtype),
            "class_counts": {str(index): int(value) for index, value in enumerate(counts)},
            "empty_image_count": int(np.count_nonzero(np.max(flat, axis=1) == 0)),
            "all_white_image_count": int(np.count_nonzero(np.min(flat, axis=1) == 255)),
        }

    @staticmethod
    def _file_report(path: Path) -> dict[str, object]:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
