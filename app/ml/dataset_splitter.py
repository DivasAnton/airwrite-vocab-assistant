import random
from collections import defaultdict
from math import floor

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.exceptions import DatasetSplitError
from app.ml.labels import CHARACTER_LABELS

SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


class DatasetSplitter:
    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("Split ratios must be between 0 and 1")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must add up to 1.0")
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(self, entries: list[DatasetManifestEntry]) -> list[DatasetManifestEntry]:
        entries_by_label: dict[str, list[DatasetManifestEntry]] = defaultdict(list)
        for entry in entries:
            entries_by_label[entry.label].append(entry)

        validation_counts = self._allocate_counts(entries_by_label, self.validation_ratio)
        test_counts = self._allocate_counts(entries_by_label, self.test_ratio)
        split_entries: list[DatasetManifestEntry] = []
        for label_index, label in enumerate(CHARACTER_LABELS):
            label_entries = sorted(
                entries_by_label.get(label, []), key=lambda entry: entry.image_path.as_posix()
            )
            if len(label_entries) < 3:
                raise DatasetSplitError(
                    f"Label {label} needs at least 3 valid samples, got {len(label_entries)}"
                )

            random.Random(self.random_seed + label_index).shuffle(label_entries)
            validation_count = validation_counts[label]
            test_count = test_counts[label]
            train_count = len(label_entries) - validation_count - test_count
            if train_count < 1:
                raise DatasetSplitError(
                    f"Cannot keep a training sample for label {label} with the configured ratios"
                )
            train_end = train_count
            validation_end = train_end + validation_count
            split_entries.extend(entry.with_split("train") for entry in label_entries[:train_end])
            split_entries.extend(
                entry.with_split("validation") for entry in label_entries[train_end:validation_end]
            )
            split_entries.extend(
                entry.with_split("test") for entry in label_entries[validation_end:]
            )

        self.validate(split_entries)
        return sorted(
            split_entries,
            key=lambda entry: (
                SPLIT_ORDER[entry.split or ""],
                entry.label_index,
                entry.image_path.as_posix(),
            ),
        )

    def validate(self, entries: list[DatasetManifestEntry]) -> None:
        paths = [entry.image_path for entry in entries]
        if len(paths) != len(set(paths)):
            raise DatasetSplitError("An image path appears in more than one split")

        labels_by_split: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            if entry.split not in SPLIT_ORDER:
                raise DatasetSplitError(f"Unsupported split value: {entry.split!r}")
            labels_by_split[entry.split].add(entry.label)

        expected_labels = set(CHARACTER_LABELS)
        for split_name in SPLIT_ORDER:
            missing = expected_labels - labels_by_split[split_name]
            if missing:
                raise DatasetSplitError(
                    f"Split {split_name} is missing labels: {', '.join(sorted(missing))}"
                )

    @staticmethod
    def _allocate_counts(
        entries_by_label: dict[str, list[DatasetManifestEntry]], ratio: float
    ) -> dict[str, int]:
        total_samples = sum(len(entries_by_label.get(label, [])) for label in CHARACTER_LABELS)
        target_count = max(len(CHARACTER_LABELS), round(total_samples * ratio))
        counts = {
            label: max(1, floor(len(entries_by_label.get(label, [])) * ratio))
            for label in CHARACTER_LABELS
        }
        remaining = target_count - sum(counts.values())
        if remaining < 0:
            raise DatasetSplitError("Configured ratio is too small to include every label")

        labels_by_remainder = sorted(
            CHARACTER_LABELS,
            key=lambda label: (
                -(len(entries_by_label.get(label, [])) * ratio % 1),
                label,
            ),
        )
        if remaining > len(labels_by_remainder):
            raise DatasetSplitError("Could not allocate stratified split counts")
        for label in labels_by_remainder[:remaining]:
            counts[label] += 1
        return counts
