import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.ml.dataset_manifest import DatasetManifestEntry, read_manifest
from app.ml.labels import CHARACTER_LABELS
from app.utils.config import PROJECT_ROOT, training_settings


def select_samples(
    entries: list[DatasetManifestEntry], samples_per_class: int
) -> dict[str, list[DatasetManifestEntry]]:
    selected: dict[str, list[DatasetManifestEntry]] = defaultdict(list)
    for entry in entries:
        if len(selected[entry.label]) < samples_per_class:
            selected[entry.label].append(entry)
    return selected


def build_preview(
    entries: list[DatasetManifestEntry],
    samples_per_class: int = 5,
    cell_size: int = 84,
) -> np.ndarray:
    if samples_per_class <= 0 or cell_size < 40:
        raise ValueError("samples_per_class must be positive and cell_size must be at least 40")
    selected = select_samples(entries, samples_per_class)
    canvas = np.full(
        (len(CHARACTER_LABELS) * cell_size, (samples_per_class + 1) * cell_size, 3),
        245,
        dtype=np.uint8,
    )
    for row_index, label in enumerate(CHARACTER_LABELS):
        y_start = row_index * cell_size
        cv2.putText(
            canvas,
            label,
            (20, y_start + cell_size // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for column_index, entry in enumerate(selected.get(label, []), start=1):
            image_path = entry.image_path
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            preview_size = cell_size - 16
            preview = cv2.resize(
                image, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST
            )
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            x_start = column_index * cell_size + 8
            canvas[y_start + 8 : y_start + 8 + preview_size, x_start : x_start + preview_size] = (
                preview_bgr
            )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an A-Z dataset orientation preview")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/reports/dataset_preview.png",
    )
    arguments = parser.parse_args()
    train_entries = [
        entry for entry in read_manifest(training_settings.split_path) if entry.split == "train"
    ]
    preview = build_preview(train_entries, samples_per_class=arguments.samples_per_class)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(arguments.output), preview):
        raise RuntimeError(f"Could not save dataset preview: {arguments.output}")
    print(f"Dataset preview: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
