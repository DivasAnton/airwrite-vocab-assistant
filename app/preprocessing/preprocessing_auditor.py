import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.preprocessing_source import PreprocessingSource


@dataclass(frozen=True)
class PreprocessingAuditRecord:
    source: PreprocessingSource
    label: str | None
    foreground_pixel_count: int
    foreground_ratio: float
    bbox_width: int
    bbox_height: int
    bbox_width_ratio: float
    bbox_height_ratio: float
    centroid_x: float
    centroid_y: float
    mean_foreground_intensity: float
    component_count: int

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["source"] = self.source.value
        return record


class PreprocessingAuditor:
    METRIC_NAMES = (
        "foreground_pixel_count",
        "foreground_ratio",
        "bbox_width",
        "bbox_height",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "centroid_x",
        "centroid_y",
        "mean_foreground_intensity",
        "component_count",
    )

    def __init__(
        self,
        contract: ModelInputContract | None = None,
        foreground_threshold: int = 20,
    ) -> None:
        if not 0 <= foreground_threshold <= 255:
            raise ValueError("foreground_threshold must be between 0 and 255")
        self.contract = contract or ModelInputContract()
        self.foreground_threshold = foreground_threshold

    def analyze(
        self,
        image: object,
        *,
        source: PreprocessingSource,
        label: str | None = None,
    ) -> PreprocessingAuditRecord:
        prepared = self._validate_image(image)
        mask = prepared > self.foreground_threshold
        foreground_pixel_count = int(np.count_nonzero(mask))
        if foreground_pixel_count:
            ys, xs = np.where(mask)
            bbox_width = int(xs.max() - xs.min() + 1)
            bbox_height = int(ys.max() - ys.min() + 1)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            mean_foreground_intensity = float(prepared[mask].mean())
            component_count = int(
                cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
            )
        else:
            bbox_width = 0
            bbox_height = 0
            centroid_x = (self.contract.width - 1) / 2.0
            centroid_y = (self.contract.height - 1) / 2.0
            mean_foreground_intensity = 0.0
            component_count = 0

        return PreprocessingAuditRecord(
            source=source,
            label=label,
            foreground_pixel_count=foreground_pixel_count,
            foreground_ratio=foreground_pixel_count / prepared.size,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_width_ratio=bbox_width / self.contract.width,
            bbox_height_ratio=bbox_height / self.contract.height,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            mean_foreground_intensity=mean_foreground_intensity,
            component_count=component_count,
        )

    def summarize(self, records: list[PreprocessingAuditRecord]) -> dict[str, object]:
        if not records:
            raise ValueError("At least one preprocessing audit record is required")
        sources = sorted({record.source.value for record in records})
        metrics: dict[str, dict[str, float]] = {}
        for metric_name in self.METRIC_NAMES:
            values = np.asarray(
                [float(getattr(record, metric_name)) for record in records], dtype=np.float64
            )
            metrics[metric_name] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        return {
            "sources": sources,
            "sample_count": len(records),
            "contract": {
                "image_shape": list(self.contract.image_shape),
                "dtype": "uint8",
                "normalized_dtype": "float32",
                "normalized_range": [0.0, 1.0],
                "background": "dark",
                "foreground": "light",
            },
            "metrics": metrics,
            "records": [record.to_dict() for record in records],
        }

    def write_reports(
        self,
        records: list[PreprocessingAuditRecord],
        *,
        json_path: Path,
        csv_path: Path,
    ) -> None:
        summary = self.summarize(records)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as output_file:
            fieldnames = list(records[0].to_dict())
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(record.to_dict() for record in records)

    def _validate_image(self, image: object) -> NDArray[np.uint8]:
        if not isinstance(image, np.ndarray):
            raise InvalidImageError("Audit image must be a NumPy array")
        if image.dtype != np.uint8:
            raise InvalidImageError("Audit image must use uint8 dtype")
        if image.ndim != 2 or image.shape != self.contract.image_shape:
            raise InvalidImageError(
                f"Audit image shape must be {self.contract.image_shape}, got {image.shape}"
            )
        return image


def build_audit_grid(
    samples: list[tuple[str, NDArray[np.uint8]]],
    *,
    columns: int = 10,
    cell_size: int = 72,
) -> NDArray[np.uint8]:
    if not samples:
        raise ValueError("At least one sample is required to build an audit grid")
    if columns <= 0 or cell_size < 40:
        raise ValueError("columns must be positive and cell_size must be at least 40")
    rows = (len(samples) + columns - 1) // columns
    grid = np.full((rows * cell_size, columns * cell_size, 3), 32, dtype=np.uint8)
    preview_size = cell_size - 22
    for index, (label, image) in enumerate(samples):
        row, column = divmod(index, columns)
        x = column * cell_size
        y = row * cell_size
        preview = cv2.resize(
            image,
            (preview_size, preview_size),
            interpolation=cv2.INTER_NEAREST,
        )
        grid[y + 4 : y + 4 + preview_size, x + 11 : x + 11 + preview_size] = cv2.cvtColor(
            preview, cv2.COLOR_GRAY2BGR
        )
        cv2.putText(
            grid,
            label,
            (x + 5, y + cell_size - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return grid
