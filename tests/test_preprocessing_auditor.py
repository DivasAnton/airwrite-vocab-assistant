from pathlib import Path

import numpy as np
import pytest

from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.preprocessing_auditor import PreprocessingAuditor, build_audit_grid
from app.preprocessing.preprocessing_source import PreprocessingSource


def two_component_image() -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[4:8, 5:9] = 200
    image[14:24, 12:17] = 255
    return image


def test_auditor_calculates_geometry_intensity_and_components() -> None:
    record = PreprocessingAuditor(foreground_threshold=20).analyze(
        two_component_image(),
        source=PreprocessingSource.EMNIST_LETTERS,
        label="I",
    )

    assert record.label == "I"
    assert record.foreground_pixel_count == 66
    assert record.foreground_ratio == pytest.approx(66 / 784)
    assert record.bbox_width == 12
    assert record.bbox_height == 20
    assert record.component_count == 2
    assert 200.0 < record.mean_foreground_intensity <= 255.0


def test_auditor_summarizes_and_writes_json_csv(tmp_path: Path) -> None:
    auditor = PreprocessingAuditor()
    records = [
        auditor.analyze(
            two_component_image(),
            source=PreprocessingSource.AIRWRITE_CANVAS,
            label="i",
        )
    ]

    summary = auditor.summarize(records)
    auditor.write_reports(
        records,
        json_path=tmp_path / "stats.json",
        csv_path=tmp_path / "stats.csv",
    )

    assert summary["sample_count"] == 1
    assert (tmp_path / "stats.json").exists()
    assert (tmp_path / "stats.csv").exists()


def test_audit_grid_has_stable_dimensions() -> None:
    grid = build_audit_grid([("I", two_component_image())], columns=2, cell_size=60)

    assert grid.shape == (60, 120, 3)


def test_auditor_rejects_wrong_contract_image() -> None:
    with pytest.raises(InvalidImageError):
        PreprocessingAuditor().analyze(
            np.zeros((27, 28), dtype=np.uint8),
            source=PreprocessingSource.AIRWRITE_CANVAS,
        )
