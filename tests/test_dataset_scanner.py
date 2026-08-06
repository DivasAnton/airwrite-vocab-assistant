from pathlib import Path

import cv2
import numpy as np

from app.ml.dataset_scanner import DatasetScanner


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((28, 28), dtype=np.uint8))


def test_scanner_uses_parent_directory_as_label_and_sorts_entries(tmp_path: Path) -> None:
    write_image(tmp_path / "B" / "b.png")
    write_image(tmp_path / "A" / "z.png")
    write_image(tmp_path / "A" / "a.png")

    scanner = DatasetScanner(tmp_path)
    entries = scanner.scan()

    assert [(entry.label, entry.image_path.name) for entry in entries] == [
        ("A", "a.png"),
        ("A", "z.png"),
        ("B", "b.png"),
    ]
    assert entries[0].label_index == 0


def test_scanner_reports_unknown_and_missing_label_directories(tmp_path: Path) -> None:
    (tmp_path / "wrong").mkdir()

    scanner = DatasetScanner(tmp_path)
    scanner.scan()

    error_types = {issue.error_type for issue in scanner.issues}
    assert "invalid_label_directory" in error_types
    assert "missing_label_directory" in error_types


def test_uppercase_scanner_ignores_lowercase_capture_root(tmp_path: Path) -> None:
    (tmp_path / "lowercase" / "a").mkdir(parents=True)

    scanner = DatasetScanner(tmp_path)
    scanner.scan()

    assert all(issue.image_path.name != "lowercase" for issue in scanner.issues)
