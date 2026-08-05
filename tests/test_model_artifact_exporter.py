import json
from pathlib import Path

import pytest

from app.ml.exceptions import ArtifactExportError
from app.ml.model_artifact_exporter import ModelArtifactExporter


def valid_metadata() -> dict[str, object]:
    return {
        "model_name": "airwrite_character_recognizer",
        "model_version": "0.1.0",
        "framework": "keras",
        "input_shape": [28, 28, 1],
        "num_classes": 26,
        "random_seed": 42,
        "dataset_manifest": "data/manifests/dataset_splits.csv",
        "training_timestamp": "2026-08-05T00:00:00+00:00",
        "best_epoch": 1,
        "validation_accuracy": 0.5,
        "test_accuracy": 0.5,
    }


def test_exporter_writes_fixed_label_order_and_metadata(tmp_path: Path) -> None:
    exporter = ModelArtifactExporter()
    labels_path = tmp_path / "labels" / "labels.json"
    metadata_path = tmp_path / "metadata" / "model_metadata.json"

    exporter.export_labels(labels_path)
    exporter.export_metadata(metadata_path, valid_metadata())

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["labels"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["best_epoch"] == 1


def test_exporter_rejects_invalid_metadata_without_writing(tmp_path: Path) -> None:
    output_path = tmp_path / "model_metadata.json"

    with pytest.raises(ArtifactExportError):
        ModelArtifactExporter().export_metadata(output_path, {"model_name": "incomplete"})

    assert not output_path.exists()


def test_exporter_has_explicit_no_overwrite_policy(tmp_path: Path) -> None:
    output_path = tmp_path / "labels.json"
    ModelArtifactExporter().export_labels(output_path)

    with pytest.raises(ArtifactExportError):
        ModelArtifactExporter(overwrite=False).export_labels(output_path)
