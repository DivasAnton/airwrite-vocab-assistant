import importlib.util
import json
from pathlib import Path

import pytest

from app.ml.model_artifact_exporter import EMNISTModelArtifactExporter


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed"
)
def test_emnist_artifact_exports_labels_metadata_and_reloadable_model(tmp_path: Path) -> None:
    tensorflow = __import__("tensorflow")
    model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.Input(shape=(28, 28, 1)),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(26, activation="softmax"),
        ]
    )
    checkpoint = tmp_path / "checkpoint.keras"
    model.save(checkpoint)
    root = tmp_path / "bundle"
    model_path = EMNISTModelArtifactExporter().export_bundle(
        root,
        checkpoint,
        {
            "model_name": "airwrite_emnist_letter_identity",
            "model_version": "1.0.0",
            "num_classes": 26,
            "case_sensitive": False,
        },
        {"orientation_transform": "transpose", "normalization_divisor": 255.0},
    )
    labels = json.loads((root / "identity_labels.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    reloaded = tensorflow.keras.models.load_model(model_path, compile=False)
    assert labels["labels"] == list("abcdefghijklmnopqrstuvwxyz")
    assert metadata["case_sensitive"] is False
    assert reloaded.output_shape == (None, 26)
    assert (root / "preprocessing_config.json").is_file()
