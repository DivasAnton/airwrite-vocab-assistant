import json
from pathlib import Path
from typing import Any

import pytest

from app.inference.exceptions import InvalidModelBundleError, ModelArtifactNotFoundError
from app.inference.model_bundle_loader import ModelBundleLoader
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


class FakeModel:
    input_shape = (None, 28, 28, 1)
    output_shape = (None, 26)


def make_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    model_path = tmp_path / "model.keras"
    identity_path = tmp_path / "identity.json"
    lowercase_path = tmp_path / "lowercase.json"
    uppercase_path = tmp_path / "uppercase.json"
    metadata_path = tmp_path / "metadata.json"
    preprocessing_path = tmp_path / "preprocessing.json"
    model_path.write_bytes(b"model")
    identity_path.write_text(json.dumps({"labels": list(LETTER_IDENTITY_LABELS)}), encoding="utf-8")
    lowercase_path.write_text(
        json.dumps({"labels": list(LETTER_IDENTITY_LABELS)}), encoding="utf-8"
    )
    uppercase_path.write_text(
        json.dumps({"labels": [label.upper() for label in LETTER_IDENTITY_LABELS]}),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "1.0.0",
                "task_type": "letter_identity_classification",
                "case_sensitive": False,
                "case_source": "user_selected_mode",
                "input_shape": [28, 28, 1],
                "num_classes": 26,
            }
        ),
        encoding="utf-8",
    )
    preprocessing_path.write_text(json.dumps({"input_width": 28}), encoding="utf-8")
    return (
        model_path,
        identity_path,
        lowercase_path,
        uppercase_path,
        metadata_path,
        preprocessing_path,
    )


def make_loader(tmp_path: Path, model_loader: Any) -> ModelBundleLoader:
    return ModelBundleLoader(*make_artifacts(tmp_path), model_loader=model_loader)


def test_loader_loads_once_with_safe_inference_options(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def load_model(path: str, **kwargs: object) -> FakeModel:
        calls.append((path, kwargs))
        return FakeModel()

    loader = make_loader(tmp_path, load_model)
    first = loader.load()
    second = loader.load()

    assert first is second
    assert first.identity_labels == LETTER_IDENTITY_LABELS
    assert first.lowercase_display_labels == LETTER_IDENTITY_LABELS
    assert first.uppercase_display_labels[0] == "A"
    assert first.model_version == "1.0.0"
    assert len(calls) == 1
    assert calls[0][1] == {"compile": False, "safe_mode": True}


def test_loader_reports_missing_artifact(tmp_path: Path) -> None:
    paths = make_artifacts(tmp_path)
    paths[1].unlink()

    with pytest.raises(ModelArtifactNotFoundError, match="labels"):
        ModelBundleLoader(*paths, model_loader=lambda *_args, **_kwargs: FakeModel()).load()


@pytest.mark.parametrize(
    ("mapping_index", "payload"),
    [
        (1, {"labels": list(LETTER_IDENTITY_LABELS[:-1])}),
        (2, {"labels": [*LETTER_IDENTITY_LABELS[:-1], "a"]}),
        (3, {"labels": list(reversed(LETTER_IDENTITY_LABELS))}),
    ],
)
def test_loader_rejects_invalid_label_mapping(
    tmp_path: Path, mapping_index: int, payload: dict[str, object]
) -> None:
    paths = make_artifacts(tmp_path)
    paths[mapping_index].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidModelBundleError):
        ModelBundleLoader(*paths, model_loader=lambda *_args, **_kwargs: FakeModel()).load()


def test_loader_rejects_malformed_metadata_json(tmp_path: Path) -> None:
    paths = make_artifacts(tmp_path)
    paths[4].write_text("{broken", encoding="utf-8")

    with pytest.raises(InvalidModelBundleError, match="metadata"):
        ModelBundleLoader(*paths, model_loader=lambda *_args, **_kwargs: FakeModel()).load()


def test_loader_wraps_keras_load_failure(tmp_path: Path) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("bad model")

    with pytest.raises(InvalidModelBundleError, match="Could not load"):
        make_loader(tmp_path, fail).load()
