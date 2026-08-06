import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.exceptions import (
    InvalidModelBundleError,
    ModelArtifactNotFoundError,
)
from app.inference.model_bundle import ModelBundle

KerasModelLoader = Callable[..., Any]


class ModelBundleLoader:
    def __init__(
        self,
        model_path: Path,
        identity_labels_path: Path,
        lowercase_display_labels_path: Path,
        uppercase_display_labels_path: Path,
        metadata_path: Path,
        preprocessing_config_path: Path,
        model_loader: KerasModelLoader | None = None,
    ) -> None:
        self.model_path = model_path
        self.identity_labels_path = identity_labels_path
        self.lowercase_display_labels_path = lowercase_display_labels_path
        self.uppercase_display_labels_path = uppercase_display_labels_path
        self.metadata_path = metadata_path
        self.preprocessing_config_path = preprocessing_config_path
        self._model_loader = model_loader or self._default_model_loader
        self._bundle: ModelBundle | None = None

    def load(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle

        for name, path in self._artifact_paths().items():
            if not path.is_file():
                raise ModelArtifactNotFoundError(f"Required {name} artifact not found: {path}")

        identity_payload = self._load_json(self.identity_labels_path, "identity labels")
        lowercase_payload = self._load_json(
            self.lowercase_display_labels_path, "lowercase display labels"
        )
        uppercase_payload = self._load_json(
            self.uppercase_display_labels_path, "uppercase display labels"
        )
        metadata = self._load_json(self.metadata_path, "model metadata")
        preprocessing_contract = self._load_json(
            self.preprocessing_config_path,
            "preprocessing configuration",
        )
        identity_labels = self._parse_labels(identity_payload, "identity labels")
        lowercase_labels = self._parse_labels(lowercase_payload, "lowercase display labels")
        uppercase_labels = self._parse_labels(uppercase_payload, "uppercase display labels")
        try:
            CaseLabelResolver.validate_mappings(
                identity_labels,
                lowercase_labels,
                uppercase_labels,
            )
        except ValueError as error:
            raise InvalidModelBundleError(str(error)) from error
        model_version = self._required_string(metadata, "model_version")
        task_type = self._required_string(metadata, "task_type")
        case_sensitive = self._required_bool(metadata, "case_sensitive")
        case_source = self._required_string(metadata, "case_source")
        input_shape = self._parse_input_shape(metadata.get("input_shape"))
        num_classes = self._required_int(metadata, "num_classes")
        if num_classes != len(identity_labels):
            raise InvalidModelBundleError(
                f"Metadata num_classes={num_classes} does not match {len(identity_labels)} labels"
            )

        try:
            model = self._model_loader(str(self.model_path), compile=False, safe_mode=True)
        except Exception as error:
            raise InvalidModelBundleError(
                f"Could not load Keras model artifact: {self.model_path}"
            ) from error

        digest = self._sha256(self.model_path)
        expected_digest = metadata.get("model_sha256")
        if expected_digest is not None and expected_digest != digest:
            raise InvalidModelBundleError("Model SHA-256 does not match model metadata")

        self._bundle = ModelBundle(
            model=model,
            identity_labels=identity_labels,
            lowercase_display_labels=lowercase_labels,
            uppercase_display_labels=uppercase_labels,
            model_version=model_version,
            task_type=task_type,
            case_sensitive=case_sensitive,
            case_source=case_source,
            expected_input_shape=input_shape,
            num_classes=num_classes,
            preprocessing_contract=preprocessing_contract,
            model_sha256=digest,
        )
        return self._bundle

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            "model": self.model_path,
            "identity labels": self.identity_labels_path,
            "lowercase display labels": self.lowercase_display_labels_path,
            "uppercase display labels": self.uppercase_display_labels_path,
            "model metadata": self.metadata_path,
            "preprocessing configuration": self.preprocessing_config_path,
        }

    @staticmethod
    def _load_json(path: Path, artifact_name: str) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InvalidModelBundleError(f"Invalid {artifact_name} JSON: {path}") from error
        if not isinstance(payload, dict):
            raise InvalidModelBundleError(f"{artifact_name} JSON must contain an object")
        return cast(dict[str, object], payload)

    @staticmethod
    def _parse_labels(payload: dict[str, object], artifact_name: str) -> tuple[str, ...]:
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise InvalidModelBundleError(f"{artifact_name} must be a non-empty list")
        if not all(isinstance(label, str) and label for label in raw_labels):
            raise InvalidModelBundleError(
                f"Every label in {artifact_name} must be a non-empty string"
            )
        labels = tuple(cast(list[str], raw_labels))
        if len(set(labels)) != len(labels):
            raise InvalidModelBundleError(f"{artifact_name} must not contain duplicates")
        return labels

    @staticmethod
    def _required_string(payload: dict[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise InvalidModelBundleError(f"Model metadata field {key!r} must be a string")
        return value

    @staticmethod
    def _required_int(payload: dict[str, object], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InvalidModelBundleError(
                f"Model metadata field {key!r} must be a positive integer"
            )
        return value

    @staticmethod
    def _required_bool(payload: dict[str, object], key: str) -> bool:
        value = payload.get(key)
        if not isinstance(value, bool):
            raise InvalidModelBundleError(f"Model metadata field {key!r} must be a boolean")
        return value

    @staticmethod
    def _parse_input_shape(value: object) -> tuple[int, int, int]:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value
            )
        ):
            raise InvalidModelBundleError(
                "Model metadata input_shape must contain 3 positive integers"
            )
        return cast(tuple[int, int, int], tuple(value))

    @staticmethod
    def _default_model_loader(path: str, **kwargs: object) -> Any:
        keras = importlib.import_module("keras")
        return keras.saving.load_model(path, **kwargs)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
