import csv
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.exceptions import ArtifactExportError
from app.ml.labels import CHARACTER_LABELS, labels_payload
from app.ml.letter_identity_labels import (
    LETTER_IDENTITY_LABELS,
    display_labels_payload,
    identity_labels_payload,
)
from app.ml.model_evaluator import EvaluationResult


class ModelArtifactExporter:
    def __init__(self, overwrite: bool = True) -> None:
        self.overwrite = overwrite

    def export_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            serialized = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        except (TypeError, ValueError) as error:
            raise ArtifactExportError(f"Artifact payload is not valid JSON: {path}") from error
        if path.exists() and not self.overwrite:
            raise ArtifactExportError(f"Artifact already exists and overwrite is disabled: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")

    def export_labels(self, labels_output_path: Path) -> None:
        self.export_json(labels_output_path, labels_payload())

    def export_identity_bundle_labels(self, artifact_root: Path) -> None:
        self.export_json(artifact_root / "identity_labels.json", identity_labels_payload())
        self.export_json(
            artifact_root / "lowercase_display_labels.json",
            display_labels_payload(uppercase=False),
        )
        self.export_json(
            artifact_root / "uppercase_display_labels.json",
            display_labels_payload(uppercase=True),
        )

    def export_preprocessing_contract(
        self,
        path: Path,
        output_width: int,
        output_height: int,
        channels: int,
        content_width: int,
        content_height: int,
        binary_threshold: int,
        crop_padding: int,
        min_foreground_pixels: int,
        invert_input: bool,
        center_of_mass: bool,
    ) -> None:
        self.export_json(
            path,
            {
                "version": 1,
                "output_width": output_width,
                "output_height": output_height,
                "input_width": output_width,
                "input_height": output_height,
                "channels": channels,
                "orientation_transform": "none",
                "intensity_inversion": invert_input,
                "background": "black",
                "foreground": "light",
                "normalized_min": 0.0,
                "normalized_max": 1.0,
                "content_width": content_width,
                "content_height": content_height,
                "binary_threshold": binary_threshold,
                "crop_padding": crop_padding,
                "min_foreground_pixels": min_foreground_pixels,
                "invert_input": invert_input,
                "center_of_mass": center_of_mass,
            },
        )

    def export_metadata(self, path: Path, metadata: dict[str, Any]) -> None:
        required = {
            "model_name",
            "model_version",
            "framework",
            "input_shape",
            "num_classes",
            "random_seed",
            "dataset_manifest",
            "training_timestamp",
            "best_epoch",
            "validation_accuracy",
            "test_accuracy",
        }
        missing = required - metadata.keys()
        if missing:
            raise ArtifactExportError(
                f"Model metadata is missing required fields: {', '.join(sorted(missing))}"
            )
        self.export_json(path, metadata)

    def export_metrics(self, path: Path, metrics: dict[str, Any]) -> None:
        self.export_json(path, metrics)

    def export_evaluation_reports(
        self,
        result: EvaluationResult,
        metrics_path: Path,
        classification_report_path: Path,
        confusion_matrix_json_path: Path,
        confusion_matrix_image_path: Path,
        error_analysis_path: Path,
        prediction_probabilities_path: Path,
    ) -> None:
        self.export_metrics(metrics_path, result.metrics_payload())
        self.export_json(
            classification_report_path,
            {"labels": list(CHARACTER_LABELS), "per_class": result.per_class_metrics},
        )
        self.export_json(
            confusion_matrix_json_path,
            {"labels": list(CHARACTER_LABELS), "matrix": result.confusion_matrix},
        )
        self._export_confusion_matrix_image(result, confusion_matrix_image_path)
        self._export_error_analysis(result, error_analysis_path)
        self._export_prediction_probabilities(result, prediction_probabilities_path)

    def append_experiment(self, path: Path, row: dict[str, object]) -> None:
        fieldnames = (
            "experiment_id",
            "model_version",
            "dataset_manifest",
            "random_seed",
            "augmentation",
            "epochs_completed",
            "best_epoch",
            "validation_accuracy",
            "test_accuracy",
            "macro_f1",
            "model_path",
            "timestamp",
            "notes",
        )
        missing = set(fieldnames) - row.keys()
        if missing:
            raise ArtifactExportError(
                f"Experiment row is missing fields: {', '.join(sorted(missing))}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row[field] for field in fieldnames})

    def _export_confusion_matrix_image(self, result: EvaluationResult, output_path: Path) -> None:
        try:
            pyplot = importlib.import_module("matplotlib.pyplot")
        except ImportError as error:
            raise ArtifactExportError(
                "Matplotlib is required to export the confusion matrix image"
            ) from error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = pyplot.subplots(figsize=(12, 10))
        image = axes.imshow(np.asarray(result.confusion_matrix), cmap="Blues")
        axes.set_xticks(range(len(CHARACTER_LABELS)), CHARACTER_LABELS)
        axes.set_yticks(range(len(CHARACTER_LABELS)), CHARACTER_LABELS)
        axes.set_xlabel("Predicted label")
        axes.set_ylabel("True label")
        axes.set_title("AirWrite Character Confusion Matrix")
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        pyplot.close(figure)

    @staticmethod
    def _export_error_analysis(result: EvaluationResult, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "top_3",
            "source",
        )
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in result.error_analysis:
                serialized_row = dict(row)
                serialized_row["top_3"] = json.dumps(row["top_3"], ensure_ascii=True)
                writer.writerow(serialized_row)

    @staticmethod
    def _export_prediction_probabilities(result: EvaluationResult, output_path: Path) -> None:
        if not result.probabilities:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        probability_columns = [f"probability_{label}" for label in CHARACTER_LABELS]
        fieldnames = [
            "image_path",
            "source",
            "true_label",
            "predicted_label",
            "confidence",
            *probability_columns,
        ]
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, row_probabilities in enumerate(result.probabilities):
                predicted_index = int(np.argmax(row_probabilities))
                row: dict[str, object] = {
                    "image_path": result.image_paths[index],
                    "source": result.sources[index],
                    "true_label": CHARACTER_LABELS[result.true_indices[index]],
                    "predicted_label": CHARACTER_LABELS[predicted_index],
                    "confidence": row_probabilities[predicted_index],
                }
                row.update(
                    {
                        column: row_probabilities[class_index]
                        for class_index, column in enumerate(probability_columns)
                    }
                )
                writer.writerow(row)


class EMNISTModelArtifactExporter(ModelArtifactExporter):
    def export_bundle(
        self,
        artifact_root: Path,
        checkpoint_path: Path,
        metadata: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> Path:
        if metadata.get("num_classes") != 26 or metadata.get("case_sensitive") is not False:
            raise ArtifactExportError(
                "EMNIST identity metadata requires num_classes=26 and case_sensitive=false"
            )
        artifact_root.mkdir(parents=True, exist_ok=True)
        model_path = artifact_root / "model.keras"
        if model_path.exists() and not self.overwrite:
            raise ArtifactExportError(f"Artifact already exists: {model_path}")
        if checkpoint_path.resolve() != model_path.resolve():
            shutil.copy2(checkpoint_path, model_path)
        self.export_json(artifact_root / "identity_labels.json", identity_labels_payload())
        self.export_json(
            artifact_root / "uppercase_display_labels.json",
            display_labels_payload(uppercase=True),
        )
        self.export_json(
            artifact_root / "lowercase_display_labels.json",
            display_labels_payload(uppercase=False),
        )
        self.export_json(artifact_root / "model_metadata.json", metadata)
        self.export_json(artifact_root / "preprocessing_config.json", preprocessing)
        return model_path

    def export_identity_evaluation(
        self,
        result: EvaluationResult,
        artifact_root: Path,
        metrics_name: str,
        report_prefix: str = "",
    ) -> None:
        self.export_json(artifact_root / metrics_name, result.metrics_payload())
        self.export_json(
            artifact_root / f"{report_prefix}classification_report.json",
            {"labels": list(LETTER_IDENTITY_LABELS), "per_class": result.per_class_metrics},
        )
        self.export_json(
            artifact_root / f"{report_prefix}confusion_matrix.json",
            {"labels": list(LETTER_IDENTITY_LABELS), "matrix": result.confusion_matrix},
        )
        self._export_identity_confusion_image(
            result, artifact_root / f"{report_prefix}confusion_matrix.png"
        )
        self._export_identity_errors(result, artifact_root / f"{report_prefix}error_analysis.csv")
        self._export_identity_predictions(
            result, artifact_root / f"{report_prefix}prediction_records.csv"
        )

    @staticmethod
    def _export_identity_confusion_image(result: EvaluationResult, output_path: Path) -> None:
        try:
            pyplot = importlib.import_module("matplotlib.pyplot")
        except ImportError as error:
            raise ArtifactExportError(
                "Matplotlib is required to export the confusion matrix image"
            ) from error
        figure, axes = pyplot.subplots(figsize=(12, 10))
        image = axes.imshow(np.asarray(result.confusion_matrix), cmap="Blues")
        axes.set_xticks(range(26), LETTER_IDENTITY_LABELS)
        axes.set_yticks(range(26), LETTER_IDENTITY_LABELS)
        axes.set_xlabel("Predicted identity")
        axes.set_ylabel("True identity")
        axes.set_title("EMNIST Letter Identity Confusion Matrix")
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        pyplot.close(figure)

    @staticmethod
    def _export_identity_errors(result: EvaluationResult, output_path: Path) -> None:
        fieldnames = (
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "top_3",
            "source",
            "session_id",
        )
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for error in result.error_analysis:
                row = dict(error)
                row["top_3"] = json.dumps(row["top_3"], ensure_ascii=True)
                writer.writerow(row)

    @staticmethod
    def _export_identity_predictions(result: EvaluationResult, output_path: Path) -> None:
        if not result.probabilities:
            return
        fieldnames = (
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "top_3",
            "source",
        )
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, probabilities in enumerate(result.probabilities):
                predicted_index = int(np.argmax(probabilities))
                top_indices = np.argsort(probabilities)[-3:][::-1]
                writer.writerow(
                    {
                        "image_path": (
                            result.image_paths[index]
                            if index < len(result.image_paths)
                            else f"dataset_sample:{index}"
                        ),
                        "true_label": LETTER_IDENTITY_LABELS[result.true_indices[index]],
                        "predicted_label": LETTER_IDENTITY_LABELS[predicted_index],
                        "confidence": probabilities[predicted_index],
                        "top_3": json.dumps(
                            [
                                {
                                    "label": LETTER_IDENTITY_LABELS[int(class_index)],
                                    "probability": probabilities[int(class_index)],
                                }
                                for class_index in top_indices
                            ],
                            ensure_ascii=True,
                        ),
                        "source": (
                            result.sources[index] if index < len(result.sources) else "dataset"
                        ),
                    }
                )
