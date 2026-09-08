import importlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ml.character_model_builder import AugmentationConfig, CharacterModelBuilder
from app.ml.character_model_trainer import CharacterModelTrainer
from app.ml.dataset_manifest import DatasetManifestEntry, read_manifest
from app.ml.exceptions import DatasetLoadError, TrainingDependencyError
from app.ml.labels import CHARACTER_LABELS
from app.ml.model_artifact_exporter import ModelArtifactExporter
from app.ml.model_evaluator import ModelEvaluator
from app.ml.training_data_loader import TrainingDataLoader
from app.utils.config import PROJECT_ROOT, settings, training_settings


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_splits() -> tuple[
    list[DatasetManifestEntry], list[DatasetManifestEntry], list[DatasetManifestEntry]
]:
    entries = read_manifest(training_settings.split_path)
    expected_labels = set(CHARACTER_LABELS)
    result: list[list[DatasetManifestEntry]] = []
    for split_name in ("train", "validation", "test"):
        split_entries = [entry for entry in entries if entry.split == split_name]
        if not split_entries:
            raise DatasetLoadError(f"Split {split_name!r} is empty")
        labels = {entry.label for entry in split_entries}
        if labels != expected_labels:
            missing = expected_labels - labels
            raise DatasetLoadError(
                f"Split {split_name!r} is missing labels: {', '.join(sorted(missing))}"
            )
        result.append(split_entries)
    return result[0], result[1], result[2]


def require_tensorflow() -> Any:
    try:
        return importlib.import_module("tensorflow")
    except ImportError as error:
        raise TrainingDependencyError(
            "TensorFlow is not installed. Run: pip install -r requirements-training.txt"
        ) from error


def main() -> int:
    training_settings.validate()
    training_settings.create_artifact_directories()
    train_entries, validation_entries, test_entries = load_splits()
    loader = TrainingDataLoader(
        batch_size=training_settings.batch_size,
        random_seed=training_settings.random_seed,
        project_root=PROJECT_ROOT,
        images_are_preprocessed=training_settings.dataset_images_preprocessed,
        expected_shape=(training_settings.input_height, training_settings.input_width),
    )
    train_dataset = loader.as_tensorflow_dataset(train_entries, shuffle=True)
    validation_dataset = loader.as_tensorflow_dataset(validation_entries, shuffle=False)
    builder = CharacterModelBuilder(
        augmentation_config=AugmentationConfig(
            enabled=training_settings.enable_augmentation,
            rotation_factor=training_settings.rotation_factor,
            translation_factor=training_settings.translation_factor,
            zoom_factor=training_settings.zoom_factor,
        ),
        learning_rate=training_settings.learning_rate,
    )
    trainer = CharacterModelTrainer(
        builder=builder,
        model_path=training_settings.model_output_path,
        history_path=training_settings.training_history_path,
        input_shape=(
            training_settings.input_height,
            training_settings.input_width,
            training_settings.input_channels,
        ),
        num_classes=training_settings.num_classes,
        max_epochs=training_settings.max_epochs,
        early_stopping_patience=training_settings.early_stopping_patience,
        random_seed=training_settings.random_seed,
    )
    training_result = trainer.train(train_dataset, validation_dataset)

    tensorflow = require_tensorflow()
    best_model = tensorflow.keras.models.load_model(training_settings.model_output_path)
    evaluation = ModelEvaluator().evaluate_model(best_model, loader, test_entries)
    exporter = ModelArtifactExporter(overwrite=True)
    exporter.export_identity_bundle_labels(training_settings.model_output_path.parent)
    exporter.export_labels(training_settings.labels_output_path)
    exporter.export_preprocessing_contract(
        training_settings.preprocessing_config_output_path,
        output_width=training_settings.input_width,
        output_height=training_settings.input_height,
        channels=training_settings.input_channels,
        content_width=settings.preprocess_content_width,
        content_height=settings.preprocess_content_height,
        binary_threshold=settings.preprocess_binary_threshold,
        crop_padding=settings.preprocess_crop_padding,
        min_foreground_pixels=settings.preprocess_min_foreground_pixels,
        invert_input=settings.preprocess_invert_input,
        center_of_mass=settings.preprocess_center_of_mass,
    )
    exporter.export_preprocessing_contract(
        training_settings.model_output_path.parent / "preprocessing_config.json",
        output_width=training_settings.input_width,
        output_height=training_settings.input_height,
        channels=training_settings.input_channels,
        content_width=settings.preprocess_content_width,
        content_height=settings.preprocess_content_height,
        binary_threshold=settings.preprocess_binary_threshold,
        crop_padding=settings.preprocess_crop_padding,
        min_foreground_pixels=settings.preprocess_min_foreground_pixels,
        invert_input=settings.preprocess_invert_input,
        center_of_mass=settings.preprocess_center_of_mass,
    )
    exporter.export_evaluation_reports(
        evaluation,
        metrics_path=training_settings.metrics_output_path,
        classification_report_path=training_settings.classification_report_path,
        confusion_matrix_json_path=training_settings.confusion_matrix_json_path,
        confusion_matrix_image_path=training_settings.confusion_matrix_image_path,
        error_analysis_path=training_settings.error_analysis_path,
        prediction_probabilities_path=training_settings.prediction_probabilities_path,
    )

    timestamp = datetime.now(UTC).isoformat()
    split_counts = Counter(
        entry.split for entry in [*train_entries, *validation_entries, *test_entries]
    )
    metadata = {
        "model_name": training_settings.model_name,
        "model_version": training_settings.model_version,
        "framework": "keras",
        "task_type": "airwrite_custom_identity",
        "case_sensitive": False,
        "case_source": "identity_only",
        "input_shape": [
            training_settings.input_height,
            training_settings.input_width,
            training_settings.input_channels,
        ],
        "num_classes": training_settings.num_classes,
        "random_seed": training_settings.random_seed,
        "dataset_manifest": relative_path(training_settings.split_path),
        "dataset_images_preprocessed": training_settings.dataset_images_preprocessed,
        "dataset_root": relative_path(training_settings.dataset_root),
        "split_counts": dict(split_counts),
        "training_timestamp": timestamp,
        "epochs_completed": training_result.epochs_completed,
        "best_epoch": training_result.best_epoch,
        "validation_loss": training_result.best_validation_loss,
        "validation_accuracy": training_result.best_validation_accuracy,
        "test_accuracy": evaluation.test_accuracy,
        "macro_f1": evaluation.macro_f1,
        "top_3_accuracy": evaluation.top_3_accuracy,
        "augmentation": {
            "enabled": training_settings.enable_augmentation,
            "rotation_factor": training_settings.rotation_factor,
            "translation_factor": training_settings.translation_factor,
            "zoom_factor": training_settings.zoom_factor,
            "horizontal_flip": False,
            "vertical_flip": False,
        },
    }
    exporter.export_metadata(training_settings.model_metadata_path, metadata)
    exporter.export_metadata(
        training_settings.model_output_path.parent / "model_metadata.json", metadata
    )
    exporter.append_experiment(
        training_settings.experiment_log_path,
        {
            "experiment_id": training_settings.experiment_id,
            "model_version": training_settings.model_version,
            "dataset_manifest": relative_path(training_settings.split_path),
            "random_seed": training_settings.random_seed,
            "augmentation": training_settings.enable_augmentation,
            "epochs_completed": training_result.epochs_completed,
            "best_epoch": training_result.best_epoch,
            "validation_accuracy": training_result.best_validation_accuracy,
            "test_accuracy": evaluation.test_accuracy,
            "macro_f1": evaluation.macro_f1,
            "model_path": relative_path(training_settings.model_output_path),
            "timestamp": timestamp,
            "notes": training_settings.experiment_notes,
        },
    )

    print(f"Best epoch: {training_result.best_epoch}")
    print(f"Validation accuracy: {training_result.best_validation_accuracy:.4f}")
    print(f"Test accuracy: {evaluation.test_accuracy:.4f}")
    print(f"Macro F1: {evaluation.macro_f1:.4f}")
    print(f"Top-3 accuracy: {evaluation.top_3_accuracy:.4f}")
    print(f"Model: {training_settings.model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
