import argparse
from datetime import UTC, datetime

import numpy as np

from app.ml.character_model_builder import AugmentationConfig, CharacterModelBuilder
from app.ml.character_model_trainer import EMNISTModelTrainer
from app.ml.emnist_letters_dataset import EMNISTDatasetSplit
from app.ml.emnist_training_data_loader import EMNISTTrainingDataLoader
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS
from app.ml.model_artifact_exporter import EMNISTModelArtifactExporter
from app.ml.model_evaluator import ModelEvaluator
from app.utils.config import emnist_training_settings, preprocessing_alignment_settings
from scripts.audit_emnist_letters import ensure_split, load_official_dataset


def preprocessing_payload() -> dict[str, object]:
    contract = preprocessing_alignment_settings
    return {
        "version": 1,
        "input_width": contract.input_width,
        "input_height": contract.input_height,
        "channels": contract.input_channels,
        "orientation_transform": "transpose",
        "intensity_inversion": False,
        "normalization_divisor": contract.normalization_divisor,
        "background": "black",
        "foreground": "light",
    }


def run_smoke_test() -> None:
    settings = emnist_training_settings
    dataset = load_official_dataset()
    selected_parts = [
        np.flatnonzero(dataset.official_train.labels == class_index)[:2]
        for class_index in range(26)
    ]
    indices = np.concatenate(selected_parts)
    tiny = EMNISTDatasetSplit(
        images=dataset.official_train.images[indices],
        labels=dataset.official_train.labels[indices],
        split_name="tiny_overfit",
    )
    loader = EMNISTTrainingDataLoader(
        batch_size=len(indices),
        shuffle_buffer=len(indices),
        random_seed=settings.random_seed,
        prefetch=False,
    )
    tiny_train_dataset = loader.build_dataset(tiny.images, tiny.labels, training=True)
    tiny_eval_dataset = loader.build_dataset(tiny.images, tiny.labels, training=False)
    root = settings.experiment_root("SMOKE")
    trainer = EMNISTModelTrainer(
        builder=CharacterModelBuilder(
            augmentation_config=AugmentationConfig(enabled=False),
            learning_rate=0.003,
            # A one-batch overfit check cannot establish reliable BatchNorm moving statistics.
            use_batch_normalization=False,
            dense_units=128,
            dropout_rate=0.0,
            model_name="airwrite_emnist_tiny_overfit",
        ),
        experiment_id="SMOKE",
        checkpoint_path=root / "checkpoints" / "best.keras",
        history_path=root / "training_history.csv",
        max_epochs=120,
        early_stopping_patience=120,
        reduce_lr_patience=20,
        random_seed=settings.random_seed,
    )
    model, result = trainer.train(tiny_train_dataset, tiny_eval_dataset, verbose=0)
    accuracy = float(model.evaluate(tiny_eval_dataset, verbose=0)[1])
    reloaded = __import__("tensorflow").keras.models.load_model(
        result.checkpoint_path, compile=False
    )
    batch_images, _ = next(iter(tiny_eval_dataset))
    difference = float(
        np.max(
            np.abs(
                model.predict(batch_images, verbose=0) - reloaded.predict(batch_images, verbose=0)
            )
        )
    )
    if accuracy < 0.95:
        raise RuntimeError(f"Tiny-overfit test failed: training accuracy={accuracy:.4f}")
    if difference > 1e-5:
        raise RuntimeError(f"Reload test failed: maximum prediction difference={difference}")
    print(f"Tiny-overfit passed: accuracy={accuracy:.4f}, reload_max_diff={difference:.8f}")


def train_experiment(experiment_id: str, augmentation_mode: str) -> None:
    settings = emnist_training_settings
    settings.validate()
    dataset = load_official_dataset()
    indices = ensure_split(dataset)
    loader = EMNISTTrainingDataLoader(
        batch_size=settings.batch_size,
        shuffle_buffer=settings.shuffle_buffer,
        random_seed=settings.random_seed,
        cache=settings.cache_dataset,
        prefetch=settings.prefetch_dataset,
    )
    datasets = loader.build_bundle(
        dataset.official_train,
        dataset.official_test,
        indices.train_indices,
        indices.validation_indices,
    )
    augmentation = AugmentationConfig(
        enabled=augmentation_mode == "light",
        rotation_factor=settings.rotation_factor,
        translation_factor=settings.translation_factor,
        zoom_factor=settings.zoom_factor,
    )
    root = settings.experiment_root(experiment_id)
    trainer = EMNISTModelTrainer(
        builder=CharacterModelBuilder(
            augmentation_config=augmentation,
            learning_rate=settings.learning_rate,
            use_batch_normalization=True,
            dense_units=128,
            dropout_rate=settings.dropout_rate,
            model_name="airwrite_emnist_letter_identity",
        ),
        experiment_id=experiment_id,
        checkpoint_path=root / "checkpoints" / "best.keras",
        history_path=root / "training_history.csv",
        max_epochs=settings.max_epochs,
        early_stopping_patience=settings.early_stopping_patience,
        reduce_lr_patience=settings.reduce_lr_patience,
        random_seed=settings.random_seed,
    )
    model, result = trainer.train(datasets.train, datasets.validation)
    validation_result = ModelEvaluator(LETTER_IDENTITY_LABELS).evaluate_dataset(
        model, datasets.validation
    )
    metadata = {
        "model_name": "airwrite_emnist_letter_identity",
        "model_version": settings.model_version,
        "task_type": "letter_identity_classification",
        "dataset": "EMNIST Letters official IDX",
        "num_classes": 26,
        "case_sensitive": False,
        "case_source": "user_selected_mode",
        "input_shape": [28, 28, 1],
        "output_labels": "identity_labels.json",
        "experiment_id": experiment_id,
        "augmentation": augmentation_mode,
        "random_seed": settings.random_seed,
        "best_epoch": result.best_epoch,
        "validation_loss": result.best_validation_loss,
        "validation_accuracy": result.best_validation_accuracy,
        "validation_top_3_accuracy": result.best_validation_top_3_accuracy,
        "validation_macro_f1": validation_result.macro_f1,
        "trained_at": datetime.now(UTC).isoformat(),
        "official_test_evaluated": False,
    }
    exporter = EMNISTModelArtifactExporter()
    exporter.export_bundle(
        root,
        result.checkpoint_path,
        metadata,
        preprocessing_payload(),
    )
    exporter.export_identity_evaluation(
        validation_result,
        root,
        "validation_metrics.json",
    )
    print(
        f"{experiment_id} complete: best_epoch={result.best_epoch}, "
        f"val_accuracy={result.best_validation_accuracy:.4f}, "
        f"val_top3={result.best_validation_top_3_accuracy:.4f}, "
        f"val_macro_f1={validation_result.macro_f1:.4f}"
    )
    print("Official test was not evaluated. Compare experiments using validation only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 26-class EMNIST identity model")
    parser.add_argument("--experiment", choices=("E01", "E02"))
    parser.add_argument("--augmentation", choices=("off", "light"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --smoke-test is used")
    mode = args.augmentation or ("off" if args.experiment == "E01" else "light")
    train_experiment(args.experiment, mode)


if __name__ == "__main__":
    main()
