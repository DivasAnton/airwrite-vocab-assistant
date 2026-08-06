import argparse
import json

from app.ml.emnist_training_data_loader import EMNISTTrainingDataLoader
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS
from app.ml.model_artifact_exporter import EMNISTModelArtifactExporter
from app.ml.model_evaluator import ModelEvaluator
from app.utils.config import emnist_training_settings
from scripts.audit_emnist_letters import load_official_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one selected experiment on the untouched official test split"
    )
    parser.add_argument("--experiment", required=True, choices=("E01", "E02"))
    args = parser.parse_args()
    settings = emnist_training_settings
    experiment_root = settings.experiment_root(args.experiment)
    experiment_model = experiment_root / "model.keras"
    metadata_path = experiment_root / "model_metadata.json"
    if not experiment_model.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Train {args.experiment} before official evaluation")

    tensorflow = __import__("tensorflow")
    model = tensorflow.keras.models.load_model(experiment_model, compile=False)
    dataset = load_official_dataset()
    loader = EMNISTTrainingDataLoader(
        batch_size=settings.batch_size,
        random_seed=settings.random_seed,
        shuffle_buffer=settings.shuffle_buffer,
        cache=settings.cache_dataset,
        prefetch=settings.prefetch_dataset,
    )
    test_dataset = loader.build_dataset(
        dataset.official_test.images,
        dataset.official_test.labels,
        training=False,
    )
    result = ModelEvaluator(LETTER_IDENTITY_LABELS).evaluate_dataset(model, test_dataset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "selected_experiment": args.experiment,
            "official_test_evaluated": True,
            "official_test_accuracy": result.test_accuracy,
            "official_test_macro_f1": result.macro_f1,
            "official_test_top_3_accuracy": result.top_3_accuracy,
        }
    )
    preprocessing = json.loads(
        (experiment_root / "preprocessing_config.json").read_text(encoding="utf-8")
    )
    exporter = EMNISTModelArtifactExporter()
    exporter.export_bundle(settings.artifact_root, experiment_model, metadata, preprocessing)
    exporter.export_identity_evaluation(result, settings.artifact_root, "emnist_metrics.json")
    print(
        f"Official test complete: accuracy={result.test_accuracy:.4f}, "
        f"macro_f1={result.macro_f1:.4f}, top3={result.top_3_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
