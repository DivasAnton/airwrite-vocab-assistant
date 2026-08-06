import argparse
import json

from app.ml.airwrite_identity_evaluator import AirWriteIdentityEvaluator
from app.ml.model_artifact_exporter import EMNISTModelArtifactExporter
from app.utils.config import emnist_training_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected model on independent AirWrite data"
    )
    parser.add_argument("--experiment", required=True, choices=("E01", "E02"))
    parser.add_argument(
        "--raw-images",
        action="store_true",
        help="Apply full AirWrite preprocessing to raw canvas images",
    )
    args = parser.parse_args()
    settings = emnist_training_settings
    model_path = settings.experiment_root(args.experiment) / "model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Train {args.experiment} before AirWrite evaluation")
    tensorflow = __import__("tensorflow")
    model = tensorflow.keras.models.load_model(model_path, compile=False)
    evaluator = AirWriteIdentityEvaluator(images_preprocessed=not args.raw_images)
    samples = evaluator.load_manifest(settings.airwrite_eval_manifest)
    result = evaluator.evaluate(model, samples)
    exporter = EMNISTModelArtifactExporter()
    exporter.export_identity_evaluation(
        result,
        settings.artifact_root,
        "airwrite_metrics.json",
        report_prefix="airwrite_",
    )
    metadata_path = settings.artifact_root / "model_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["airwrite_evaluation"] = {
            "manifest": str(settings.airwrite_eval_manifest),
            "sample_count": result.sample_count,
            "accuracy": result.test_accuracy,
            "macro_f1": result.macro_f1,
            "top_3_accuracy": result.top_3_accuracy,
            "writing_style_metrics": result.source_metrics,
        }
        exporter.export_json(metadata_path, metadata)
    print(
        f"AirWrite evaluation complete: accuracy={result.test_accuracy:.4f}, "
        f"macro_f1={result.macro_f1:.4f}, top3={result.top_3_accuracy:.4f}"
    )
    for style, metrics in result.source_metrics.items():
        print(
            f"{style}: accuracy={metrics['accuracy']:.4f}, samples={int(metrics['sample_count'])}"
        )


if __name__ == "__main__":
    main()
