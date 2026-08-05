from app.ml.model_artifact_exporter import ModelArtifactExporter
from app.ml.model_evaluator import ModelEvaluator
from app.ml.training_data_loader import TrainingDataLoader
from app.utils.config import PROJECT_ROOT, training_settings
from scripts.train_character_model import load_splits, require_tensorflow


def main() -> int:
    training_settings.validate()
    if not training_settings.model_output_path.is_file():
        raise FileNotFoundError(
            f"Trained model does not exist: {training_settings.model_output_path}"
        )
    _, _, test_entries = load_splits()
    loader = TrainingDataLoader(
        batch_size=training_settings.batch_size,
        random_seed=training_settings.random_seed,
        project_root=PROJECT_ROOT,
        images_are_preprocessed=training_settings.dataset_images_preprocessed,
        expected_shape=(training_settings.input_height, training_settings.input_width),
    )
    model = require_tensorflow().keras.models.load_model(training_settings.model_output_path)
    evaluation = ModelEvaluator().evaluate_model(model, loader, test_entries)
    ModelArtifactExporter(overwrite=True).export_evaluation_reports(
        evaluation,
        metrics_path=training_settings.metrics_output_path,
        classification_report_path=training_settings.classification_report_path,
        confusion_matrix_json_path=training_settings.confusion_matrix_json_path,
        confusion_matrix_image_path=training_settings.confusion_matrix_image_path,
        error_analysis_path=training_settings.error_analysis_path,
        prediction_probabilities_path=training_settings.prediction_probabilities_path,
    )
    print(f"Test accuracy: {evaluation.test_accuracy:.4f}")
    print(f"Macro F1: {evaluation.macro_f1:.4f}")
    print(f"Top-3 accuracy: {evaluation.top_3_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
