import importlib
from collections import Counter
from typing import Any

import numpy as np

from app.ml.exceptions import TrainingDependencyError
from app.ml.model_artifact_exporter import ModelArtifactExporter
from app.ml.model_evaluator import ModelEvaluator
from app.ml.training_data_loader import TrainingDataLoader
from app.utils.config import PROJECT_ROOT, training_settings
from scripts.train_character_model import load_splits


def require_logistic_regression() -> Any:
    try:
        return importlib.import_module("sklearn.linear_model").LogisticRegression
    except ImportError as error:
        raise TrainingDependencyError(
            "Scikit-learn is not installed. Run: pip install -r requirements-training.txt"
        ) from error


def main() -> int:
    training_settings.validate()
    train_entries, _, test_entries = load_splits()
    loader = TrainingDataLoader(
        batch_size=training_settings.batch_size,
        random_seed=training_settings.random_seed,
        project_root=PROJECT_ROOT,
        images_are_preprocessed=training_settings.dataset_images_preprocessed,
        expected_shape=(training_settings.input_height, training_settings.input_width),
    )
    train_images, train_labels = loader.load_all(train_entries)
    test_images, test_labels = loader.load_all(test_entries)
    classifier = require_logistic_regression()(
        max_iter=1000,
        random_state=training_settings.random_seed,
    )
    classifier.fit(train_images.reshape(len(train_images), -1), train_labels)
    probabilities = np.asarray(
        classifier.predict_proba(test_images.reshape(len(test_images), -1)), dtype=np.float32
    )
    evaluation = ModelEvaluator().evaluate_predictions(test_labels, probabilities)
    majority_label = Counter(train_labels.tolist()).most_common(1)[0][0]
    majority_accuracy = float(np.mean(test_labels == majority_label))
    ModelArtifactExporter(overwrite=True).export_metrics(
        training_settings.baseline_metrics_path,
        {
            "model": "logistic_regression",
            "random_seed": training_settings.random_seed,
            "train_samples": len(train_entries),
            "test_samples": len(test_entries),
            "majority_class_test_accuracy": majority_accuracy,
            **evaluation.metrics_payload(),
            "per_class_metrics": evaluation.per_class_metrics,
        },
    )
    print(f"Baseline test accuracy: {evaluation.test_accuracy:.4f}")
    print(f"Baseline macro F1: {evaluation.macro_f1:.4f}")
    print(f"Metrics: {training_settings.baseline_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
