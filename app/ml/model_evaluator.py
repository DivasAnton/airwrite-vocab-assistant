from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.exceptions import ModelEvaluationError
from app.ml.labels import CHARACTER_LABELS, index_to_label
from app.ml.training_data_loader import TrainingDataLoader


@dataclass(frozen=True)
class EvaluationResult:
    test_loss: float | None
    test_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    top_3_accuracy: float
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    sample_count: int = 0
    source_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    error_analysis: list[dict[str, object]] = field(default_factory=list)
    probabilities: list[list[float]] = field(default_factory=list)
    true_indices: list[int] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def metrics_payload(self) -> dict[str, object]:
        return {
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "top_3_accuracy": self.top_3_accuracy,
            "test_sample_count": self.sample_count,
            "source_metrics": self.source_metrics,
        }


class ModelEvaluator:
    def evaluate_model(
        self,
        model: object,
        loader: TrainingDataLoader,
        entries: list[DatasetManifestEntry],
    ) -> EvaluationResult:
        if not entries:
            raise ModelEvaluationError("Test entries must not be empty")

        probability_batches: list[NDArray[np.float32]] = []
        true_batches: list[NDArray[np.int64]] = []
        ordered_entries: list[DatasetManifestEntry] = []
        predict_on_batch = getattr(model, "predict_on_batch", None)
        if not callable(predict_on_batch):
            raise ModelEvaluationError("Model must provide a callable predict_on_batch method")

        for batch_entries, images, labels in loader.iter_entry_batches(entries, shuffle=False):
            raw_probabilities = np.asarray(predict_on_batch(images), dtype=np.float32)
            probability_batches.append(raw_probabilities)
            true_batches.append(labels)
            ordered_entries.extend(batch_entries)

        probabilities = np.concatenate(probability_batches, axis=0)
        y_true = np.concatenate(true_batches, axis=0)
        clipped = np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-7, 1.0)
        test_loss = float(-np.log(clipped).mean())
        result = self.evaluate_predictions(y_true, probabilities, test_loss=test_loss)
        y_pred = np.argmax(probabilities, axis=1)
        source_metrics = self._source_metrics(ordered_entries, y_true, y_pred)
        error_analysis = self._error_analysis(ordered_entries, y_true, probabilities)
        return replace(
            result,
            source_metrics=source_metrics,
            error_analysis=error_analysis,
            probabilities=probabilities.astype(float).tolist(),
            true_indices=y_true.astype(int).tolist(),
            image_paths=[entry.image_path.as_posix() for entry in ordered_entries],
            sources=[entry.source for entry in ordered_entries],
        )

    def evaluate_predictions(
        self,
        y_true: NDArray[np.int64],
        probabilities: NDArray[np.float32],
        test_loss: float | None = None,
    ) -> EvaluationResult:
        self._validate_predictions(y_true, probabilities)
        y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
        confusion = self.confusion_matrix(y_true, y_pred, len(CHARACTER_LABELS))
        per_class = self.per_class_metrics(confusion)
        precision_values = [metrics["precision"] for metrics in per_class.values()]
        recall_values = [metrics["recall"] for metrics in per_class.values()]
        f1_values = [metrics["f1"] for metrics in per_class.values()]
        return EvaluationResult(
            test_loss=test_loss,
            test_accuracy=float(np.mean(y_pred == y_true)),
            macro_precision=float(np.mean(precision_values)),
            macro_recall=float(np.mean(recall_values)),
            macro_f1=float(np.mean(f1_values)),
            top_3_accuracy=self.top_k_accuracy(y_true, probabilities, k=3),
            sample_count=len(y_true),
            per_class_metrics=per_class,
            confusion_matrix=confusion.astype(int).tolist(),
        )

    @staticmethod
    def confusion_matrix(
        y_true: NDArray[np.int64], y_pred: NDArray[np.int64], num_classes: int
    ) -> NDArray[np.int64]:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_index, predicted_index in zip(y_true, y_pred, strict=True):
            confusion[int(true_index), int(predicted_index)] += 1
        return confusion

    @staticmethod
    def per_class_metrics(
        confusion: NDArray[np.int64],
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for index in range(confusion.shape[0]):
            true_positive = int(confusion[index, index])
            false_positive = int(confusion[:, index].sum()) - true_positive
            false_negative = int(confusion[index, :].sum()) - true_positive
            support = int(confusion[index, :].sum())
            precision = ModelEvaluator._safe_divide(true_positive, true_positive + false_positive)
            recall = ModelEvaluator._safe_divide(true_positive, true_positive + false_negative)
            f1 = ModelEvaluator._safe_divide(2.0 * precision * recall, precision + recall)
            metrics[index_to_label(index)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(support),
            }
        return metrics

    @staticmethod
    def top_k_accuracy(
        y_true: NDArray[np.int64], probabilities: NDArray[np.float32], k: int = 3
    ) -> float:
        if k <= 0 or k > probabilities.shape[1]:
            raise ValueError(f"k must be between 1 and {probabilities.shape[1]}, got {k}")
        top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
        return float(np.mean(np.any(top_k == y_true[:, np.newaxis], axis=1)))

    @staticmethod
    def _source_metrics(
        entries: list[DatasetManifestEntry],
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.int64],
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for source in sorted({entry.source for entry in entries}):
            indices = [index for index, entry in enumerate(entries) if entry.source == source]
            source_true = y_true[indices]
            source_pred = y_pred[indices]
            metrics[source] = {
                "accuracy": float(np.mean(source_true == source_pred)),
                "sample_count": float(len(indices)),
            }
        return metrics

    @staticmethod
    def _error_analysis(
        entries: list[DatasetManifestEntry],
        y_true: NDArray[np.int64],
        probabilities: NDArray[np.float32],
    ) -> list[dict[str, object]]:
        y_pred = np.argmax(probabilities, axis=1)
        errors: list[dict[str, object]] = []
        for index, entry in enumerate(entries):
            if int(y_pred[index]) == int(y_true[index]):
                continue
            top_indices = np.argsort(probabilities[index])[-3:][::-1]
            errors.append(
                {
                    "image_path": entry.image_path.as_posix(),
                    "true_label": index_to_label(int(y_true[index])),
                    "predicted_label": index_to_label(int(y_pred[index])),
                    "confidence": float(probabilities[index, y_pred[index]]),
                    "top_3": [
                        {
                            "label": index_to_label(int(class_index)),
                            "probability": float(probabilities[index, class_index]),
                        }
                        for class_index in top_indices
                    ],
                    "source": entry.source,
                }
            )
        return errors

    @staticmethod
    def _validate_predictions(
        y_true: NDArray[np.int64], probabilities: NDArray[np.float32]
    ) -> None:
        if y_true.ndim != 1 or len(y_true) == 0:
            raise ModelEvaluationError("y_true must be a non-empty 1D array")
        expected_shape = (len(y_true), len(CHARACTER_LABELS))
        if probabilities.shape != expected_shape:
            raise ModelEvaluationError(
                f"Expected probabilities shape {expected_shape}, got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ModelEvaluationError("Probabilities contain NaN or Inf")
        if np.any(y_true < 0) or np.any(y_true >= len(CHARACTER_LABELS)):
            raise ModelEvaluationError("y_true contains an invalid class index")

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0
