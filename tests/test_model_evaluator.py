import numpy as np

from app.ml.model_evaluator import ModelEvaluator


def test_evaluator_computes_perfect_multiclass_metrics() -> None:
    y_true = np.arange(26, dtype=np.int64)
    probabilities = np.eye(26, dtype=np.float32)

    result = ModelEvaluator().evaluate_predictions(y_true, probabilities, test_loss=0.0)

    assert result.test_accuracy == 1.0
    assert result.macro_precision == 1.0
    assert result.macro_recall == 1.0
    assert result.macro_f1 == 1.0
    assert result.top_3_accuracy == 1.0
    assert result.confusion_matrix == np.eye(26, dtype=int).tolist()


def test_top_three_accuracy_can_exceed_top_one_accuracy() -> None:
    y_true = np.array([0], dtype=np.int64)
    probabilities = np.zeros((1, 26), dtype=np.float32)
    probabilities[0, 0] = 0.35
    probabilities[0, 1] = 0.40
    probabilities[0, 2] = 0.25

    result = ModelEvaluator().evaluate_predictions(y_true, probabilities)

    assert result.test_accuracy == 0.0
    assert result.top_3_accuracy == 1.0
