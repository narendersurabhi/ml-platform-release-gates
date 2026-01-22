from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlprg.pipelines.evaluator import DefaultEvaluator, Thresholds


def test_evaluator_thresholds() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(x, y)
    thresholds = Thresholds(
        min_auc=0.99,
        min_accuracy=0.99,
        max_model_size_mb=0.0001,
        max_p95_inference_ms=0.0001,
        enable_safety_checks=False,
    )
    evaluator = DefaultEvaluator(thresholds)
    report = evaluator.evaluate(model, x, y, model_size_mb=1.0)
    assert report.passed is False
    assert "AUC below threshold" in report.reasons or "Accuracy below threshold" in report.reasons
    assert report.checks["max_model_size_mb"] is False
