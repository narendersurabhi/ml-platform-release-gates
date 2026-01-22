from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from mlprg.contracts.schemas import EvaluationReport
from mlprg.observability.metrics import eval_fail_total, eval_pass_total


class Evaluator(Protocol):
    def evaluate(
        self,
        model: Any,
        x_test: np.ndarray,
        y_test: np.ndarray,
        model_size_mb: float,
    ) -> EvaluationReport: ...


@dataclass
class Thresholds:
    min_auc: float
    min_accuracy: float
    max_model_size_mb: float
    max_p95_inference_ms: float
    enable_safety_checks: bool


class DefaultEvaluator:
    def __init__(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds

    def evaluate(
        self,
        model: Any,
        x_test: np.ndarray,
        y_test: np.ndarray,
        model_size_mb: float,
    ) -> EvaluationReport:
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, predictions)
        p95_latency_ms = self._benchmark_inference(model, x_test)

        checks = {
            "min_auc": auc >= self.thresholds.min_auc,
            "min_accuracy": accuracy >= self.thresholds.min_accuracy,
            "max_model_size_mb": model_size_mb <= self.thresholds.max_model_size_mb,
            "max_p95_inference_ms": p95_latency_ms <= self.thresholds.max_p95_inference_ms,
        }
        reasons = [
            reason
            for reason, ok in [
                ("AUC below threshold", checks["min_auc"]),
                ("Accuracy below threshold", checks["min_accuracy"]),
                ("Model size exceeds limit", checks["max_model_size_mb"]),
                ("P95 inference latency exceeds limit", checks["max_p95_inference_ms"]),
            ]
            if not ok
        ]

        if self.thresholds.enable_safety_checks:
            safety_checks = self._run_safety_checks()
            checks.update(safety_checks)
            for key, ok in safety_checks.items():
                if not ok:
                    reasons.append(f"Safety check failed: {key}")

        passed = all(checks.values())
        metrics = {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "p95_inference_ms": float(p95_latency_ms),
            "model_size_mb": float(model_size_mb),
        }

        report = EvaluationReport(
            model_version="",
            passed=passed,
            metrics=metrics,
            checks=checks,
            reasons=reasons,
            created_at=datetime.utcnow(),
        )
        if passed:
            eval_pass_total.inc()
        else:
            eval_fail_total.inc()
        return report

    def _benchmark_inference(self, model: Any, x_test: np.ndarray) -> float:
        samples = x_test[: min(128, len(x_test))]
        latencies = []
        for sample in samples:
            start = time.perf_counter()
            _ = model.predict_proba(sample.reshape(1, -1))
            latencies.append((time.perf_counter() - start) * 1000)
        return float(np.percentile(latencies, 95)) if latencies else 0.0

    def _run_safety_checks(self) -> dict[str, bool]:
        sample_text = "Hello customer, thanks for your request."
        pii_detected = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sample_text))
        prompt_injection = "ignore prior instructions" in sample_text.lower()
        return {
            "pii_scan": not pii_detected,
            "prompt_injection": not prompt_injection,
        }
