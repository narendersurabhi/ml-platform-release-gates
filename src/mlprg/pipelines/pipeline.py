from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from opentelemetry import trace

from mlprg.contracts.schemas import EvaluationReport
from mlprg.observability.tracing import configure_tracing
from mlprg.pipelines.evaluator import DefaultEvaluator, Thresholds
from mlprg.registry.base import ArtifactRecord, EvaluationRecord, Registry
from mlprg.storage.base import ArtifactStore
from mlprg.utils.config import settings
from mlprg.utils.modeling import train_model


@dataclass
class PipelineResult:
    version: str
    passed: bool
    report: EvaluationReport


def run_pipeline(
    registry: Registry,
    artifact_store: ArtifactStore,
    thresholds: Thresholds | None = None,
    output_dir: str = "./runs",
) -> PipelineResult:
    configure_tracing("mlprg-pipeline")
    tracer = trace.get_tracer(__name__)
    thresholds = thresholds or Thresholds(
        min_auc=settings.min_auc,
        min_accuracy=settings.min_accuracy,
        max_model_size_mb=settings.max_model_size_mb,
        max_p95_inference_ms=settings.max_p95_inference_ms,
        enable_safety_checks=settings.enable_safety_checks,
    )
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_name = settings.model_name
    run_dir = Path(output_dir) / version

    with tracer.start_as_current_span("train"):
        artifacts = train_model(run_dir)
        registry.register_model(
            name=model_name,
            version=version,
            metadata={"training_accuracy": artifacts.metrics["accuracy"]},
        )

    with tracer.start_as_current_span("store-artifacts"):
        model_artifact = artifact_store.put(str(artifacts.model_path), f"{version}/model.pkl")
        metrics_artifact = artifact_store.put(
            str(artifacts.metrics_path), f"{version}/metrics.json"
        )
        metadata_artifact = artifact_store.put(
            str(artifacts.metadata_path), f"{version}/training_metadata.json"
        )
        for artifact in [model_artifact, metrics_artifact, metadata_artifact]:
            registry.add_artifact(
                ArtifactRecord(
                    model_version=version,
                    uri=artifact.uri,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                )
            )

    with tracer.start_as_current_span("evaluate"):
        evaluator = DefaultEvaluator(thresholds)
        model_size_mb = artifacts.model_path.stat().st_size / (1024 * 1024)
        report = evaluator.evaluate(
            artifacts.model,
            artifacts.x_test,
            artifacts.y_test,
            model_size_mb,
        )
        report = report.model_copy(update={"model_version": version})
        registry.add_evaluation(
            EvaluationRecord(
                model_version=version,
                passed=report.passed,
                report=report.model_dump(),
                created_at=report.created_at,
            )
        )

    with tracer.start_as_current_span("promotion"):
        if report.passed:
            registry.update_status(model_name, version, "STAGING")
            registry.update_status(model_name, version, "PRODUCTION")
        else:
            registry.update_status(model_name, version, "REJECTED")

    return PipelineResult(version=version, passed=report.passed, report=report)
