from __future__ import annotations

import json
from pathlib import Path

from mlprg.pipelines.evaluator import Thresholds
from mlprg.pipelines.pipeline import run_pipeline
from mlprg.registry.sqlite import SqliteRegistry
from mlprg.storage.local import LocalArtifactStore
from mlprg.storage.s3 import S3ArtifactStore
from mlprg.utils.config import settings


def _artifact_store() -> LocalArtifactStore | S3ArtifactStore:
    if settings.artifact_store == "s3":
        return S3ArtifactStore(
            bucket=settings.artifact_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
    return LocalArtifactStore(settings.artifact_local_path)


def main() -> None:
    registry = SqliteRegistry(settings.database_url)
    store = _artifact_store()
    thresholds = Thresholds(
        min_auc=settings.min_auc,
        min_accuracy=settings.min_accuracy,
        max_model_size_mb=settings.max_model_size_mb,
        max_p95_inference_ms=settings.max_p95_inference_ms,
        enable_safety_checks=settings.enable_safety_checks,
    )
    result = run_pipeline(registry=registry, artifact_store=store, thresholds=thresholds)
    output = {
        "version": result.version,
        "passed": result.passed,
        "report": result.report.model_dump(),
    }
    Path("./pipeline_output.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
