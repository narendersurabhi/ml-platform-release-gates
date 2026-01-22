from __future__ import annotations

from pathlib import Path

from mlprg.pipelines.evaluator import Thresholds
from mlprg.pipelines.pipeline import run_pipeline
from mlprg.registry.sqlite import SqliteRegistry
from mlprg.storage.local import LocalArtifactStore


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    registry = SqliteRegistry(f"sqlite:///{db_path}")
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    thresholds = Thresholds(
        min_auc=0.5,
        min_accuracy=0.5,
        max_model_size_mb=50.0,
        max_p95_inference_ms=1000.0,
        enable_safety_checks=False,
    )
    result = run_pipeline(
        registry=registry,
        artifact_store=store,
        thresholds=thresholds,
        output_dir=str(tmp_path / "runs"),
    )
    assert result.passed is True
    model = registry.get_latest_by_status("breast_cancer_classifier", "PRODUCTION")
    assert model is not None
    assert model.version == result.version
