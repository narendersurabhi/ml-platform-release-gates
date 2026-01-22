from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from mlprg.pipelines.evaluator import Thresholds
from mlprg.pipelines.pipeline import run_pipeline
from mlprg.registry.sqlite import SqliteRegistry
from mlprg.storage.local import LocalArtifactStore
from mlprg.utils.config import settings


def test_api_predict_and_model(tmp_path: Path) -> None:
    settings.database_url = f"sqlite:///{tmp_path / 'registry.db'}"
    settings.artifact_store = "local"
    settings.artifact_local_path = str(tmp_path / "artifacts")
    settings.api_key = "test-key"

    registry = SqliteRegistry(settings.database_url)
    store = LocalArtifactStore(settings.artifact_local_path)
    thresholds = Thresholds(
        min_auc=0.5,
        min_accuracy=0.5,
        max_model_size_mb=50.0,
        max_p95_inference_ms=1000.0,
        enable_safety_checks=False,
    )
    run_pipeline(
        registry=registry,
        artifact_store=store,
        thresholds=thresholds,
        output_dir=str(tmp_path / "runs"),
    )

    app_module = importlib.import_module("mlprg.api.app")
    importlib.reload(app_module)
    app = app_module.create_app()
    client = TestClient(app)

    response = client.get("/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PRODUCTION"

    predict = client.post("/predict", json={"features": [0.1] * 30})
    assert predict.status_code == 200
    assert "prediction" in predict.json()
