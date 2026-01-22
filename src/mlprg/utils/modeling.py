from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class TrainingArtifacts:
    model: Any
    x_test: np.ndarray
    y_test: np.ndarray
    metrics: dict[str, float]
    model_path: Path
    metadata_path: Path
    metrics_path: Path


def train_model(artifact_dir: Path) -> TrainingArtifacts:
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    accuracy = float(model.score(x_test, y_test))
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "framework": "scikit-learn",
        "dataset": "breast_cancer",
        "features": dataset.feature_names.tolist(),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    metadata_path = artifact_dir / "training_metadata.json"
    metrics_path = artifact_dir / "metrics.json"
    joblib.dump(model, model_path)
    metrics = {"accuracy": accuracy}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return TrainingArtifacts(
        model=model,
        x_test=x_test,
        y_test=y_test,
        metrics=metrics,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
    )
