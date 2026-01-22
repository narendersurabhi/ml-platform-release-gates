from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ModelRecord:
    name: str
    version: str
    status: str
    created_at: datetime
    metadata: dict[str, Any]


@dataclass
class ArtifactRecord:
    model_version: str
    uri: str
    sha256: str
    size_bytes: int


@dataclass
class EvaluationRecord:
    model_version: str
    passed: bool
    report: dict[str, Any]
    created_at: datetime


class Registry(ABC):
    @abstractmethod
    def register_model(self, name: str, version: str, metadata: dict[str, Any]) -> ModelRecord:
        raise NotImplementedError

    @abstractmethod
    def update_status(self, name: str, version: str, status: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, name: str, version: str) -> ModelRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_by_status(self, name: str, status: str) -> ModelRecord | None:
        raise NotImplementedError

    @abstractmethod
    def add_artifact(self, record: ArtifactRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(self, model_version: str) -> list[ArtifactRecord]:
        raise NotImplementedError

    @abstractmethod
    def add_evaluation(self, record: EvaluationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_evaluation(self, model_version: str) -> EvaluationRecord | None:
        raise NotImplementedError
