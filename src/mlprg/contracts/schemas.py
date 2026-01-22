from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector for inference")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    status: str
    metadata: dict[str, Any]


class EvaluationReport(BaseModel):
    model_version: str
    passed: bool
    metrics: dict[str, float]
    checks: dict[str, bool]
    reasons: list[str]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str


class ReloadResponse(BaseModel):
    reloaded: bool
    model_version: str
