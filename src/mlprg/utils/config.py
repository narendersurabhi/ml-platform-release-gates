from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLPRG_", case_sensitive=False)

    app_name: str = "mlprg-api"
    api_key: str = "dev-api-key"
    database_url: str = "sqlite:///./mlprg.db"
    model_name: str = "breast_cancer_classifier"
    artifact_store: str = "local"
    artifact_bucket: str = "mlprg-artifacts"
    artifact_local_path: str = "./artifacts"
    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    min_auc: float = 0.9
    min_accuracy: float = 0.9
    max_model_size_mb: float = 5.0
    max_p95_inference_ms: float = 25.0
    enable_safety_checks: bool = True

    log_json: bool = True
    prometheus_port: int = 8001


settings = Settings()
