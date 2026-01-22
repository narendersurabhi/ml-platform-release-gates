from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from mlprg.registry.base import ArtifactRecord, EvaluationRecord, ModelRecord, Registry


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str] = mapped_column(String, index=True, unique=True)
    status: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, index=True)
    uri: Mapped[str] = mapped_column(String)
    sha256: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int] = mapped_column(Integer)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, index=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SqliteRegistry(Registry):
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)

    def register_model(self, name: str, version: str, metadata: dict[str, Any]) -> ModelRecord:
        with Session(self.engine) as session:
            model = Model(
                name=name,
                version=version,
                status="CANDIDATE",
                metadata_json=metadata,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return ModelRecord(
                name=model.name,
                version=model.version,
                status=model.status,
                created_at=model.created_at,
                metadata=model.metadata_json,
            )

    def update_status(self, name: str, version: str, status: str) -> None:
        with Session(self.engine) as session:
            stmt = select(Model).where(Model.name == name, Model.version == version)
            model = session.scalar(stmt)
            if not model:
                raise ValueError(f"Model {name}:{version} not found")
            model.status = status
            session.commit()

    def get_model(self, name: str, version: str) -> ModelRecord | None:
        with Session(self.engine) as session:
            stmt = select(Model).where(Model.name == name, Model.version == version)
            model = session.scalar(stmt)
            if not model:
                return None
            return ModelRecord(
                name=model.name,
                version=model.version,
                status=model.status,
                created_at=model.created_at,
                metadata=model.metadata_json,
            )

    def get_latest_by_status(self, name: str, status: str) -> ModelRecord | None:
        with Session(self.engine) as session:
            stmt = (
                select(Model)
                .where(Model.name == name, Model.status == status)
                .order_by(Model.created_at.desc())
            )
            model = session.scalar(stmt)
            if not model:
                return None
            return ModelRecord(
                name=model.name,
                version=model.version,
                status=model.status,
                created_at=model.created_at,
                metadata=model.metadata_json,
            )

    def add_artifact(self, record: ArtifactRecord) -> None:
        with Session(self.engine) as session:
            artifact = Artifact(
                model_version=record.model_version,
                uri=record.uri,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
            )
            session.add(artifact)
            session.commit()

    def list_artifacts(self, model_version: str) -> list[ArtifactRecord]:
        with Session(self.engine) as session:
            stmt = select(Artifact).where(Artifact.model_version == model_version)
            artifacts = session.scalars(stmt).all()
            return [
                ArtifactRecord(
                    model_version=item.model_version,
                    uri=item.uri,
                    sha256=item.sha256,
                    size_bytes=item.size_bytes,
                )
                for item in artifacts
            ]

    def add_evaluation(self, record: EvaluationRecord) -> None:
        with Session(self.engine) as session:
            evaluation = Evaluation(
                model_version=record.model_version,
                passed=record.passed,
                report_json=record.report,
                created_at=record.created_at,
            )
            session.add(evaluation)
            session.commit()

    def get_latest_evaluation(self, model_version: str) -> EvaluationRecord | None:
        with Session(self.engine) as session:
            stmt = (
                select(Evaluation)
                .where(Evaluation.model_version == model_version)
                .order_by(Evaluation.created_at.desc())
            )
            evaluation = session.scalar(stmt)
            if not evaluation:
                return None
            return EvaluationRecord(
                model_version=evaluation.model_version,
                passed=bool(evaluation.passed),
                report=evaluation.report_json,
                created_at=evaluation.created_at,
            )
