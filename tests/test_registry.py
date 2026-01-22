from __future__ import annotations

from pathlib import Path

from mlprg.registry.sqlite import SqliteRegistry


def test_registry_status_transitions(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    registry = SqliteRegistry(f"sqlite:///{db_path}")
    record = registry.register_model("model", "v1", {"accuracy": 0.9})
    assert record.status == "CANDIDATE"
    registry.update_status("model", "v1", "STAGING")
    registry.update_status("model", "v1", "PRODUCTION")
    latest = registry.get_latest_by_status("model", "PRODUCTION")
    assert latest is not None
    assert latest.version == "v1"
    registry.update_status("model", "v1", "REJECTED")
    rejected = registry.get_model("model", "v1")
    assert rejected is not None
    assert rejected.status == "REJECTED"
