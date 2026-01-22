from __future__ import annotations

import hashlib
from pathlib import Path

from mlprg.storage.local import LocalArtifactStore


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_local_artifact_store_put_get(tmp_path: Path) -> None:
    store = LocalArtifactStore(str(tmp_path / "store"))
    source = tmp_path / "artifact.txt"
    source.write_text("hello")
    artifact = store.put(str(source), "data/artifact.txt")
    assert artifact.size_bytes == source.stat().st_size
    assert artifact.sha256 == _sha256(source)
    dest = tmp_path / "downloaded.txt"
    store.get(artifact.uri, str(dest))
    assert dest.read_text() == "hello"
