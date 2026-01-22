from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from mlprg.storage.base import Artifact, ArtifactStore


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def put(self, local_path: str, dest_name: str) -> Artifact:
        source = Path(local_path)
        dest = self.base_path / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        sha256 = self._sha256(dest)
        size_bytes = dest.stat().st_size
        return Artifact(uri=str(dest), sha256=sha256, size_bytes=size_bytes)

    def get(self, uri: str, local_path: str) -> None:
        source = Path(uri)
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    def _sha256(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
