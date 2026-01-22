from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Artifact:
    uri: str
    sha256: str
    size_bytes: int


class ArtifactStore(ABC):
    @abstractmethod
    def put(self, local_path: str, dest_name: str) -> Artifact:
        raise NotImplementedError

    @abstractmethod
    def get(self, uri: str, local_path: str) -> None:
        raise NotImplementedError
