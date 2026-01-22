from __future__ import annotations

import hashlib
from pathlib import Path

from boto3.session import Session

from mlprg.storage.base import Artifact, ArtifactStore


class S3ArtifactStore(ArtifactStore):
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        self.bucket = bucket
        self.client = Session().client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        existing = [item["Name"] for item in self.client.list_buckets().get("Buckets", [])]
        if self.bucket not in existing:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, local_path: str, dest_name: str) -> Artifact:
        path = Path(local_path)
        self.client.upload_file(str(path), self.bucket, dest_name)
        sha256 = self._sha256(path)
        size_bytes = path.stat().st_size
        uri = f"s3://{self.bucket}/{dest_name}"
        return Artifact(uri=uri, sha256=sha256, size_bytes=size_bytes)

    def get(self, uri: str, local_path: str) -> None:
        _, _, key = uri.partition("s3://")
        _, _, object_key = key.partition("/")
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, object_key, str(dest))

    def _sha256(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
