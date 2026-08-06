"""Memory-mapped feature-window cache and worker-safe dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .experiment import ContractError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowCache:
    data_path: Path
    index_path: Path
    manifest_path: Path


class TemporaryWindowMemmap:
    """Own a temporary read-only NPY memmap for one experiment process."""

    def __init__(self, features: np.ndarray) -> None:
        self._closed = True
        self._directory = Path(tempfile.mkdtemp(prefix="tcn-window-cache-"))
        self.path = self._directory / "feature-windows.npy"
        np.save(self.path, np.asarray(features, dtype="float32"), allow_pickle=False)
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        mapped = getattr(self.array, "_mmap", None)
        if mapped is not None:
            mapped.close()
        shutil.rmtree(self._directory)
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, PermissionError):
            pass


class TemporaryWindowOutput:
    """Own a temporary NPY target that a chunked materializer can fill."""

    def __init__(self, filename: str = "feature-windows.npy") -> None:
        self._closed = True
        self._directory = Path(tempfile.mkdtemp(prefix="tcn-window-output-"))
        self.path = self._directory / filename
        self.array: np.ndarray | None = None
        self._closed = False

    def adopt_read_only(self, written: np.ndarray) -> np.ndarray:
        """Flush/close the writer and retain a read-only mapping until cleanup."""

        mapping = getattr(written, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return self.array

    def close(self) -> None:
        if self._closed:
            return
        mapping = getattr(self.array, "_mmap", None)
        if mapping is not None:
            mapping.close()
        shutil.rmtree(self._directory)
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, PermissionError):
            pass


def write_window_cache(
    output_dir: str | Path,
    features: np.ndarray,
    index: pd.DataFrame,
    *,
    source_fingerprint: str,
    feature_version: str,
) -> WindowCache:
    """Persist a rebuildable memory-mapped tensor and its sample index."""

    if len(features) != len(index):
        raise ContractError("feature tensor and window index sample counts must match")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "feature-windows.npy"
    index_path = root / "window-index.parquet"
    manifest_path = root / "window-cache.json"
    np.save(data_path, np.asarray(features, dtype="float32"), allow_pickle=False)
    index.to_parquet(index_path, index=False)
    manifest = {
        "schema_version": 1,
        "source_fingerprint": source_fingerprint,
        "feature_version": feature_version,
        "sample_count": int(len(index)),
        "shape": list(features.shape),
        "dtype": "float32",
        "data_sha256": _file_sha256(data_path),
        "index_sha256": _file_sha256(index_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return WindowCache(
        data_path=data_path, index_path=index_path, manifest_path=manifest_path
    )


class StreamingWindowDataset:
    """Read-only window dataset that reopens its mmap after process serialization."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        index_path: str | Path,
        manifest_path: str | Path,
        expected_source_fingerprint: str,
    ) -> None:
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.manifest_path = Path(manifest_path)
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read window cache manifest: {exc}") from exc
        if self.manifest.get("source_fingerprint") != expected_source_fingerprint:
            raise ContractError("cache source fingerprint mismatch")
        if _file_sha256(self.data_path) != self.manifest.get("data_sha256"):
            raise ContractError("cache feature data fingerprint mismatch")
        if _file_sha256(self.index_path) != self.manifest.get("index_sha256"):
            raise ContractError("cache index fingerprint mismatch")
        self.index = (
            pd.read_parquet(self.index_path)
            .sort_values("sample_position")
            .reset_index(drop=True)
        )
        if len(self.index) != int(self.manifest.get("sample_count", -1)):
            raise ContractError("cache manifest sample count mismatch")
        self._array: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _features(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        return self._array

    def __getitem__(self, position: int) -> dict[str, Any]:
        row = self.index.iloc[position]
        features = self._features()[int(row["sample_position"])]
        return {"sample_id": row["sample_id"], "features": features}

    def positions_for_worker(self, worker_id: int, worker_count: int) -> Iterator[int]:
        if worker_count <= 0 or worker_id < 0 or worker_id >= worker_count:
            raise ValueError("worker_id must identify one of worker_count workers")
        return iter(range(worker_id, len(self), worker_count))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_array"] = None
        return state
