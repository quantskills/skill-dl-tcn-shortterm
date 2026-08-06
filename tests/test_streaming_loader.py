from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest
from torch.utils.data import DataLoader

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.streaming import StreamingWindowDataset, write_window_cache
from skill_dl_tcn_shortterm.training_data import LazyWindowDataset


def test_window_cache_streams_without_duplicate_worker_samples(tmp_path: Path) -> None:
    features = np.arange(5 * 2 * 4, dtype="float32").reshape(5, 2, 4)
    index = pd.DataFrame(
        {
            "sample_position": range(5),
            "sample_id": [f"sample-{i}" for i in range(5)],
            "instrument_id": "600000.XSHG",
            "signal_date": pd.bdate_range("2024-01-02", periods=5).strftime("%Y-%m-%d"),
        }
    )
    cache = write_window_cache(
        tmp_path,
        features,
        index,
        source_fingerprint="data-v1",
        feature_version="features-v1",
    )

    dataset = StreamingWindowDataset(
        data_path=cache.data_path,
        index_path=cache.index_path,
        manifest_path=cache.manifest_path,
        expected_source_fingerprint="data-v1",
    )
    assert len(dataset) == 5
    assert dataset[2]["sample_id"] == "sample-2"
    np.testing.assert_array_equal(dataset[2]["features"], features[2])
    assert not dataset[2]["features"].flags.writeable

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._array is None
    np.testing.assert_array_equal(restored[4]["features"], features[4])

    worker_positions = [
        position
        for worker in range(3)
        for position in restored.positions_for_worker(worker, 3)
    ]
    assert sorted(worker_positions) == list(range(5))
    assert len(worker_positions) == len(set(worker_positions))

    with pytest.raises(ContractError, match="cache source fingerprint mismatch"):
        StreamingWindowDataset(
            data_path=cache.data_path,
            index_path=cache.index_path,
            manifest_path=cache.manifest_path,
            expected_source_fingerprint="different-data",
        )


def test_training_loader_reopens_memmap_in_multiple_workers(tmp_path: Path) -> None:
    feature_path = tmp_path / "training-features.npy"
    features = np.arange(8 * 2 * 4, dtype="float32").reshape(8, 2, 4)
    np.save(feature_path, features)
    mapped = np.load(feature_path, mmap_mode="r")
    dataset = LazyWindowDataset(
        mapped,
        np.arange(8),
        np.zeros((8, 4), dtype="float32"),
        np.ones((8, 4), dtype="bool"),
        np.zeros(2, dtype="float32"),
        np.ones(2, dtype="float32"),
    )

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._features is None
    loader = DataLoader(
        restored,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    positions = []
    for batch_features, _, _, batch_positions in loader:
        positions.extend(batch_positions.tolist())
        assert batch_features.shape[1:] == (2, 4)

    assert positions == list(range(8))
    assert restored._features is None


def test_stress_memmap_rss_is_bounded_by_batch_prefetch_not_dataset_size(
    tmp_path: Path,
) -> None:
    sample_count = 4096
    feature_count = 8
    time_steps = 240
    batch_size = 16
    feature_path = tmp_path / "stress-features.npy"
    mapped = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype="float32",
        shape=(sample_count, feature_count, time_steps),
    )
    for start in range(0, sample_count, 128):
        mapped[start : start + 128] = float(start)
    mapped.flush()
    del mapped
    gc.collect()
    feature_bytes = feature_path.stat().st_size
    process = psutil.Process()
    rss_before = process.memory_info().rss
    dataset = LazyWindowDataset(
        np.load(feature_path, mmap_mode="r"),
        np.arange(sample_count),
        np.zeros((sample_count, 4), dtype="float32"),
        np.ones((sample_count, 4), dtype="bool"),
        np.zeros(feature_count, dtype="float32"),
        np.ones(feature_count, dtype="float32"),
    )
    serialized_bytes = len(pickle.dumps(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    iterator = iter(loader)
    observed = 0
    for _ in range(8):
        batch_features, _, _, batch_positions = next(iterator)
        observed += len(batch_positions)
        assert batch_features.shape == (batch_size, feature_count, time_steps)
    del iterator
    del loader
    gc.collect()
    rss_delta = max(0, process.memory_info().rss - rss_before)

    assert observed == batch_size * 8
    assert dataset._features is None
    assert serialized_bytes < 1_000_000
    assert rss_delta < feature_bytes
