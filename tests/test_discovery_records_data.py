from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mangrove_seg.config import load_config
from mangrove_seg.data import load_tile
from mangrove_seg.discovery import DataLayoutError, discover_temporal_pairs
from mangrove_seg.records import TileRecord, read_records, resolve_record_paths, write_records


def _record() -> TileRecord:
    return TileRecord(
        tile_id="alpha_2019_2020_r0_c0",
        split="train",
        region="alpha",
        input_year=2019,
        target_year=2020,
        row=0,
        column=0,
        positive_fraction=0.25,
        valid_pixels=16,
        is_positive=True,
        feature_path="train/features/tile.npy",
        label_path="train/labels/tile.npy",
        valid_mask_path="train/valid/tile.npy",
    )


def test_discovery_reports_all_missing_assets(config_file: Path) -> None:
    inventory = discover_temporal_pairs(load_config(config_file))
    assert not inventory.is_complete
    assert len(inventory.pairs) == 0
    assert len(inventory.missing) == 20
    with pytest.raises(DataLayoutError, match="Missing 20"):
        inventory.require_complete()


def test_discovery_returns_complete_sorted_pairs(config_file: Path) -> None:
    config = load_config(config_file)
    for region in config.data.regions:
        for year in config.data.years:
            feature = config.data.embeddings_root / region / f"{year}.tif"
            label = config.data.labels_root / region / f"{year}.tif"
            feature.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            feature.touch()
            label.touch()
    inventory = discover_temporal_pairs(config)
    assert inventory.is_complete
    assert len(inventory.pairs) == 10
    assert inventory.pairs[0].region == "alpha"
    assert inventory.pairs[0].target_year == 2020


def test_record_round_trip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    write_records(path, [_record()])
    assert read_records(path) == [_record()]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_malformed_record_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2"):
        read_records(path)


def test_load_tile_converts_layout_and_encodes_validity(tmp_path: Path) -> None:
    record = _record()
    for relative in (record.feature_path, record.label_path, record.valid_mask_path):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    features = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    labels = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.uint8,
    )
    valid = np.ones((4, 4), dtype=np.uint8)
    valid[3, 3] = 0
    np.save(tmp_path / record.feature_path, features, allow_pickle=False)
    np.save(tmp_path / record.label_path, labels, allow_pickle=False)
    np.save(tmp_path / record.valid_mask_path, valid, allow_pickle=False)

    tile = load_tile(record, tmp_path, expected_channels=2, expected_size=4)
    assert tile.features.shape == (4, 4, 2)
    assert tile.labels[3, 3] == 0
    assert np.array_equal(tile.encoded_target()[..., 1], valid)


def test_load_tile_rejects_unaligned_arrays(tmp_path: Path) -> None:
    record = _record()
    for relative in (record.feature_path, record.label_path, record.valid_mask_path):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / record.feature_path, np.zeros((2, 4, 4)), allow_pickle=False)
    np.save(tmp_path / record.label_path, np.zeros((3, 3)), allow_pickle=False)
    np.save(tmp_path / record.valid_mask_path, np.zeros((3, 3)), allow_pickle=False)
    with pytest.raises(ValueError, match="Unaligned"):
        load_tile(record, tmp_path)


def test_load_tile_rejects_nonfinite_features(tmp_path: Path) -> None:
    record = _record()
    for relative in (record.feature_path, record.label_path, record.valid_mask_path):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    features = np.zeros((2, 4, 4))
    features[0, 0, 0] = np.nan
    np.save(tmp_path / record.feature_path, features, allow_pickle=False)
    np.save(tmp_path / record.label_path, np.zeros((4, 4)), allow_pickle=False)
    np.save(tmp_path / record.valid_mask_path, np.ones((4, 4)), allow_pickle=False)
    with pytest.raises(ValueError, match="NaN"):
        load_tile(record, tmp_path)


def test_record_paths_cannot_escape_tile_root(tmp_path: Path) -> None:
    record = _record()
    unsafe = replace(record, feature_path="../../outside.npy")
    with pytest.raises(ValueError, match="escapes"):
        resolve_record_paths(unsafe, tmp_path)
