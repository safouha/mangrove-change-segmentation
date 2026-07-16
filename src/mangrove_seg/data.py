"""Loading and validating prepared NumPy tiles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mangrove_seg.records import TileRecord, resolve_record_paths
from mangrove_seg.tiling import channels_first_to_last


@dataclass(frozen=True)
class LoadedTile:
    """One channels-last feature tile and its aligned label/validity masks."""

    features: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray

    def encoded_target(self) -> np.ndarray:
        """Return ``[..., label, validity]`` for validity-aware training losses."""
        return np.stack((self.labels, self.valid_mask), axis=-1).astype(np.float32)


def load_tile(
    record: TileRecord,
    tiles_root: str | Path,
    *,
    expected_channels: int | None = None,
    expected_size: int | None = None,
) -> LoadedTile:
    """Load one prepared tile and fail early on shape or numeric corruption."""
    feature_path, label_path, valid_path = resolve_record_paths(record, tiles_root)
    features = np.load(feature_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    valid = np.load(valid_path, allow_pickle=False)

    if features.ndim != 3 or labels.ndim != 2 or valid.ndim != 2:
        raise ValueError(f"Malformed arrays for tile {record.tile_id}.")
    if features.shape[1:] != labels.shape or labels.shape != valid.shape:
        raise ValueError(f"Unaligned arrays for tile {record.tile_id}.")
    if expected_channels is not None and features.shape[0] != expected_channels:
        raise ValueError(
            f"Tile {record.tile_id} has {features.shape[0]} channels; expected {expected_channels}."
        )
    if expected_size is not None and labels.shape != (expected_size, expected_size):
        raise ValueError(
            f"Tile {record.tile_id} has spatial shape {labels.shape}; "
            f"expected {(expected_size, expected_size)}."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Feature tile {record.tile_id} contains NaN or infinite values.")

    valid_binary = (valid > 0).astype(np.float32)
    label_binary = (labels > 0).astype(np.float32)
    label_binary[valid_binary == 0] = 0
    return LoadedTile(
        features=channels_first_to_last(features).astype(np.float32, copy=False),
        labels=label_binary,
        valid_mask=valid_binary,
    )


def iter_loaded_tiles(
    records: Sequence[TileRecord],
    tiles_root: str | Path,
    *,
    expected_channels: int | None = None,
    expected_size: int | None = None,
) -> Iterator[LoadedTile]:
    """Stream validated tiles without retaining the whole dataset in memory."""
    for record in records:
        yield load_tile(
            record,
            tiles_root,
            expected_channels=expected_channels,
            expected_size=expected_size,
        )
