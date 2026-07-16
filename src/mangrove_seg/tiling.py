"""Pure NumPy tiling primitives shared by preparation and inference."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True)
class TileWindow:
    row: int
    column: int
    size: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.size)

    @property
    def column_slice(self) -> slice:
        return slice(self.column, self.column + self.size)


@dataclass(frozen=True)
class TileStatistics:
    valid_pixels: int
    positive_pixels: int
    positive_fraction: float


def tile_origins(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Return monotonic origins with the final tile anchored to the far edge."""
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("Length, tile size, and stride must be positive.")
    if stride > tile_size:
        raise ValueError("Stride cannot exceed tile size; gaps would be introduced.")
    if length <= tile_size:
        return (0,)

    final_origin = length - tile_size
    origins = list(range(0, final_origin + 1, stride))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return tuple(origins)


def iter_tile_windows(
    image_shape: tuple[int, int], tile_size: int, stride: int
) -> Iterator[TileWindow]:
    """Yield deterministic row-major windows that cover every image edge."""
    height, width = image_shape
    for row in tile_origins(height, tile_size, stride):
        for column in tile_origins(width, tile_size, stride):
            yield TileWindow(row=row, column=column, size=tile_size)


def pad_pair_to_tile(
    features: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a channels-first feature/label pair so at least one full tile exists."""
    if features.ndim != 3 or labels.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("Expected features (C,H,W), labels (H,W), and valid mask (H,W).")
    if features.shape[1:] != labels.shape or labels.shape != valid_mask.shape:
        raise ValueError("Feature, label, and validity grids must match.")

    height, width = labels.shape
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    feature_pad = ((0, 0), (0, pad_height), (0, pad_width))
    spatial_pad = ((0, pad_height), (0, pad_width))
    return (
        np.pad(features, feature_pad, mode="constant"),
        np.pad(labels, spatial_pad, mode="constant"),
        np.pad(valid_mask, spatial_pad, mode="constant"),
    )


def extract_tile(
    features: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    window: TileWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one channels-first feature tile and its aligned targets."""
    rows, columns = window.row_slice, window.column_slice
    feature_tile = features[:, rows, columns]
    label_tile = labels[rows, columns]
    valid_tile = valid_mask[rows, columns]
    expected_feature_shape = (features.shape[0], window.size, window.size)
    if feature_tile.shape != expected_feature_shape or label_tile.shape != (
        window.size,
        window.size,
    ):
        raise ValueError("Window extends beyond the padded array.")
    return feature_tile, label_tile, valid_tile


def tile_statistics(labels: np.ndarray, valid_mask: np.ndarray) -> TileStatistics:
    """Compute positive prevalence using only valid pixels."""
    if labels.shape != valid_mask.shape:
        raise ValueError("Label and validity masks must have the same shape.")
    valid = valid_mask.astype(bool)
    valid_pixels = int(valid.sum())
    positive_pixels = int(np.logical_and(labels > 0, valid).sum())
    fraction = positive_pixels / valid_pixels if valid_pixels else 0.0
    return TileStatistics(valid_pixels, positive_pixels, fraction)


def channels_first_to_last(features: np.ndarray) -> np.ndarray:
    """Convert a stored (C,H,W) tile to TensorFlow's (H,W,C) convention."""
    if features.ndim != 3:
        raise ValueError("Expected a three-dimensional feature tile.")
    return np.ascontiguousarray(np.moveaxis(features, 0, -1))
