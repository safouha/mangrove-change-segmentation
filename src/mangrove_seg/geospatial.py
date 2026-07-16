"""Raster alignment and output helpers behind the optional geospatial extra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mangrove_seg.optional import require_rasterio


@dataclass(frozen=True)
class AlignedRasterPair:
    features: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    reference_profile: dict[str, Any]


def load_aligned_pair(
    feature_path: str | Path,
    label_path: str | Path,
) -> AlignedRasterPair:
    """Align continuous feature bands to a categorical label grid.

    Bilinear resampling is used only for continuous feature bands. The validity
    mask uses nearest-neighbor resampling, and the categorical target is never
    interpolated.
    """
    rasterio = require_rasterio()
    from rasterio.warp import Resampling, reproject

    with rasterio.open(label_path) as label_source:
        if label_source.crs is None:
            raise ValueError(f"Label raster has no CRS: {label_path}")
        labels = (label_source.read(1) > 0).astype(np.uint8)
        label_valid = label_source.read_masks(1) > 0
        destination_shape = (label_source.height, label_source.width)
        destination_crs = label_source.crs
        destination_transform = label_source.transform
        profile = label_source.profile.copy()

    with rasterio.open(feature_path) as feature_source:
        if feature_source.crs is None:
            raise ValueError(f"Feature raster has no CRS: {feature_path}")
        features = np.full(
            (feature_source.count, *destination_shape),
            np.nan,
            dtype=np.float32,
        )
        for band_index in feature_source.indexes:
            reproject(
                source=rasterio.band(feature_source, band_index),
                destination=features[band_index - 1],
                src_transform=feature_source.transform,
                src_crs=feature_source.crs,
                src_nodata=feature_source.nodata,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        feature_valid = np.zeros(destination_shape, dtype=np.uint8)
        reproject(
            source=feature_source.dataset_mask(),
            destination=feature_valid,
            src_transform=feature_source.transform,
            src_crs=feature_source.crs,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            resampling=Resampling.nearest,
        )

    finite = np.isfinite(features).all(axis=0)
    valid = np.logical_and.reduce((label_valid, feature_valid > 0, finite))
    features[:, ~valid] = 0.0
    labels[~valid] = 0
    return AlignedRasterPair(features, labels, valid.astype(np.uint8), profile)


def write_probability_raster(
    path: str | Path,
    probability: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    """Write a single-band float32 GeoTIFF on the reference grid."""
    rasterio = require_rasterio()
    array = np.asarray(probability, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError("Probability output must be a two-dimensional grid.")
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        height=array.shape[0],
        width=array.shape[1],
        dtype="float32",
        compress="deflate",
        predictor=3,
        nodata=None,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as sink:
        sink.write(array, 1)
