"""Overlap-aware tiled inference using a small model.predict interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from mangrove_seg.tiling import iter_tile_windows


def _primary_prediction(prediction: Any) -> np.ndarray:
    if isinstance(prediction, dict):
        if "mask" not in prediction:
            raise ValueError("Prediction dictionary does not contain the primary 'mask' output.")
        prediction = prediction["mask"]
    elif isinstance(prediction, (list, tuple)):
        if not prediction:
            raise ValueError("Model returned an empty prediction sequence.")
        prediction = prediction[0]
    array = np.asarray(prediction)
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError(f"Expected predictions shaped (B,H,W,1), received {array.shape}.")
    return array.astype(np.float32, copy=False)


def predict_tiled(
    model: Any,
    image: np.ndarray,
    *,
    tile_size: int,
    stride: int | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Predict a large channels-last image and average overlapping tiles."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Image must be channels-last with shape (H,W,C).")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    stride = tile_size if stride is None else stride
    height, width, _ = array.shape
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.pad(
        array,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="constant",
    )

    windows = list(iter_tile_windows(padded.shape[:2], tile_size, stride))
    probability_sum = np.zeros((*padded.shape[:2], 1), dtype=np.float32)
    visit_count = np.zeros((*padded.shape[:2], 1), dtype=np.float32)

    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch = np.stack(
            [padded[window.row_slice, window.column_slice, :] for window in batch_windows]
        )
        prediction = _primary_prediction(model.predict(batch, verbose=0))
        if prediction.shape[1:3] != (tile_size, tile_size):
            raise ValueError("Model output tile dimensions do not match tile_size.")
        for window, tile_prediction in zip(batch_windows, prediction, strict=True):
            probability_sum[window.row_slice, window.column_slice, :] += tile_prediction
            visit_count[window.row_slice, window.column_slice, :] += 1

    if np.any(visit_count == 0):  # protected by the no-gap stride invariant
        raise RuntimeError("Tiling left uncovered output pixels.")
    return (probability_sum / visit_count)[:height, :width, :]
