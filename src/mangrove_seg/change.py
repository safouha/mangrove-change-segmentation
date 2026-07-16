"""Pixel-level change categories and transparent summary statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BACKGROUND = 0
STABLE_MANGROVE = 1
LOSS = 2
GAIN = 3
INVALID = 255


def _mask_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


@dataclass(frozen=True)
class ChangeSummary:
    baseline_pixels: int
    current_pixels: int
    stable_pixels: int
    loss_pixels: int
    gain_pixels: int
    net_change_pixels: int
    percent_change: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def change_map(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Encode background, stable mangrove, loss, and gain as 0..3."""
    before = _mask_array(previous) > 0
    after = _mask_array(current) > 0
    if before.shape != after.shape:
        raise ValueError("Previous and current masks must have the same shape.")
    result = np.full(before.shape, BACKGROUND, dtype=np.uint8)
    result[np.logical_and(before, after)] = STABLE_MANGROVE
    result[np.logical_and(before, ~after)] = LOSS
    result[np.logical_and(~before, after)] = GAIN
    if valid_mask is not None:
        valid = _mask_array(valid_mask).astype(bool)
        if valid.shape != before.shape:
            raise ValueError("Validity mask must match the input masks.")
        result[~valid] = INVALID
    return result


def summarize_change(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> ChangeSummary:
    """Summarize change without inventing an area conversion or resolution."""
    encoded = change_map(previous, current, valid_mask=valid_mask)
    stable = int((encoded == STABLE_MANGROVE).sum())
    loss = int((encoded == LOSS).sum())
    gain = int((encoded == GAIN).sum())
    baseline = stable + loss
    current_count = stable + gain
    net = current_count - baseline
    percentage = 100.0 * net / baseline if baseline else None
    return ChangeSummary(baseline, current_count, stable, loss, gain, net, percentage)
