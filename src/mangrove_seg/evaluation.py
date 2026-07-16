"""Dependency-light segmentation metrics for valid raster pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _mask_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    iou: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_segmentation_metrics(
    labels: np.ndarray,
    scores_or_predictions: np.ndarray,
    *,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> BinaryMetrics:
    """Compute confusion-derived metrics without requiring scikit-learn."""
    true = _mask_array(labels)
    scores = _mask_array(scores_or_predictions)
    if true.shape != scores.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    valid = (
        np.ones_like(true, dtype=bool)
        if valid_mask is None
        else _mask_array(valid_mask).astype(bool)
    )
    if valid.shape != true.shape:
        raise ValueError("Validity mask must match the label grid.")

    true_binary = true[valid] > 0
    predicted_binary = scores[valid] >= threshold
    tp = int(np.logical_and(true_binary, predicted_binary).sum())
    tn = int(np.logical_and(~true_binary, ~predicted_binary).sum())
    fp = int(np.logical_and(~true_binary, predicted_binary).sum())
    fn = int(np.logical_and(true_binary, ~predicted_binary).sum())

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return BinaryMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        accuracy=_ratio(tp + tn, tp + tn + fp + fn),
        precision=precision,
        recall=recall,
        specificity=_ratio(tn, tn + fp),
        f1=_ratio(2 * precision * recall, precision + recall),
        iou=_ratio(tp, tp + fp + fn),
    )
