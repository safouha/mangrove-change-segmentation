from __future__ import annotations

import numpy as np
import pytest

from mangrove_seg.change import (
    BACKGROUND,
    GAIN,
    INVALID,
    LOSS,
    STABLE_MANGROVE,
    change_map,
    summarize_change,
)
from mangrove_seg.evaluation import binary_segmentation_metrics


def test_binary_metrics_for_known_confusion_matrix() -> None:
    labels = np.array([[1, 1], [0, 0]])
    predictions = np.array([[0.9, 0.2], [0.8, 0.1]])
    metrics = binary_segmentation_metrics(labels, predictions)
    assert (metrics.true_positive, metrics.true_negative) == (1, 1)
    assert (metrics.false_positive, metrics.false_negative) == (1, 1)
    assert metrics.accuracy == 0.5
    assert metrics.f1 == 0.5
    assert metrics.iou == pytest.approx(1 / 3)


def test_binary_metrics_ignore_invalid_pixels() -> None:
    labels = np.array([1, 0, 1])
    predictions = np.array([1.0, 1.0, 0.0])
    valid = np.array([1, 0, 1])
    metrics = binary_segmentation_metrics(labels, predictions, valid_mask=valid)
    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1


def test_empty_valid_set_has_defined_zero_metrics() -> None:
    metrics = binary_segmentation_metrics(np.array([1]), np.array([1]), valid_mask=np.array([0]))
    assert metrics.accuracy == 0
    assert metrics.iou == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_metric_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        binary_segmentation_metrics(np.zeros(2), np.zeros(2), threshold=threshold)


def test_metric_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        binary_segmentation_metrics(np.zeros((2, 2)), np.zeros(3))


def test_change_map_uses_documented_codes() -> None:
    previous = np.array([[0, 1], [1, 0]])
    current = np.array([[0, 1], [0, 1]])
    result = change_map(previous, current)
    assert np.array_equal(
        result,
        np.array([[BACKGROUND, STABLE_MANGROVE], [LOSS, GAIN]], dtype=np.uint8),
    )


def test_change_map_marks_invalid_pixels() -> None:
    result = change_map(
        np.ones((2, 2)),
        np.ones((2, 2)),
        valid_mask=np.array([[1, 1], [1, 0]]),
    )
    assert result[1, 1] == INVALID


def test_change_summary_reports_net_and_percentage() -> None:
    previous = np.array([1, 1, 1, 1, 0])
    current = np.array([1, 1, 1, 0, 1])
    summary = summarize_change(previous, current)
    assert summary.stable_pixels == 3
    assert summary.loss_pixels == 1
    assert summary.gain_pixels == 1
    assert summary.net_change_pixels == 0
    assert summary.percent_change == 0


def test_change_percentage_is_none_without_baseline() -> None:
    summary = summarize_change(np.zeros(2), np.array([0, 1]))
    assert summary.percent_change is None
    assert summary.gain_pixels == 1


def test_change_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        change_map(np.zeros(2), np.zeros(3))


def test_change_preserves_single_row_spatial_shape() -> None:
    previous = np.array([[0, 1]])
    current = np.array([[1, 1]])
    assert change_map(previous, current).shape == (1, 2)


def test_metrics_accept_singleton_prediction_channel() -> None:
    labels = np.array([[1, 0], [0, 1]])
    predictions = labels[..., None].astype(np.float32)
    assert binary_segmentation_metrics(labels, predictions).accuracy == 1
