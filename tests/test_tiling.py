from __future__ import annotations

import numpy as np
import pytest

from mangrove_seg.tiling import (
    TileWindow,
    channels_first_to_last,
    extract_tile,
    iter_tile_windows,
    pad_pair_to_tile,
    tile_origins,
    tile_statistics,
)


@pytest.mark.parametrize(
    ("length", "tile_size", "stride", "expected"),
    [
        (4, 8, 8, (0,)),
        (8, 8, 8, (0,)),
        (20, 8, 8, (0, 8, 12)),
        (20, 8, 4, (0, 4, 8, 12)),
    ],
)
def test_tile_origins_cover_far_edge(
    length: int, tile_size: int, stride: int, expected: tuple[int, ...]
) -> None:
    assert tile_origins(length, tile_size, stride) == expected


@pytest.mark.parametrize("values", [(0, 8, 8), (8, 0, 8), (8, 8, 0), (20, 8, 9)])
def test_invalid_tile_origins(values: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        tile_origins(*values)


def test_windows_are_row_major_and_cover_corners() -> None:
    windows = list(iter_tile_windows((10, 12), tile_size=8, stride=8))
    assert windows == [
        TileWindow(0, 0, 8),
        TileWindow(0, 4, 8),
        TileWindow(2, 0, 8),
        TileWindow(2, 4, 8),
    ]


def test_padding_marks_new_pixels_invalid() -> None:
    features = np.ones((2, 3, 4), dtype=np.float32)
    labels = np.ones((3, 4), dtype=np.uint8)
    valid = np.ones((3, 4), dtype=np.uint8)
    padded_features, padded_labels, padded_valid = pad_pair_to_tile(features, labels, valid, 6)
    assert padded_features.shape == (2, 6, 6)
    assert padded_labels.shape == (6, 6)
    assert int(padded_valid.sum()) == 12


def test_extract_tile_preserves_alignment() -> None:
    labels = np.arange(36).reshape(6, 6)
    features = np.stack((labels, labels + 100))
    valid = np.ones((6, 6), dtype=np.uint8)
    feature_tile, label_tile, _ = extract_tile(features, labels, valid, TileWindow(2, 1, 3))
    assert np.array_equal(feature_tile[0], label_tile)
    assert np.array_equal(feature_tile[1], label_tile + 100)


def test_extract_out_of_bounds_window_is_rejected() -> None:
    array = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="beyond"):
        extract_tile(array[None, ...], array, array, TileWindow(2, 2, 4))


def test_tile_statistics_ignore_invalid_positive_pixels() -> None:
    labels = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    statistics = tile_statistics(labels, valid)
    assert statistics.valid_pixels == 3
    assert statistics.positive_pixels == 1
    assert statistics.positive_fraction == pytest.approx(1 / 3)


def test_channels_first_to_last() -> None:
    features = np.zeros((3, 4, 5), dtype=np.float32)
    assert channels_first_to_last(features).shape == (4, 5, 3)
    with pytest.raises(ValueError):
        channels_first_to_last(np.zeros((4, 5)))
