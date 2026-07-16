from __future__ import annotations

from pathlib import Path

import pytest

from mangrove_seg.config import ConfigError, load_config


def _replace(path: Path, old: str, new: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_load_config_resolves_relative_paths(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.seed == 17
    assert config.data.embeddings_root == config_file.parent / "data" / "embeddings"
    assert config.data.regions[0] == "alpha"
    assert config.model.filters == (8, 16)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("years = [2019, 2020, 2021]", "years = [2020, 2019]", "ascending"),
        ("years = [2019, 2020, 2021]", "years = [2019]", "at least two"),
        ("train_stride = 8", "train_stride = 32", "strides"),
        ("minimum_positive_fraction = 0.01", "minimum_positive_fraction = 1.2", "between"),
        ("dropout = 0.1", "dropout = 1.0", "dropout"),
        ('normalization = "batch"', 'normalization = "layer"', "normalization"),
        ("deep_supervision = true", 'deep_supervision = "true"', "true or false"),
        ('loss = "focal_dice"', 'loss = "unknown"', "loss"),
    ],
)
def test_invalid_values_are_rejected(config_file: Path, old: str, new: str, message: str) -> None:
    _replace(config_file, old, new)
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_region_names_cannot_escape_root(config_file: Path) -> None:
    _replace(config_file, '"alpha"', '"../alpha"')
    with pytest.raises(ConfigError, match="plain directory names"):
        load_config(config_file)


def test_region_names_cannot_be_empty(config_file: Path) -> None:
    _replace(config_file, '"alpha"', '""')
    with pytest.raises(ConfigError, match="empty strings"):
        load_config(config_file)


def test_tile_size_must_match_model_depth(config_file: Path) -> None:
    _replace(config_file, "tile_size = 16", "tile_size = 15")
    _replace(config_file, "evaluation_stride = 16", "evaluation_stride = 15")
    with pytest.raises(ConfigError, match="divisible"):
        load_config(config_file)


def test_explicit_split_must_cover_regions(config_file: Path) -> None:
    _replace(config_file, 'mode = "hash"', 'mode = "explicit"')
    with pytest.raises(ConfigError, match="cover every"):
        load_config(config_file)


def test_missing_section_is_reported(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8")
    config_file.write_text(text.replace("[training]", "[other]"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[training\]"):
        load_config(config_file)
