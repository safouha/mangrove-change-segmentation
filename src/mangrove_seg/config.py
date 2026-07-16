"""Typed TOML configuration with validation and portable path resolution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    embeddings_root: Path
    labels_root: Path
    tiles_root: Path
    artifacts_root: Path
    regions: tuple[str, ...]
    years: tuple[int, ...]
    channels: int = 64
    tile_size: int = 128
    train_stride: int = 64
    evaluation_stride: int = 128
    minimum_positive_fraction: float = 0.001
    maximum_negative_tiles_per_region: int = 50_000


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "hash"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    train_regions: tuple[str, ...] = ()
    validation_regions: tuple[str, ...] = ()
    test_regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    filters: tuple[int, ...] = (64, 128, 256, 512)
    dropout: float = 0.1
    normalization: str = "batch"
    se_reduction: int = 16
    deep_supervision: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clipnorm: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.8
    loss: str = "focal_dice"
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing TOML section [{name}].")
    return value


def _resolve(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be an array of strings.")
    cleaned = tuple(item.strip() for item in value)
    if any(not item for item in cleaned):
        raise ConfigError(f"{field} cannot contain empty strings.")
    if len(cleaned) != len(set(cleaned)):
        raise ConfigError(f"{field} contains duplicate values.")
    return cleaned


def _integers(value: Any, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigError(f"{field} must be a non-empty array of integers.")
    return tuple(value)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer.")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric.")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false.")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string.")
    return value.strip()


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a TOML configuration.

    Relative paths are resolved against the configuration file's directory, not
    the caller's current working directory.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project_raw = _section(raw, "project")
    data_raw = _section(raw, "data")
    split_raw = _section(raw, "split")
    model_raw = _section(raw, "model")
    training_raw = _section(raw, "training")

    base = config_path.parent
    regions = _strings(data_raw.get("regions"), "data.regions")
    if not regions:
        raise ConfigError("data.regions must list at least one region.")

    data = DataConfig(
        embeddings_root=_resolve(base, data_raw.get("embeddings_root"), "data.embeddings_root"),
        labels_root=_resolve(base, data_raw.get("labels_root"), "data.labels_root"),
        tiles_root=_resolve(base, data_raw.get("tiles_root"), "data.tiles_root"),
        artifacts_root=_resolve(base, data_raw.get("artifacts_root"), "data.artifacts_root"),
        regions=regions,
        years=_integers(data_raw.get("years"), "data.years"),
        channels=_integer(data_raw.get("channels", 64), "data.channels"),
        tile_size=_integer(data_raw.get("tile_size", 128), "data.tile_size"),
        train_stride=_integer(data_raw.get("train_stride", 64), "data.train_stride"),
        evaluation_stride=_integer(
            data_raw.get("evaluation_stride", 128), "data.evaluation_stride"
        ),
        minimum_positive_fraction=_number(
            data_raw.get("minimum_positive_fraction", 0.001),
            "data.minimum_positive_fraction",
        ),
        maximum_negative_tiles_per_region=_integer(
            data_raw.get("maximum_negative_tiles_per_region", 50_000),
            "data.maximum_negative_tiles_per_region",
        ),
    )
    split = SplitConfig(
        mode=_text(split_raw.get("mode", "hash"), "split.mode"),
        validation_fraction=_number(
            split_raw.get("validation_fraction", 0.2), "split.validation_fraction"
        ),
        test_fraction=_number(split_raw.get("test_fraction", 0.2), "split.test_fraction"),
        train_regions=_strings(split_raw.get("train_regions"), "split.train_regions"),
        validation_regions=_strings(
            split_raw.get("validation_regions"), "split.validation_regions"
        ),
        test_regions=_strings(split_raw.get("test_regions"), "split.test_regions"),
    )
    model = ModelConfig(
        filters=_integers(model_raw.get("filters", [64, 128, 256, 512]), "model.filters"),
        dropout=_number(model_raw.get("dropout", 0.1), "model.dropout"),
        normalization=_text(model_raw.get("normalization", "batch"), "model.normalization"),
        se_reduction=_integer(model_raw.get("se_reduction", 16), "model.se_reduction"),
        deep_supervision=_boolean(
            model_raw.get("deep_supervision", True), "model.deep_supervision"
        ),
    )
    training = TrainingConfig(
        batch_size=_integer(training_raw.get("batch_size", 16), "training.batch_size"),
        epochs=_integer(training_raw.get("epochs", 50), "training.epochs"),
        learning_rate=_number(training_raw.get("learning_rate", 1e-4), "training.learning_rate"),
        weight_decay=_number(training_raw.get("weight_decay", 1e-4), "training.weight_decay"),
        clipnorm=_number(training_raw.get("clipnorm", 1.0), "training.clipnorm"),
        focal_gamma=_number(training_raw.get("focal_gamma", 2.0), "training.focal_gamma"),
        focal_alpha=_number(training_raw.get("focal_alpha", 0.8), "training.focal_alpha"),
        loss=_text(training_raw.get("loss", "focal_dice"), "training.loss"),
        early_stopping_patience=_integer(
            training_raw.get("early_stopping_patience", 10),
            "training.early_stopping_patience",
        ),
        reduce_lr_patience=_integer(
            training_raw.get("reduce_lr_patience", 5), "training.reduce_lr_patience"
        ),
    )
    config = ProjectConfig(
        seed=_integer(project_raw.get("seed", 42), "project.seed"),
        data=data,
        split=split,
        model=model,
        training=training,
    )
    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    """Validate cross-field invariants before expensive work starts."""
    data, split, model, training = config.data, config.split, config.model, config.training

    if (
        len(data.years) < 2
        or tuple(sorted(data.years)) != data.years
        or len(set(data.years)) != len(data.years)
        or any(year <= 0 for year in data.years)
    ):
        raise ConfigError("data.years must contain at least two unique years in ascending order.")
    if any(Path(region).name != region or region in {".", ".."} for region in data.regions):
        raise ConfigError("data.regions entries must be plain directory names, not paths.")
    for name, value in {
        "channels": data.channels,
        "tile_size": data.tile_size,
        "train_stride": data.train_stride,
        "evaluation_stride": data.evaluation_stride,
    }.items():
        if value <= 0:
            raise ConfigError(f"data.{name} must be positive.")
    if data.train_stride > data.tile_size or data.evaluation_stride > data.tile_size:
        raise ConfigError("Tiling strides cannot exceed data.tile_size.")
    if not 0 <= data.minimum_positive_fraction <= 1:
        raise ConfigError("data.minimum_positive_fraction must be between 0 and 1.")
    if data.maximum_negative_tiles_per_region < 0:
        raise ConfigError("data.maximum_negative_tiles_per_region cannot be negative.")

    if split.mode not in {"hash", "explicit"}:
        raise ConfigError("split.mode must be 'hash' or 'explicit'.")
    if split.mode == "hash":
        if not 0 <= split.validation_fraction < 1 or not 0 <= split.test_fraction < 1:
            raise ConfigError("Split fractions must be in [0, 1).")
        if split.validation_fraction + split.test_fraction >= 1:
            raise ConfigError(
                "Validation and test fractions must leave a non-empty train fraction."
            )
    else:
        configured = split.train_regions + split.validation_regions + split.test_regions
        if set(configured) != set(data.regions) or len(configured) != len(set(configured)):
            raise ConfigError(
                "Explicit region splits must be disjoint and cover every "
                "data.regions entry exactly once."
            )

    if model.normalization not in {"batch", "none"}:
        raise ConfigError("model.normalization must be 'batch' or 'none'.")
    if not 0 <= model.dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1).")
    if model.se_reduction <= 0 or any(value <= 0 for value in model.filters):
        raise ConfigError("Model filters and SE reduction must be positive.")
    downsample_factor = 2 ** len(model.filters)
    if data.tile_size % downsample_factor:
        raise ConfigError(
            f"data.tile_size must be divisible by {downsample_factor} "
            f"for {len(model.filters)} encoder levels."
        )

    if training.loss not in {"focal_dice", "lovasz"}:
        raise ConfigError("training.loss must be 'focal_dice' or 'lovasz'.")
    if training.batch_size <= 0 or training.epochs <= 0:
        raise ConfigError("training.batch_size and training.epochs must be positive.")
    if training.early_stopping_patience < 0 or training.reduce_lr_patience < 0:
        raise ConfigError("Training callback patience values cannot be negative.")
    if training.learning_rate <= 0 or training.weight_decay < 0 or training.clipnorm <= 0:
        raise ConfigError(
            "Learning rate and clip norm must be positive; weight decay cannot be negative."
        )
    if not 0 <= training.focal_alpha <= 1 or training.focal_gamma < 0:
        raise ConfigError("Focal alpha must be in [0, 1] and gamma cannot be negative.")
