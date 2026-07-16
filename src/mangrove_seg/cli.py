"""Command-line interface for validation, preparation, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mangrove_seg.change import change_map, summarize_change
from mangrove_seg.config import ConfigError, load_config
from mangrove_seg.discovery import DataLayoutError, discover_temporal_pairs
from mangrove_seg.evaluation import binary_segmentation_metrics
from mangrove_seg.optional import MissingOptionalDependency
from mangrove_seg.prepare import prepare_tiles, resolve_region_split
from mangrove_seg.training import train


def _write_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    inventory = discover_temporal_pairs(config)
    if not args.config_only:
        inventory.require_complete()
    _write_json(
        {
            "configuration": "valid",
            "seed": config.seed,
            "regions": len(config.data.regions),
            "years": list(config.data.years),
            "split": resolve_region_split(config).as_dict(),
            "inventory": inventory.as_dict(),
        }
    )
    return 0


def _split(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _write_json(resolve_region_split(config).as_dict())
    return 0


def _prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _write_json(prepare_tiles(config, overwrite=args.overwrite).as_dict())
    return 0


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _write_json(train(config).as_dict())
    return 0


def _metrics(args: argparse.Namespace) -> int:
    labels = np.load(args.labels, allow_pickle=False)
    predictions = np.load(args.predictions, allow_pickle=False)
    valid = None if args.valid_mask is None else np.load(args.valid_mask, allow_pickle=False)
    result = binary_segmentation_metrics(
        labels,
        predictions,
        threshold=args.threshold,
        valid_mask=valid,
    )
    _write_json(result.as_dict())
    return 0


def _change(args: argparse.Namespace) -> int:
    previous = np.load(args.previous, allow_pickle=False)
    current = np.load(args.current, allow_pickle=False)
    valid = None if args.valid_mask is None else np.load(args.valid_mask, allow_pickle=False)
    if args.output is not None:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, change_map(previous, current, valid_mask=valid), allow_pickle=False)
    _write_json(summarize_change(previous, current, valid_mask=valid).as_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mangrove-seg",
        description="Reproducible temporal mangrove segmentation and change analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate configuration and expected raster layout"
    )
    validate_parser.add_argument("config", type=Path)
    validate_parser.add_argument(
        "--config-only",
        action="store_true",
        help="validate schema and splits without requiring raster files",
    )
    validate_parser.set_defaults(handler=_validate)

    split_parser = subparsers.add_parser("split", help="preview deterministic region assignment")
    split_parser.add_argument("config", type=Path)
    split_parser.set_defaults(handler=_split)

    prepare_parser = subparsers.add_parser(
        "prepare", help="align rasters and create leakage-safe tile manifests"
    )
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=_prepare)

    train_parser = subparsers.add_parser("train", help="train the configured TensorFlow model")
    train_parser.add_argument("config", type=Path)
    train_parser.set_defaults(handler=_train)

    metrics_parser = subparsers.add_parser(
        "metrics", help="evaluate NumPy prediction and label arrays"
    )
    metrics_parser.add_argument("labels", type=Path)
    metrics_parser.add_argument("predictions", type=Path)
    metrics_parser.add_argument("--valid-mask", type=Path)
    metrics_parser.add_argument("--threshold", type=float, default=0.5)
    metrics_parser.set_defaults(handler=_metrics)

    change_parser = subparsers.add_parser("change", help="summarize two NumPy segmentation masks")
    change_parser.add_argument("previous", type=Path)
    change_parser.add_argument("current", type=Path)
    change_parser.add_argument("--valid-mask", type=Path)
    change_parser.add_argument("--output", type=Path, help="optional encoded change-map .npy")
    change_parser.set_defaults(handler=_change)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.handler
    try:
        return int(handler(args))
    except (
        ConfigError,
        DataLayoutError,
        MissingOptionalDependency,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
