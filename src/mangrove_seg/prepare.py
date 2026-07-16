"""Reproducible raster-to-tile preparation workflow."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mangrove_seg.config import ProjectConfig
from mangrove_seg.discovery import discover_temporal_pairs
from mangrove_seg.geospatial import load_aligned_pair
from mangrove_seg.records import TileRecord, write_records
from mangrove_seg.splitting import (
    RegionSplit,
    deterministic_region_split,
    explicit_region_split,
)
from mangrove_seg.tiling import (
    TileStatistics,
    TileWindow,
    extract_tile,
    iter_tile_windows,
    pad_pair_to_tile,
    tile_statistics,
)


@dataclass(frozen=True)
class PreparationSummary:
    split: RegionSplit
    tile_counts: dict[str, int]
    positive_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "regions": self.split.as_dict(),
            "tile_counts": self.tile_counts,
            "positive_counts": self.positive_counts,
        }


@dataclass(frozen=True)
class _Candidate:
    window: TileWindow
    statistics: TileStatistics
    is_positive: bool
    rank: bytes


def resolve_region_split(config: ProjectConfig) -> RegionSplit:
    split = config.split
    if split.mode == "explicit":
        return explicit_region_split(
            config.data.regions,
            train=split.train_regions,
            validation=split.validation_regions,
            test=split.test_regions,
        )
    return deterministic_region_split(
        config.data.regions,
        seed=config.seed,
        validation_fraction=split.validation_fraction,
        test_fraction=split.test_fraction,
    )


def write_split_manifest(config: ProjectConfig, split: RegionSplit) -> Path:
    destination = config.data.artifacts_root / "region-split.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": config.seed, "strategy": config.split.mode, **split.as_dict()}
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "region"


def _candidate_rank(seed: int, region: str, input_year: int, window: TileWindow) -> bytes:
    value = f"{seed}\0{region}\0{input_year}\0{window.row}\0{window.column}"
    return hashlib.sha256(value.encode()).digest()


def _pair_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tile_size: int,
    stride: int,
    minimum_positive_fraction: float,
    seed: int,
    region: str,
    input_year: int,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for window in iter_tile_windows(labels.shape, tile_size, stride):
        _, label_tile, valid_tile = extract_tile(features, labels, valid_mask, window)
        statistics = tile_statistics(label_tile, valid_tile)
        if statistics.valid_pixels == 0:
            continue
        is_positive = (
            statistics.positive_pixels > 0
            and statistics.positive_fraction >= minimum_positive_fraction
        )
        candidates.append(
            _Candidate(
                window=window,
                statistics=statistics,
                is_positive=is_positive,
                rank=_candidate_rank(seed, region, input_year, window),
            )
        )
    return candidates


def prepare_tiles(config: ProjectConfig, *, overwrite: bool = False) -> PreparationSummary:
    """Align yearly pairs, assign whole regions, and persist portable NumPy tiles."""
    root = config.data.tiles_root
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Tile output is not empty: {root}. Use --overwrite deliberately."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    split = resolve_region_split(config)
    assignment = split.assignment()
    inventory = discover_temporal_pairs(config).require_complete()
    records_by_split: dict[str, list[TileRecord]] = defaultdict(list)
    negative_count_by_region: dict[str, int] = defaultdict(int)

    for temporal_pair in inventory.pairs:
        region = temporal_pair.region
        input_year = temporal_pair.input_year
        target_year = temporal_pair.target_year
        split_name = assignment[region]
        stride = config.data.train_stride
        if split_name != "train":
            stride = config.data.evaluation_stride

        pair = load_aligned_pair(temporal_pair.feature_path, temporal_pair.label_path)
        if pair.features.shape[0] != config.data.channels:
            raise ValueError(
                f"Expected {config.data.channels} channels in {temporal_pair.feature_path}, "
                f"found {pair.features.shape[0]}."
            )
        features, labels, valid_mask = pad_pair_to_tile(
            pair.features,
            pair.labels,
            pair.valid_mask,
            config.data.tile_size,
        )
        candidates = _pair_candidates(
            features,
            labels,
            valid_mask,
            tile_size=config.data.tile_size,
            stride=stride,
            minimum_positive_fraction=config.data.minimum_positive_fraction,
            seed=config.seed,
            region=region,
            input_year=input_year,
        )
        positives = [candidate for candidate in candidates if candidate.is_positive]
        negatives = sorted(
            (candidate for candidate in candidates if not candidate.is_positive),
            key=lambda candidate: candidate.rank,
        )
        if split_name == "train":
            remaining = max(
                0,
                config.data.maximum_negative_tiles_per_region - negative_count_by_region[region],
            )
            negatives = negatives[:remaining]
            negative_count_by_region[region] += len(negatives)
        selected = sorted(positives + negatives, key=lambda item: item.window)

        for candidate in selected:
            feature_tile, label_tile, valid_tile = extract_tile(
                features, labels, valid_mask, candidate.window
            )
            tile_id = (
                f"{_slug(region)}_{input_year}_{target_year}_"
                f"r{candidate.window.row}_c{candidate.window.column}"
            )
            relative_base = Path(split_name)
            feature_relative = relative_base / "features" / f"{tile_id}.npy"
            label_relative = relative_base / "labels" / f"{tile_id}.npy"
            valid_relative = relative_base / "valid" / f"{tile_id}.npy"
            for relative in (feature_relative, label_relative, valid_relative):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / feature_relative, feature_tile.astype(np.float32), allow_pickle=False)
            np.save(root / label_relative, label_tile.astype(np.uint8), allow_pickle=False)
            np.save(root / valid_relative, valid_tile.astype(np.uint8), allow_pickle=False)
            records_by_split[split_name].append(
                TileRecord(
                    tile_id=tile_id,
                    split=split_name,
                    region=region,
                    input_year=input_year,
                    target_year=target_year,
                    row=candidate.window.row,
                    column=candidate.window.column,
                    positive_fraction=candidate.statistics.positive_fraction,
                    valid_pixels=candidate.statistics.valid_pixels,
                    is_positive=candidate.is_positive,
                    feature_path=feature_relative.as_posix(),
                    label_path=label_relative.as_posix(),
                    valid_mask_path=valid_relative.as_posix(),
                )
            )

    for split_name in ("train", "validation", "test"):
        write_records(root / split_name / "metadata.jsonl", records_by_split[split_name])
    write_split_manifest(config, split)
    return PreparationSummary(
        split=split,
        tile_counts={name: len(records_by_split[name]) for name in ("train", "validation", "test")},
        positive_counts={
            name: sum(record.is_positive for record in records_by_split[name])
            for name in ("train", "validation", "test")
        },
    )
