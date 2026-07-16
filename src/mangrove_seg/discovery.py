"""Discover and validate the temporal raster pairs declared by a project config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from mangrove_seg.config import ProjectConfig


class DataLayoutError(FileNotFoundError):
    """Raised when configured rasters are missing or are not regular files."""


@dataclass(frozen=True, order=True)
class TemporalRasterPair:
    """One feature year paired with the following year's segmentation label."""

    region: str
    input_year: int
    target_year: int
    feature_path: Path
    label_path: Path

    def as_dict(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = asdict(self)
        payload["feature_path"] = str(self.feature_path)
        payload["label_path"] = str(self.label_path)
        return payload


@dataclass(frozen=True)
class DataInventory:
    """Portable inventory of expected pairs and any absent assets."""

    pairs: tuple[TemporalRasterPair, ...]
    missing: tuple[Path, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def require_complete(self) -> DataInventory:
        if self.missing:
            preview = "\n".join(f"  - {path}" for path in self.missing[:10])
            remaining = len(self.missing) - min(10, len(self.missing))
            suffix = f"\n  ... and {remaining} more" if remaining else ""
            raise DataLayoutError(
                f"Missing {len(self.missing)} configured raster(s):\n{preview}{suffix}"
            )
        return self

    def as_dict(self) -> dict[str, object]:
        return {
            "complete": self.is_complete,
            "pair_count": len(self.pairs),
            "missing": [str(path) for path in self.missing],
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def discover_temporal_pairs(config: ProjectConfig) -> DataInventory:
    """Resolve every region/year pair without opening heavyweight raster data.

    Feature rasters at year ``t`` are paired with the label raster at ``t + 1``.
    Only complete pairs are returned in ``pairs``; every absent or non-file path
    is recorded in ``missing`` for one-shot validation feedback.
    """
    pairs: list[TemporalRasterPair] = []
    missing: set[Path] = set()
    years = config.data.years

    for region in sorted(config.data.regions):
        for input_year, target_year in pairwise(years):
            feature = config.data.embeddings_root / region / f"{input_year}.tif"
            label = config.data.labels_root / region / f"{target_year}.tif"
            feature_ok = feature.is_file()
            label_ok = label.is_file()
            if not feature_ok:
                missing.add(feature)
            if not label_ok:
                missing.add(label)
            if feature_ok and label_ok:
                pairs.append(
                    TemporalRasterPair(
                        region=region,
                        input_year=input_year,
                        target_year=target_year,
                        feature_path=feature,
                        label_path=label,
                    )
                )

    return DataInventory(tuple(pairs), tuple(sorted(missing)))
