"""Portable JSONL metadata for prepared tiles."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer.")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric.")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean.")
    return value


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    split: str
    region: str
    input_year: int
    target_year: int
    row: int
    column: int
    positive_fraction: float
    valid_pixels: int
    is_positive: bool
    feature_path: str
    label_path: str
    valid_mask_path: str

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TileRecord:
        return cls(
            tile_id=_string(value["tile_id"], "tile_id"),
            split=_string(value["split"], "split"),
            region=_string(value["region"], "region"),
            input_year=_integer(value["input_year"], "input_year"),
            target_year=_integer(value["target_year"], "target_year"),
            row=_integer(value["row"], "row"),
            column=_integer(value["column"], "column"),
            positive_fraction=_number(value["positive_fraction"], "positive_fraction"),
            valid_pixels=_integer(value["valid_pixels"], "valid_pixels"),
            is_positive=_boolean(value["is_positive"], "is_positive"),
            feature_path=_string(value["feature_path"], "feature_path"),
            label_path=_string(value["label_path"], "label_path"),
            valid_mask_path=_string(value["valid_mask_path"], "valid_mask_path"),
        )


def write_records(path: str | Path, records: Iterable[TileRecord]) -> None:
    """Atomically replace a metadata file with deterministic JSON lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(destination)


def read_records(path: str | Path) -> list[TileRecord]:
    """Read tile metadata and report malformed rows with line numbers."""
    source = Path(path)
    records: list[TileRecord] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TileRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid record at {source}:{line_number}: {exc}") from exc
    return records


def resolve_record_paths(record: TileRecord, tiles_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(tiles_root).resolve()

    def resolve(relative_value: str) -> Path:
        relative = Path(relative_value)
        if relative.is_absolute():
            raise ValueError("Tile metadata paths must be relative to tiles_root.")
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Tile metadata path escapes tiles_root.")
        return destination

    return (
        resolve(record.feature_path),
        resolve(record.label_path),
        resolve(record.valid_mask_path),
    )
