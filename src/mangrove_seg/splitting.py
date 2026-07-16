"""Deterministic region-level splitting to prevent spatial tile leakage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def assignment(self) -> dict[str, str]:
        return {
            **{region: "train" for region in self.train},
            **{region: "validation" for region in self.validation},
            **{region: "test" for region in self.test},
        }


def _count(size: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, int(size * fraction + 0.5))


def deterministic_region_split(
    regions: Iterable[str],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> RegionSplit:
    """Split whole regions using stable SHA-256 ranking.

    The result is independent of input order and Python's random implementation.
    Adding a region can change allocations, so the CLI persists the final manifest.
    """
    unique = sorted(set(regions))
    if not unique:
        raise ValueError("At least one region is required.")
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave room for training regions.")

    ranked = sorted(
        unique,
        key=lambda region: hashlib.sha256(f"{seed}\0{region}".encode()).digest(),
    )
    test_count = _count(len(ranked), test_fraction)
    validation_count = _count(len(ranked), validation_fraction)
    if test_count + validation_count >= len(ranked):
        raise ValueError(
            "The requested non-training fractions require more regions to keep a training split."
        )

    test = tuple(sorted(ranked[:test_count]))
    validation = tuple(sorted(ranked[test_count : test_count + validation_count]))
    train = tuple(sorted(ranked[test_count + validation_count :]))
    return RegionSplit(train=train, validation=validation, test=test)


def explicit_region_split(
    all_regions: Iterable[str],
    *,
    train: Iterable[str],
    validation: Iterable[str],
    test: Iterable[str],
) -> RegionSplit:
    """Validate and normalize an explicitly configured split."""
    result = RegionSplit(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test)))
    combined = result.train + result.validation + result.test
    if len(combined) != len(set(combined)):
        raise ValueError("Explicit region splits overlap.")
    if set(combined) != set(all_regions):
        missing = sorted(set(all_regions) - set(combined))
        extra = sorted(set(combined) - set(all_regions))
        raise ValueError(f"Explicit split mismatch; missing={missing}, extra={extra}.")
    return result
