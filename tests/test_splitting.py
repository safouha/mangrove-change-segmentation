from __future__ import annotations

import pytest

from mangrove_seg.splitting import deterministic_region_split, explicit_region_split

REGIONS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]


def test_hash_split_is_independent_of_input_order() -> None:
    forward = deterministic_region_split(
        REGIONS, seed=7, validation_fraction=0.2, test_fraction=0.2
    )
    backward = deterministic_region_split(
        reversed(REGIONS), seed=7, validation_fraction=0.2, test_fraction=0.2
    )
    assert forward == backward


def test_hash_split_is_disjoint_and_complete() -> None:
    split = deterministic_region_split(REGIONS, seed=11, validation_fraction=0.2, test_fraction=0.2)
    combined = split.train + split.validation + split.test
    assert set(combined) == set(REGIONS)
    assert len(combined) == len(set(combined))
    assert set(split.assignment().values()) == {"train", "validation", "test"}


def test_zero_holdout_fractions_are_supported() -> None:
    split = deterministic_region_split(REGIONS, seed=1, validation_fraction=0, test_fraction=0)
    assert split.train == tuple(sorted(REGIONS))
    assert split.validation == ()
    assert split.test == ()


@pytest.mark.parametrize(
    ("validation", "test"),
    [(-0.1, 0.2), (0.2, 1.0), (0.5, 0.5)],
)
def test_invalid_fractions_are_rejected(validation: float, test: float) -> None:
    with pytest.raises(ValueError):
        deterministic_region_split(
            REGIONS,
            seed=1,
            validation_fraction=validation,
            test_fraction=test,
        )


def test_too_few_regions_fails_instead_of_empty_training() -> None:
    with pytest.raises(ValueError, match="more regions"):
        deterministic_region_split(
            ["only", "other"], seed=1, validation_fraction=0.2, test_fraction=0.2
        )


def test_explicit_split_is_normalized() -> None:
    split = explicit_region_split(
        REGIONS,
        train=["delta", "alpha", "bravo", "charlie"],
        validation=["echo"],
        test=["foxtrot"],
    )
    assert split.train == ("alpha", "bravo", "charlie", "delta")


def test_explicit_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        explicit_region_split(
            ["alpha", "bravo"], train=["alpha"], validation=["alpha"], test=["bravo"]
        )


def test_explicit_mismatch_reports_missing_and_extra() -> None:
    with pytest.raises(ValueError, match=r"missing=.*bravo"):
        explicit_region_split(["alpha", "bravo"], train=["alpha"], validation=[], test=["charlie"])
