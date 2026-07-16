from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mangrove_seg.cli import main
from mangrove_seg.config import ModelConfig
from mangrove_seg.losses import build_loss, focal_dice_loss
from mangrove_seg.model import output_names


def test_validate_config_only_cli(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(config_file), "--config-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"] == "valid"
    assert payload["inventory"]["complete"] is False


def test_validate_cli_fails_for_missing_data(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate", str(config_file)]) == 2
    assert "Missing 20 configured raster" in capsys.readouterr().err


def test_split_cli_is_json(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["split", str(config_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"train", "validation", "test"}
    assert sum(len(value) for value in payload.values()) == 5


def test_metrics_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = tmp_path / "labels.npy"
    predictions = tmp_path / "predictions.npy"
    np.save(labels, np.array([1, 0, 1]), allow_pickle=False)
    np.save(predictions, np.array([0.9, 0.2, 0.1]), allow_pickle=False)
    assert main(["metrics", str(labels), str(predictions)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["true_positive"] == 1
    assert payload["false_negative"] == 1


def test_change_cli_saves_encoded_map(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    previous = tmp_path / "previous.npy"
    current = tmp_path / "current.npy"
    destination = tmp_path / "change.npy"
    np.save(previous, np.array([0, 1]), allow_pickle=False)
    np.save(current, np.array([1, 0]), allow_pickle=False)
    assert main(["change", str(previous), str(current), "--output", str(destination)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gain_pixels"] == 1
    assert payload["loss_pixels"] == 1
    assert destination.is_file()


def test_cli_reports_missing_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["split", "/does/not/exist.toml"]) == 2
    assert "error:" in capsys.readouterr().err


def test_output_names_match_deep_supervision_contract() -> None:
    deep = ModelConfig(filters=(8, 16, 32), deep_supervision=True)
    shallow = ModelConfig(filters=(8, 16, 32), deep_supervision=False)
    assert output_names(deep) == ("mask", "aux_1", "aux_2")
    assert output_names(shallow) == ("mask",)


def test_loss_configuration_is_validated_before_tensorflow_import() -> None:
    with pytest.raises(ValueError, match="alpha"):
        focal_dice_loss(alpha=-0.1)
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("not-a-loss")
