from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "project.toml"
    path.write_text(
        """
[project]
seed = 17

[data]
embeddings_root = "data/embeddings"
labels_root = "data/labels"
tiles_root = "workspace/tiles"
artifacts_root = "workspace/artifacts"
regions = ["alpha", "bravo", "charlie", "delta", "echo"]
years = [2019, 2020, 2021]
channels = 4
tile_size = 16
train_stride = 8
evaluation_stride = 16
minimum_positive_fraction = 0.01
maximum_negative_tiles_per_region = 20

[split]
mode = "hash"
validation_fraction = 0.2
test_fraction = 0.2
train_regions = []
validation_regions = []
test_regions = []

[model]
filters = [8, 16]
dropout = 0.1
normalization = "batch"
se_reduction = 4
deep_supervision = true

[training]
batch_size = 2
epochs = 3
learning_rate = 0.001
weight_decay = 0.0
clipnorm = 1.0
focal_gamma = 2.0
focal_alpha = 0.8
loss = "focal_dice"
early_stopping_patience = 2
reduce_lr_patience = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path
