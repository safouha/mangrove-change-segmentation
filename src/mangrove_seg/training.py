"""Optional TensorFlow dataset construction and training orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mangrove_seg.config import ProjectConfig
from mangrove_seg.data import load_tile
from mangrove_seg.losses import build_loss, split_encoded_target
from mangrove_seg.model import build_unet, output_names
from mangrove_seg.optional import require_tensorflow
from mangrove_seg.records import TileRecord, read_records


@dataclass(frozen=True)
class TrainingArtifacts:
    best_model: Path
    final_model: Path
    history: Path
    training_tiles: int
    validation_tiles: int

    def as_dict(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = asdict(self)
        payload["best_model"] = str(self.best_model)
        payload["final_model"] = str(self.final_model)
        payload["history"] = str(self.history)
        return payload


def _targets(encoded: Any, names: tuple[str, ...]) -> Any:
    return encoded if len(names) == 1 else {name: encoded for name in names}


def _tile_generator(
    records: list[TileRecord],
    config: ProjectConfig,
    names: tuple[str, ...],
) -> Iterator[tuple[Any, Any]]:
    for record in records:
        tile = load_tile(
            record,
            config.data.tiles_root,
            expected_channels=config.data.channels,
            expected_size=config.data.tile_size,
        )
        yield tile.features, _targets(tile.encoded_target(), names)


def create_dataset(
    records: list[TileRecord],
    config: ProjectConfig,
    *,
    training: bool,
) -> Any:
    """Create a streaming ``tf.data`` pipeline from prepared metadata."""
    if not records:
        raise ValueError("Cannot create a dataset from an empty metadata list.")
    tf = require_tensorflow()
    names = output_names(config.model)
    feature_spec = tf.TensorSpec(
        shape=(config.data.tile_size, config.data.tile_size, config.data.channels),
        dtype=tf.float32,
    )
    target_spec = tf.TensorSpec(
        shape=(config.data.tile_size, config.data.tile_size, 2),
        dtype=tf.float32,
    )
    output_target_spec: Any = (
        target_spec if len(names) == 1 else {name: target_spec for name in names}
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: _tile_generator(records, config, names),
        output_signature=(feature_spec, output_target_spec),
    )
    if training:
        dataset = dataset.shuffle(
            min(len(records), 10_000),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)


def _masked_metric(name: str) -> Any:
    tf = require_tensorflow()

    def metric(y_true: Any, y_pred: Any) -> Any:
        target, valid = split_encoded_target(tf, y_true)
        prediction = tf.cast(y_pred >= 0.5, tf.float32)
        if name == "accuracy":
            correct = tf.cast(tf.equal(target, prediction), tf.float32) * valid
            return tf.reduce_sum(correct) / tf.maximum(tf.reduce_sum(valid), 1.0)
        intersection = tf.reduce_sum(target * prediction * valid)
        if name == "iou":
            union = tf.reduce_sum(tf.cast((target + prediction) > 0, tf.float32) * valid)
            return intersection / tf.maximum(union, 1.0)
        denominator = tf.reduce_sum((target + prediction) * valid)
        return (2 * intersection) / tf.maximum(denominator, 1.0)

    metric.__name__ = f"masked_{name}"
    return metric


def compile_model(model: Any, config: ProjectConfig) -> Any:
    """Compile a model with matching losses for its supervision heads."""
    tf = require_tensorflow()
    names = output_names(config.model)
    loss = build_loss(
        config.training.loss,
        gamma=config.training.focal_gamma,
        alpha=config.training.focal_alpha,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clipnorm=config.training.clipnorm,
    )
    metrics = [_masked_metric("accuracy"), _masked_metric("iou"), _masked_metric("dice")]
    if len(names) == 1:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        losses = {name: loss for name in names}
        loss_weights = {"mask": 1.0, **{name: 0.3 for name in names if name != "mask"}}
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics={"mask": metrics},
        )
    return model


def train(config: ProjectConfig) -> TrainingArtifacts:
    """Train from prepared tile manifests and persist models plus raw history."""
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(config.seed)
    train_records = read_records(config.data.tiles_root / "train" / "metadata.jsonl")
    validation_records = read_records(config.data.tiles_root / "validation" / "metadata.jsonl")
    train_dataset = create_dataset(train_records, config, training=True)
    validation_dataset = create_dataset(validation_records, config, training=False)

    artifacts = config.data.artifacts_root
    artifacts.mkdir(parents=True, exist_ok=True)
    best_path = artifacts / "best-model.keras"
    final_path = artifacts / "final-model.keras"
    history_path = artifacts / "training-history.json"

    model = compile_model(
        build_unet(
            tile_size=config.data.tile_size,
            channels=config.data.channels,
            config=config.model,
        ),
        config,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=config.training.reduce_lr_patience,
            factor=0.5,
            min_lr=1e-7,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.training.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(final_path)
    serializable_history = {
        name: [float(value) for value in values] for name, values in history.history.items()
    }
    temporary = history_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(serializable_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(history_path)
    return TrainingArtifacts(
        best_model=best_path,
        final_model=final_path,
        history=history_path,
        training_tiles=len(train_records),
        validation_tiles=len(validation_records),
    )
