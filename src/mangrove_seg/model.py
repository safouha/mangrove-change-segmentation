"""Squeeze-and-excitation U-Net with optional multi-scale supervision heads."""

from __future__ import annotations

from typing import Any

from mangrove_seg.config import ModelConfig
from mangrove_seg.optional import require_tensorflow


def _normalize(layers: Any, value: Any, normalization: str, name: str) -> Any:
    if normalization == "batch":
        return layers.BatchNormalization(name=name)(value)
    return value


def _convolution_block(
    layers: Any,
    value: Any,
    filters: int,
    *,
    dropout: float,
    normalization: str,
    name: str,
) -> Any:
    residual = layers.Conv2D(filters, 1, padding="same", name=f"{name}_projection")(value)
    value = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(value)
    value = _normalize(layers, value, normalization, f"{name}_norm1")
    value = layers.Activation("relu", name=f"{name}_relu1")(value)
    if dropout:
        value = layers.SpatialDropout2D(dropout, name=f"{name}_dropout")(value)
    value = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(value)
    value = _normalize(layers, value, normalization, f"{name}_norm2")
    value = layers.Add(name=f"{name}_residual")([value, residual])
    return layers.Activation("relu", name=f"{name}_relu2")(value)


def _squeeze_excitation(layers: Any, value: Any, filters: int, reduction: int, name: str) -> Any:
    hidden = max(filters // reduction, 1)
    weights = layers.GlobalAveragePooling2D(name=f"{name}_pool")(value)
    weights = layers.Dense(hidden, activation="relu", name=f"{name}_reduce")(weights)
    weights = layers.Dense(filters, activation="sigmoid", name=f"{name}_expand")(weights)
    weights = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(weights)
    return layers.Multiply(name=f"{name}_scale")([value, weights])


def output_names(config: ModelConfig) -> tuple[str, ...]:
    """Return stable model output names for dataset and loss construction."""
    if not config.deep_supervision or len(config.filters) < 2:
        return ("mask",)
    auxiliary_count = min(2, len(config.filters) - 1)
    return ("mask", *(f"aux_{index}" for index in range(1, auxiliary_count + 1)))


def build_unet(
    *,
    tile_size: int,
    channels: int,
    config: ModelConfig,
) -> Any:
    """Construct a residual SE U-Net without importing TensorFlow at package import time."""
    if tile_size <= 0 or channels <= 0:
        raise ValueError("tile_size and channels must be positive.")
    tf = require_tensorflow()
    layers = tf.keras.layers

    inputs = layers.Input((tile_size, tile_size, channels), name="embeddings")
    value = inputs
    skips: list[Any] = []
    for index, filters in enumerate(config.filters, start=1):
        value = _convolution_block(
            layers,
            value,
            filters,
            dropout=config.dropout,
            normalization=config.normalization,
            name=f"encoder_{index}",
        )
        value = _squeeze_excitation(
            layers, value, filters, config.se_reduction, f"encoder_{index}_se"
        )
        skips.append(value)
        value = layers.MaxPool2D(name=f"encoder_{index}_pool")(value)

    value = _convolution_block(
        layers,
        value,
        config.filters[-1] * 2,
        dropout=config.dropout,
        normalization=config.normalization,
        name="bridge",
    )

    decoder_features: list[Any] = []
    for index, (filters, skip) in enumerate(
        zip(reversed(config.filters), reversed(skips), strict=True), start=1
    ):
        value = layers.Conv2DTranspose(
            filters, 2, strides=2, padding="same", name=f"decoder_{index}_up"
        )(value)
        value = layers.Concatenate(name=f"decoder_{index}_skip")([value, skip])
        value = _convolution_block(
            layers,
            value,
            filters,
            dropout=config.dropout,
            normalization=config.normalization,
            name=f"decoder_{index}",
        )
        decoder_features.append(value)

    outputs: dict[str, Any] = {
        "mask": layers.Conv2D(1, 1, activation="sigmoid", name="mask")(decoder_features[-1])
    }
    auxiliary_sources = decoder_features[:-1][-2:]
    for index, source in enumerate(reversed(auxiliary_sources), start=1):
        auxiliary = layers.Conv2D(1, 1, activation="sigmoid", name=f"aux_{index}_raw")(source)
        outputs[f"aux_{index}"] = layers.Resizing(
            tile_size,
            tile_size,
            interpolation="bilinear",
            name=f"aux_{index}",
        )(auxiliary)

    if not config.deep_supervision:
        return tf.keras.Model(inputs=inputs, outputs=outputs["mask"], name="mangrove_se_unet")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mangrove_se_unet")
