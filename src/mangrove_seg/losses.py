"""Validity-aware TensorFlow losses, imported only when the ML extra is used."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mangrove_seg.optional import require_tensorflow

TensorLoss = Callable[[Any, Any], Any]


def split_encoded_target(tf: Any, y_true: Any) -> tuple[Any, Any]:
    """Split a one-channel mask or ``[mask, validity]`` encoded target."""
    target = tf.cast(y_true[..., :1], tf.float32)
    channels = y_true.shape[-1]
    if channels is not None and channels < 2:
        valid = tf.ones_like(target)
    else:
        valid = tf.cast(y_true[..., 1:2] > 0, tf.float32)
    return target, valid


def focal_dice_loss(*, gamma: float = 2.0, alpha: float = 0.8) -> TensorLoss:
    """Build a per-image focal-plus-Dice loss that ignores invalid pixels.

    Targets may be a normal one-channel mask or a two-channel tensor containing
    ``[binary_label, valid_pixel]``. Prepared datasets use the latter form.
    """
    if gamma < 0:
        raise ValueError("gamma cannot be negative.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    tf = require_tensorflow()

    def loss(y_true: Any, y_pred: Any) -> Any:
        target, valid = split_encoded_target(tf, y_true)
        probability = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        positive = -alpha * tf.pow(1 - probability, gamma) * target * tf.math.log(probability)
        negative = (
            -(1 - alpha) * tf.pow(probability, gamma) * (1 - target) * tf.math.log(1 - probability)
        )
        axes = tuple(range(1, len(target.shape)))
        valid_count = tf.maximum(tf.reduce_sum(valid, axis=axes), 1.0)
        focal = tf.reduce_sum((positive + negative) * valid, axis=axes) / valid_count

        intersection = tf.reduce_sum(target * probability * valid, axis=axes)
        denominator = tf.reduce_sum((target + probability) * valid, axis=axes)
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        return focal + (1 - dice)

    loss.__name__ = "focal_dice_loss"
    return loss


def lovasz_hinge_loss() -> TensorLoss:
    """Build a validity-aware binary Lovasz hinge loss for fine-tuning."""
    tf = require_tensorflow()

    def flat_loss(labels: Any, probabilities: Any, valid: Any) -> Any:
        selected_labels = tf.boolean_mask(labels, valid > 0)
        selected_probabilities = tf.boolean_mask(probabilities, valid > 0)

        def empty() -> Any:
            return tf.constant(0.0, dtype=tf.float32)

        def populated() -> Any:
            clipped = tf.clip_by_value(selected_probabilities, 1e-7, 1 - 1e-7)
            logits = tf.math.log(clipped) - tf.math.log1p(-clipped)
            signs = 2.0 * selected_labels - 1.0
            errors = 1.0 - logits * signs
            order = tf.argsort(errors, direction="DESCENDING")
            errors_sorted = tf.gather(errors, order)
            labels_sorted = tf.gather(selected_labels, order)
            total_positive = tf.reduce_sum(labels_sorted)
            intersection = total_positive - tf.cumsum(labels_sorted)
            union = total_positive + tf.cumsum(1.0 - labels_sorted)
            jaccard = 1.0 - intersection / tf.maximum(union, 1.0)
            gradient = tf.concat((jaccard[:1], jaccard[1:] - jaccard[:-1]), axis=0)
            return tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(gradient), axes=1)

        return tf.cond(tf.size(selected_labels) > 0, populated, empty)

    def loss(y_true: Any, y_pred: Any) -> Any:
        target, valid = split_encoded_target(tf, y_true)
        return tf.map_fn(
            lambda values: flat_loss(values[0], values[1], values[2]),
            (target, tf.cast(y_pred, tf.float32), valid),
            fn_output_signature=tf.float32,
        )

    loss.__name__ = "lovasz_hinge_loss"
    return loss


def build_loss(name: str, *, gamma: float = 2.0, alpha: float = 0.8) -> TensorLoss:
    """Resolve a configured loss name to a callable."""
    if name == "focal_dice":
        return focal_dice_loss(gamma=gamma, alpha=alpha)
    if name == "lovasz":
        return lovasz_hinge_loss()
    raise ValueError(f"Unknown loss: {name!r}.")
