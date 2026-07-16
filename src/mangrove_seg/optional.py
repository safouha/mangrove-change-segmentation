"""Helpers for optional dependency boundaries."""

from __future__ import annotations

from typing import Any


class MissingOptionalDependency(ImportError):
    """Raised when a command needs an extra that is not installed."""


def require_tensorflow() -> Any:
    """Import TensorFlow or explain which project extra provides it."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingOptionalDependency(
            "TensorFlow is required for this operation. Install the ML extra with "
            "`pip install -e '.[ml]'`."
        ) from exc
    return tf


def require_rasterio() -> Any:
    """Import Rasterio or explain which project extra provides it."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingOptionalDependency(
            "Rasterio is required for this operation. Install the geospatial extra with "
            "`pip install -e '.[geo]'`."
        ) from exc
    return rasterio
