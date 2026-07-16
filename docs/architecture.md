# Architecture and design decisions

## Component map

| Module | Responsibility | Heavy dependency |
| --- | --- | --- |
| `config` | Typed TOML loading and cross-field validation | None |
| `discovery` | Declared temporal-pair inventory and missing-file report | None |
| `splitting` | Deterministic or explicit whole-region assignment | None |
| `geospatial` | Raster alignment and GeoTIFF output | Rasterio |
| `tiling` | Edge-complete windows, padding, extraction, tile statistics | None |
| `prepare` | End-to-end raster-to-tile orchestration and manifests | Rasterio |
| `records` / `data` | Portable metadata and validated NumPy loading | None |
| `model` / `losses` | Residual SE U-Net and validity-aware objectives | TensorFlow |
| `training` | Streaming datasets, compilation, callbacks, artifacts | TensorFlow |
| `inference` | Overlap-averaged prediction for large arrays | None |
| `evaluation` / `change` | Binary metrics and temporal change categories | None |

Optional imports happen inside the function that requires them. Importing `mangrove_seg`, running
array tests, or inspecting a split therefore does not require a geospatial or ML installation.

## Leakage boundary

The unit of randomization is a **region**, never a tile. This matters because adjacent and
overlapping satellite tiles share most of their pixels. A random tile split would make validation
look better while measuring spatial memorization.

Hash-mode splitting ranks `SHA-256(seed + region)` values. Test and validation counts are rounded
from configured fractions, with at least one region assigned when a nonzero fraction is requested.
The operation fails when there are too few regions to leave a training set. Explicit mode requires
disjoint lists that cover the declared region set exactly.

The final assignment is written to `region-split.json`. Downstream experiment tracking should save
that file with the model artifacts.

## Raster alignment

The label raster defines output CRS, affine transform, height, and width. Each continuous feature
band is reprojected with bilinear interpolation. The feature dataset mask is reprojected separately
with nearest-neighbor interpolation. Labels are read directly on their native grid and converted to
a binary mask; they are never bilinearly resampled.

A pixel is valid only when:

1. the label dataset mask is valid;
2. the reprojected feature mask is valid; and
3. every feature channel is finite.

Invalid features and labels are zeroed for safe storage, but the validity mask remains authoritative.

## Tile selection

Windows are row-major and always include the far image edge. Images smaller than one tile are
zero-padded, with padded pixels marked invalid. Training can use overlapping windows, while
validation and test normally use a full-tile stride.

Positive prevalence is calculated only over valid pixels. Every tile meeting
`minimum_positive_fraction` is retained. Remaining background tiles are ordered by a stable hash of
seed, region, input year, row, and column, then capped per training region. Validation and test data
are not downsampled, preserving real prevalence for evaluation.

## Model targets

Prepared labels and validity masks are stacked into a two-channel training target:

```text
target[..., 0] = binary mangrove label
target[..., 1] = valid-pixel indicator
```

The model still predicts one probability channel. Custom focal-Dice and Lovasz losses unpack the
encoded target and exclude invalid pixels. The main output has weight `1.0`; each auxiliary output
has weight `0.3`. Auxiliary heads are training aids and tiled inference always selects `mask`.

## Reproducibility boundaries

The package fixes Python-side split/sampling behavior and calls TensorFlow's unified seed helper.
Bit-for-bit GPU training is not promised: TensorFlow kernels, drivers, and hardware can still differ.
For a reproducible result, record the TOML file, split manifest, package lock, hardware, TensorFlow
version, learned threshold, and the exact input-data checksums.

## Extension points

- Add a new raster naming convention by producing `TemporalRasterPair` values; keep the region as
  the split unit.
- Add a model by preserving a primary output named `mask` with shape `(B, H, W, 1)`.
- Add metrics in `evaluation.py` when they can operate on NumPy arrays and respect `valid_mask`.
- Add area summaries only when the reference CRS and pixel-area calculation are explicit. Geographic
  degrees must not be treated as square metres.
