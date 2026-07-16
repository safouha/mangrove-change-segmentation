"""Mangrove segmentation and change-analysis utilities."""

from mangrove_seg.change import ChangeSummary, change_map, summarize_change
from mangrove_seg.config import ProjectConfig, load_config
from mangrove_seg.discovery import DataInventory, TemporalRasterPair, discover_temporal_pairs
from mangrove_seg.evaluation import BinaryMetrics, binary_segmentation_metrics
from mangrove_seg.splitting import RegionSplit, deterministic_region_split

__all__ = [
    "BinaryMetrics",
    "ChangeSummary",
    "DataInventory",
    "ProjectConfig",
    "RegionSplit",
    "TemporalRasterPair",
    "binary_segmentation_metrics",
    "change_map",
    "deterministic_region_split",
    "discover_temporal_pairs",
    "load_config",
    "summarize_change",
]

__version__ = "0.1.0"
