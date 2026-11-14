"""
MAGFiLO Panoptic Solar Filament Segmentation Package.

Published methodological foundations:
- MAGFiLO Dataset & Multi-Annotator Spine Protocol:
  Ahmadzadeh et al., Scientific Data (2024), doi:10.1038/s41597-024-03876-y.
- Centerline Topology Loss (soft-clDice):
  Shit et al., CVPR 2021.
- Panoptic Quality (PQ = SQ * RQ):
  Kirillov et al., CVPR 2019.
"""

from .metrics import compute_exact_magfilo_metrics
from .network import MagfiloPanopticUNet, SolarMultiTaskLoss, soft_cldice_loss
from .postprocessing import (
    FEATURE_NAMES,
    apply_spine_bridging_to_selection,
    extract_disk_component_tree,
    solve_tree_dp_antichain,
    train_and_evaluate_oof_lgbm,
)
from .preprocessing import (
    IMG_SIZE,
    disk_crop_box,
    estimate_solar_disk,
    preprocess_solar_disk,
)
from .rle import decode_rle_counts, encode_rle_counts, rle_dict_from_counts

__all__ = [
    "IMG_SIZE",
    "FEATURE_NAMES",
    "encode_rle_counts",
    "decode_rle_counts",
    "rle_dict_from_counts",
    "compute_exact_magfilo_metrics",
    "estimate_solar_disk",
    "preprocess_solar_disk",
    "disk_crop_box",
    "MagfiloPanopticUNet",
    "SolarMultiTaskLoss",
    "soft_cldice_loss",
    "extract_disk_component_tree",
    "solve_tree_dp_antichain",
    "apply_spine_bridging_to_selection",
    "train_and_evaluate_oof_lgbm",
]
