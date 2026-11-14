#!/usr/bin/env python3
"""
MAGFiLO Panoptic Solar Filament Segmentation Pipeline.

Grounded strictly in published astrophysical and computer vision methods:
- MAGFiLO Dataset & Spine/Multi-Annotator Protocol:
  Ahmadzadeh et al., "MAGFiLO: Manually Annotated GONG Filaments from H-alpha
  Observations", Scientific Data (2024), doi:10.1038/s41597-024-03876-y.
- Centerline Topology Loss (soft-clDice):
  Shit et al., "clDice - a Novel Topology-Preserving Loss Function for Tubular
  Structure Segmentation", CVPR 2021.
- Panoptic Quality (PQ = SQ * RQ):
  Kirillov et al., "Panoptic Segmentation", CVPR 2019.

Pipeline Stages:
  Stage 0: Parse MAGFiLO COCO JSON, group by physical solar disk and
           YYYYMMDD observation date into 3 zero-leakage folds, and synthesize
           per-disk soft consensus foreground (y_fg), centerline/spine
           (y_spine), and same-annotator gap corridor (y_sep) targets.
  Stage P: Solar-disk center/radius estimation (cx, cy, R), radial median
           Center-to-Limb Variation (CLV) flat-fielding, off-limb masking
           (r > 0.995 R), normalized-convolution local absorption contrast
           (B_sigma48 - F) / B_sigma48, and heliocentric limb cosine mu.
  Stage N: 3-head (fg, spine, sep) MagfiloPanopticUNet (timm encoder with
           torchvision fallback), boundary-upweighted BCE + SoftDice + 0.25 *
           soft_clDice. Auto-adapts to T4 GPU (1536x1536 tight disk crop,
           18 epochs, AMP fp16, 4-view flip TTA) or 4-core CPU fallback
           (scale-matched 384x384 patches from 1024x1024 disk crop, batch-1
           dynamic step cap < 2.2h total runtime, single-pass 1024x1024
           channels_last inference upsampled to 2048x2048).
           Saves compact float16 weights (< 30 MB/file, < 80 MB total).
  Stage H: Effective map P_eff = clip(0.75*P_fg + 0.25*P_spine - 0.35*P_sep),
           laminar component-tree candidate extraction over 7 superlevels,
           spine-tangent corridor bridging, astrophysical sunspot rejection,
           3-fold OOF LightGBM expected-PQ regressor u_hat(c), exact bottom-up
           tree DP maximizing sum(u_hat(c) - tau) over pairwise-disjoint
           antichains, tau* tuning on exact competition PQ, rubric metrics &
           diagnostic plots, and 3 validated 2048x2048 RLE submission CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision.models as tv_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

IMG_SIZE = 2048
MAX_CPU_RUNTIME_SEC = 7920.0  # Strict 2.20h ceiling (< 2.3h CPU cap)
TREE_LEVELS = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68)
FALLBACK_LEVELS = (0.14, 0.09, 0.05)
MIN_COMPONENT_AREA = 120
FALLBACK_MIN_AREA = 24
BRIDGE_MAX_GAP_PX = 14.0
BRIDGE_MAX_P_SEP = 0.24
BRIDGE_MIN_DOT = 0.60

FEATURE_NAMES = [
    "log_area",
    "area_px",
    "perimeter",
    "bbox_w",
    "bbox_h",
    "bbox_fill",
    "circularity",
    "eccentricity",
    "elongation",
    "solidity",
    "skel_len_ratio",
    "mean_width",
    "level_thresh",
    "n_children",
    "mean_mu",
    "min_mu",
    "mean_p_fg",
    "p90_p_fg",
    "p10_p_fg",
    "mean_p_spine",
    "p90_p_spine",
    "spine_cov",
    "mean_p_sep",
    "interior_p_sep_p90",
    "boundary_p_sep",
    "mean_contrast",
    "p90_contrast",
    "mean_norm_clv",
    "min_norm_clv",
    "umbral_dark_frac",
    "sunspot_score",
]


def log_msg(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pycocotools RLE Utilities & Exact Official Competition Metric
# ---------------------------------------------------------------------------


def encode_rle_counts(mask_u8: np.ndarray) -> str:
    """Encode a 2D binary mask into compressed Fortran-order COCO RLE counts."""
    rle = mask_util.encode(np.asfortranarray(mask_u8.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def decode_rle_counts(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    """Decode compressed COCO RLE counts string into uint8 binary mask."""
    return mask_util.decode({"size": [h, w], "counts": counts}).astype(np.uint8)


def rle_dict_from_counts(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> Dict[str, Any]:
    return {"size": [h, w], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}


def compute_exact_magfilo_metrics(
    gt_records: List[Dict[str, Any]],
    pred_by_disk: Dict[str, List[str]],
    img_h: int = IMG_SIZE,
    img_w: int = IMG_SIZE,
) -> Dict[str, Any]:
    """
    Exact reproduction of the competition self-evaluation-notebook.ipynb metric.
    Evaluates PQ (IoU > 0.5), SQ, RQ, IoU > 0 / Dice > 0 distributions, and
    1:n / n:1 matching degree counts across all multi-annotator records.
    """
    tp_ious: List[float] = []
    fp_count = 0
    fn_count = 0
    nonzero_ious: List[float] = []
    nonzero_dices: List[float] = []
    gt_match_degrees: List[int] = []
    pred_match_degrees: List[int] = []

    for rec in gt_records:
        disk_stem = rec["disk_stem"]
        gt_rles = rec["gt_rles"]
        pred_counts = pred_by_disk.get(disk_stem, [])
        n_gt = len(gt_rles)
        n_pred = len(pred_counts)

        if n_gt == 0:
            fp_count += n_pred
            pred_match_degrees.extend([0] * n_pred)
            continue
        if n_pred == 0:
            fn_count += n_gt
            gt_match_degrees.extend([0] * n_gt)
            continue

        pred_rles = [rle_dict_from_counts(c, img_h, img_w) for c in pred_counts]
        iou_mat = mask_util.iou(pred_rles, gt_rles, [0] * n_gt).T.astype(np.float64)

        hit_50 = iou_mat > 0.5
        tp_vals = iou_mat[hit_50]
        if tp_vals.size > 0:
            tp_ious.extend(tp_vals.tolist())
        fp_count += int((hit_50.sum(axis=0) == 0).sum())
        fn_count += int((hit_50.sum(axis=1) == 0).sum())

        nz = iou_mat[iou_mat > 0.0]
        if nz.size > 0:
            nonzero_ious.extend(nz.tolist())
            nonzero_dices.extend((2.0 * nz / (1.0 + nz)).tolist())

        hit_any = iou_mat > 0.0
        gt_match_degrees.extend(hit_any.sum(axis=1).tolist())
        pred_match_degrees.extend(hit_any.sum(axis=0).tolist())

    tp_count = len(tp_ious)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = float(sum(tp_ious) / denom) if denom > 0 else 0.0
    sq = float(np.mean(tp_ious)) if tp_count > 0 else 0.0
    rq = float(tp_count / denom) if denom > 0 else 0.0

    iou_arr = np.asarray(nonzero_ious, dtype=np.float64)
    dice_arr = np.asarray(nonzero_dices, dtype=np.float64)
    gt_deg = Counter(gt_match_degrees)
    pred_deg = Counter(pred_match_degrees)

    def _pcts(arr: np.ndarray) -> Tuple[float, float, float, float]:
        if arr.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(arr.mean()),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 75)),
        )

    iou_mean, iou_p25, iou_p50, iou_p75 = _pcts(iou_arr)
    dice_mean, dice_p25, dice_p50, dice_p75 = _pcts(dice_arr)

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "iou_mean": iou_mean,
        "iou_p25": iou_p25,
        "iou_p50": iou_p50,
        "iou_p75": iou_p75,
        "dice_mean": dice_mean,
        "dice_p25": dice_p25,
        "dice_p50": dice_p50,
        "dice_p75": dice_p75,
        "gt_1_to_0_miss": int(gt_deg.get(0, 0)),
        "gt_1_to_1": int(gt_deg.get(1, 0)),
        "gt_1_to_many": int(sum(v for k, v in gt_deg.items() if k >= 2)),
        "pred_0_to_1_fp": int(pred_deg.get(0, 0)),
        "pred_1_to_1": int(pred_deg.get(1, 0)),
        "pred_many_to_1": int(sum(v for k, v in pred_deg.items() if k >= 2)),
        "gt_degree_hist": {int(k): int(v) for k, v in sorted(gt_deg.items())},
        "pred_degree_hist": {int(k): int(v) for k, v in sorted(pred_deg.items())},
        "nonzero_ious": iou_arr,
        "nonzero_dices": dice_arr,
    }


# ---------------------------------------------------------------------------
# Stage P: Astrophysical Solar-Disk Preprocessing (Vectorized Fast Path)
# ---------------------------------------------------------------------------


def _fit_circle_kasa(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic Kasa least-squares circle fit."""
    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(float(c + cx * cx + cy * cy), 1.0))
    return float(cx), float(cy), float(r)


def estimate_solar_disk(gray_u8: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate solar-disk center and radius (cx, cy, R) on an H-alpha image
    via downsampled Otsu segmentation, moment init, and 72-ray radial 50%-limb
    Kasa circle regression with MAD outlier rejection.
    """
    h, w = gray_u8.shape[:2]
    ds_size = 512
    scale_x = w / float(ds_size)
    scale_y = h / float(ds_size)
    ds = cv2.resize(gray_u8, (ds_size, ds_size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(ds, (5, 5), 1.0)

    otsu_thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(10.0, float(otsu_thr) * 0.45)
    mask = (blur > thr).astype(np.uint8)

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lbl <= 1:
        return w * 0.5, h * 0.5, min(h, w) * 0.44

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_id = int(np.argmax(areas)) + 1
    disk = ndimage.binary_fill_holes(labels == best_id).astype(np.uint8)

    M = cv2.moments(disk)
    if M["m00"] < 100.0:
        return w * 0.5, h * 0.5, min(h, w) * 0.44
    cx0 = float(M["m10"] / M["m00"])
    cy0 = float(M["m01"] / M["m00"])
    r0 = math.sqrt(float(M["m00"]) / math.pi)

    thetas = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    radii_grid = np.linspace(max(10.0, 0.88 * r0), min(ds_size * 0.7, 1.12 * r0), 48)
    limb_x: List[float] = []
    limb_y: List[float] = []
    limb_r: List[float] = []

    blur_f = blur.astype(np.float32)
    for th in thetas:
        ct, st = math.cos(th), math.sin(th)
        rx = np.clip(np.round(cx0 + radii_grid * ct).astype(int), 0, ds_size - 1)
        ry = np.clip(np.round(cy0 + radii_grid * st).astype(int), 0, ds_size - 1)
        prof = blur_f[ry, rx]
        inner_level = float(np.median(prof[:8]))
        outer_level = float(np.median(prof[-8:]))
        if inner_level <= outer_level + 10.0:
            continue
        mid = 0.5 * (inner_level + outer_level)
        below = np.where(prof <= mid)[0]
        if below.size == 0:
            continue
        idx = int(below[0])
        if idx == 0:
            r_cross = float(radii_grid[0])
        else:
            v0, v1 = float(prof[idx - 1]), float(prof[idx])
            alpha = (mid - v0) / (v1 - v0 - 1e-6) if abs(v1 - v0) > 1e-6 else 0.5
            alpha = float(np.clip(alpha, 0.0, 1.0))
            r_cross = float(radii_grid[idx - 1] + alpha * (radii_grid[idx] - radii_grid[idx - 1]))
        limb_r.append(r_cross)
        limb_x.append(cx0 + r_cross * ct)
        limb_y.append(cy0 + r_cross * st)

    if len(limb_r) >= 24:
        r_arr = np.asarray(limb_r, dtype=np.float64)
        med_r = float(np.median(r_arr))
        mad = float(np.median(np.abs(r_arr - med_r))) + 1e-3
        keep = np.abs(r_arr - med_r) <= max(2.5, 3.0 * 1.4826 * mad)
        if keep.sum() >= 16:
            xs = np.asarray(limb_x, dtype=np.float64)[keep]
            ys = np.asarray(limb_y, dtype=np.float64)[keep]
            cx_fit, cy_fit, r_fit = _fit_circle_kasa(xs, ys)
            if 0.25 * ds_size <= r_fit <= 0.55 * ds_size and 0.1 * ds_size <= cx_fit <= 0.9 * ds_size:
                return cx_fit * scale_x, cy_fit * scale_y, r_fit * 0.5 * (scale_x + scale_y)

    return cx0 * scale_x, cy0 * scale_y, r0 * 0.5 * (scale_x + scale_y)


def preprocess_solar_disk(
    gray_u8: np.ndarray,
    disk_params: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute 3-channel solar-physics input [norm_CLV, contrast, mu] and off-limb mask.
    Vectorized multi-scale evaluation (~50-90 ms per 2048x2048 solar disk).
    """
    h, w = gray_u8.shape[:2]
    if disk_params is None:
        cx, cy, R = estimate_solar_disk(gray_u8)
    else:
        cx, cy, R = disk_params
    R = max(float(R), 64.0)

    # Radial median CLV curve via single-argsort on 256x256
    ds_prof = 256
    ds = cv2.resize(gray_u8, (ds_prof, ds_prof), interpolation=cv2.INTER_AREA).astype(np.float32)
    sx_p = ds_prof / float(w)
    sy_p = ds_prof / float(h)
    yy_p, xx_p = np.mgrid[0:ds_prof, 0:ds_prof].astype(np.float32)
    rho_p = np.sqrt(((xx_p - cx * sx_p) / (R * sx_p)) ** 2 + ((yy_p - cy * sy_p) / (R * sy_p)) ** 2).ravel()
    ds_f = ds.ravel()

    n_bins = 128
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    global_med = float(np.median(ds_f[rho_p <= 0.90])) if np.any(rho_p <= 0.90) else 100.0

    bins = (rho_p * n_bins).astype(np.int32)
    valid_b = (bins >= 0) & (bins < n_bins)
    b_v = bins[valid_b]
    v_v = ds_f[valid_b]
    order = np.argsort(b_v.astype(np.float64) * 1e4 + v_v.astype(np.float64))
    b_s = b_v[order]
    v_s = v_v[order]
    ends = np.searchsorted(b_s, np.arange(n_bins + 1))

    prof = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        lo, hi = int(ends[b]), int(ends[b + 1])
        if hi > lo:
            prof[b] = float(v_s[(lo + hi) // 2])
        else:
            prof[b] = prof[b - 1] if b > 0 else global_med
    prof = ndimage.gaussian_filter1d(prof, sigma=2.0, mode="nearest")
    prof = np.maximum(prof, 8.0)

    # Interpolate smooth CLV field at 512x512 and upsample bilinearly
    ds_clv = 512
    sx_c = ds_clv / float(w)
    sy_c = ds_clv / float(h)
    yy_c, xx_c = np.mgrid[0:ds_clv, 0:ds_clv].astype(np.float32)
    rho_c = np.sqrt(((xx_c - cx * sx_c) / (R * sx_c)) ** 2 + ((yy_c - cy * sy_c) / (R * sy_c)) ** 2)
    clv_512 = np.interp(np.clip(rho_c, 0.0, 1.0), bin_centers, prof).astype(np.float32)
    clv_bg = np.maximum(cv2.resize(clv_512, (w, h), interpolation=cv2.INTER_LINEAR), 8.0)

    # Squared radial coordinate rho2 -> disk_mask and heliocentric cosine mu
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    rho2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (R * R)
    disk_mask = rho2 <= (0.995 * 0.995)

    raw_f = gray_u8.astype(np.float32)
    norm_clv = np.where(disk_mask, raw_f / clv_bg, 0.0).astype(np.float32)

    # Local chromospheric absorption background B_sigma48 via 0.25x normalized convolution
    q_h, q_w = max(64, h // 4), max(64, w // 4)
    sig_q = 48.0 * (w / float(IMG_SIZE)) * 0.25
    f_q = cv2.resize(norm_clv, (q_w, q_h), interpolation=cv2.INTER_AREA)
    m_q = cv2.resize(disk_mask.astype(np.float32), (q_w, q_h), interpolation=cv2.INTER_AREA)
    num_q = cv2.GaussianBlur(f_q * m_q, (0, 0), sigmaX=sig_q, sigmaY=sig_q)
    den_q = cv2.GaussianBlur(m_q, (0, 0), sigmaX=sig_q, sigmaY=sig_q)
    b_q = np.where(den_q > 1e-4, num_q / np.maximum(den_q, 1e-4), 1.0).astype(np.float32)
    b_sigma48 = cv2.resize(b_q, (w, h), interpolation=cv2.INTER_LINEAR)

    contrast = np.where(
        disk_mask,
        (b_sigma48 - norm_clv) / np.maximum(b_sigma48, 1e-3),
        0.0,
    ).astype(np.float32)
    mu = np.where(
        disk_mask,
        np.sqrt(np.maximum(0.0, 1.0 - np.minimum(rho2, 1.0))),
        0.0,
    ).astype(np.float32)

    ch0 = np.clip((norm_clv - 0.30) / 1.10, 0.0, 1.0) * disk_mask
    ch1 = np.clip(contrast * 2.5 + 0.30, 0.0, 1.0) * disk_mask
    ch2 = mu.astype(np.float32)
    x_3ch = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

    return {
        "x_3ch": x_3ch,
        "disk_mask": disk_mask,
        "disk_params": (float(cx), float(cy), float(R)),
        "norm_clv": norm_clv,
        "contrast": contrast,
        "mu": mu,
    }


def disk_crop_box(
    cx: float, cy: float, R: float, h: int, w: int, pad_ratio: float = 1.015
) -> Tuple[int, int, int, int]:
    """Square bounding box tightly framing the solar disk inside image bounds."""
    rad = int(math.ceil(R * pad_ratio))
    x0 = max(0, int(round(cx - rad)))
    y0 = max(0, int(round(cy - rad)))
    x1 = min(w, int(round(cx + rad)))
    y1 = min(h, int(round(cy + rad)))
    if x1 <= x0 + 16 or y1 <= y0 + 16:
        return 0, 0, w, h
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Stage 0: MAGFiLO COCO Parsing, Multi-Annotator Targets & Date-Grouped Folds
# ---------------------------------------------------------------------------


def _walk_competition_tree(data_root: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Scan data root (or standard Kaggle input dirs) for MAGFiLO assets."""
    roots = [data_root] if data_root else ["/kaggle/input/filament-segmentation-2026", "/kaggle/input"]
    found_json: Optional[str] = None
    found_train: Optional[str] = None
    found_test: Optional[str] = None

    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            for fn in filenames:
                if fn.endswith(".json") and "magfilo" in fn.lower() and "train" in fn.lower():
                    found_json = os.path.join(dirpath, fn)
            if base == "train_images" and any(fn.lower().endswith((".jpeg", ".jpg", ".png")) for fn in filenames):
                found_train = dirpath
            if base == "test_images" and any(fn.lower().endswith((".jpeg", ".jpg", ".png")) for fn in filenames):
                found_test = dirpath
    return found_json, found_train, found_test


def discover_competition_paths(
    data_root: Optional[str] = None,
    train_json_arg: Optional[str] = None,
    train_dir_arg: Optional[str] = None,
    test_dir_arg: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Locate MAGFiLO train JSON, train_images dir, and test_images dir."""
    if train_json_arg and train_dir_arg and test_dir_arg:
        return train_json_arg, train_dir_arg, test_dir_arg

    found_json, found_train, found_test = _walk_competition_tree(data_root)
    if train_json_arg:
        found_json = train_json_arg
    if train_dir_arg:
        found_train = train_dir_arg
    if test_dir_arg:
        found_test = test_dir_arg

    if not (found_json and found_train and found_test):
        raise FileNotFoundError(
            f"Could not find MAGFiLO paths under data_root={data_root}: "
            f"json={found_json}, train={found_train}, test={found_test}"
        )
    return found_json, found_train, found_test


def discover_test_dir(data_root: Optional[str] = None, test_dir_arg: Optional[str] = None) -> str:
    """Locate test_images directory for predict-only inference."""
    if test_dir_arg and os.path.isdir(test_dir_arg):
        return test_dir_arg
    _, _, found_test = _walk_competition_tree(data_root)
    if found_test:
        return found_test
    if data_root and os.path.isdir(data_root) and any(
        fn.lower().endswith((".jpeg", ".jpg", ".png")) for fn in os.listdir(data_root)
    ):
        return data_root
    raise FileNotFoundError(f"Could not locate test_images directory (data_root={data_root}, test_dir={test_dir_arg})")


def resolve_disk_image_path(img_dir: str, file_name: str, stem_index: Dict[str, str]) -> str:
    """Resolve COCO image file_name robustly across .jpeg / .jpg / .png."""
    cand = os.path.join(img_dir, os.path.basename(file_name))
    if os.path.exists(cand):
        return cand
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if stem in stem_index:
        return stem_index[stem]
    raise FileNotFoundError(f"Cannot resolve image {file_name} inside {img_dir}")


def extract_obs_date(disk_stem: str) -> str:
    """Extract YYYYMMDD observation date from GONG filename stem YYYYMMDDHHMMSSII."""
    if len(disk_stem) >= 8 and disk_stem[:8].isdigit():
        return disk_stem[:8]
    digits = "".join(ch for ch in disk_stem if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return disk_stem


def rasterize_annotation_mask(seg: Any, h: int, w: int) -> np.ndarray:
    """Rasterize COCO polygon or RLE annotation into binary uint8 (H, W)."""
    if isinstance(seg, dict):
        return mask_util.decode(seg).astype(np.uint8)
    if isinstance(seg, list):
        valid_polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if not valid_polys:
            return np.zeros((h, w), dtype=np.uint8)
        rles = mask_util.frPyObjects(valid_polys, h, w)
        merged = mask_util.merge(rles)
        return mask_util.decode(merged).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def draw_spine_mask(ann: Dict[str, Any], inst_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Build centerline mask combining MAGFiLO explicit annotation["spine"]
    polyline and morphological medial-axis skeletonization of inst_mask.
    """
    spine_canvas = np.zeros((h, w), dtype=np.uint8)
    raw_spine = ann.get("spine", None)
    paths: List[List[float]] = []
    if isinstance(raw_spine, list) and len(raw_spine) > 0:
        if isinstance(raw_spine[0], (int, float)):
            paths = [raw_spine]
        elif isinstance(raw_spine[0], list):
            paths = [p for p in raw_spine if isinstance(p, list) and len(p) >= 4]

    for coords in paths:
        pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 2:
            ipts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(spine_canvas, [ipts], isClosed=False, color=1, thickness=3)
        elif pts.shape[0] == 1:
            px, py = int(round(pts[0, 0])), int(round(pts[0, 1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(spine_canvas, (px, py), 1, 1, -1)

    ys, xs = np.where(inst_mask > 0)
    if ys.size > 0:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = inst_mask[y0:y1, x0:x1] > 0
        skel = skeletonize(sub).astype(np.uint8)
        skel_dil = cv2.dilate(skel, np.ones((3, 3), dtype=np.uint8), iterations=1)
        spine_canvas[y0:y1, x0:x1] = np.maximum(spine_canvas[y0:y1, x0:x1], skel_dil)

    return spine_canvas


def compute_gap_corridor_mask(inst_masks: List[np.ndarray], h: int, w: int) -> np.ndarray:
    """
    Compute same-annotator inter-filament gap corridor target y_sep in {0, 1}.
    Marks 1-8 px background corridors sitting within radius 4 px of distinct
    instances plus 1 px direct touching interfaces.
    """
    n_inst = len(inst_masks)
    if n_inst < 2:
        return np.zeros((h, w), dtype=np.uint8)

    label_map = np.zeros((h, w), dtype=np.int32)
    touch_sep = np.zeros((h, w), dtype=np.uint8)
    for idx, m in enumerate(inst_masks, start=1):
        overlap = (label_map > 0) & (m > 0)
        if np.any(overlap):
            touch_sep[overlap] = 1
        label_map = np.where(m > 0, idx, label_map)

    union_mask = label_map > 0
    k_r4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lm_u16 = label_map.astype(np.uint16)
    max_id = cv2.dilate(lm_u16, k_r4).astype(np.int32)
    inv_map = np.where(union_mask, n_inst + 1 - label_map, 0).astype(np.uint16)
    dil_inv = cv2.dilate(inv_map, k_r4).astype(np.int32)
    min_id = np.where(dil_inv > 0, n_inst + 1 - dil_inv, 0)

    gap_corridor = (~union_mask) & (max_id > 0) & (min_id > 0) & (max_id != min_id)

    k_r1 = np.ones((3, 3), dtype=np.uint8)
    max_r1 = cv2.dilate(lm_u16, k_r1).astype(np.int32)
    dil_inv_r1 = cv2.dilate(inv_map, k_r1).astype(np.int32)
    min_r1 = np.where(dil_inv_r1 > 0, n_inst + 1 - dil_inv_r1, 0)
    touch_boundary = union_mask & (max_r1 > 0) & (min_r1 > 0) & (max_r1 != min_r1)

    return (gap_corridor | touch_boundary | (touch_sep > 0)).astype(np.uint8)


def assign_date_grouped_folds(
    disks: List[Dict[str, Any]], n_folds: int = 3
) -> List[Dict[str, Any]]:
    """
    Assign all unique physical disks to 3 observation-date-grouped folds (YYYYMMDD)
    with balanced counts and strictly ZERO date or disk leakage.
    """
    date_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in disks:
        date_groups[d["obs_date"]].append(d)

    sorted_dates = sorted(
        date_groups.keys(),
        key=lambda dt: (
            -sum(x["n_annotations"] for x in date_groups[dt]),
            -len(date_groups[dt]),
            dt,
        ),
    )

    fold_disk_counts = [0] * n_folds
    fold_ann_counts = [0] * n_folds
    for dt in sorted_dates:
        f_idx = min(range(n_folds), key=lambda f: (fold_disk_counts[f], fold_ann_counts[f], f))
        grp = date_groups[dt]
        for d in grp:
            d["fold"] = f_idx
        fold_disk_counts[f_idx] += len(grp)
        fold_ann_counts[f_idx] += sum(x["n_annotations"] for x in grp)

    dates_by_fold = [set(d["obs_date"] for d in disks if d["fold"] == f) for f in range(n_folds)]
    stems_by_fold = [set(d["disk_stem"] for d in disks if d["fold"] == f) for f in range(n_folds)]
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            assert dates_by_fold[i].isdisjoint(dates_by_fold[j]), "Date leakage between folds!"
            assert stems_by_fold[i].isdisjoint(stems_by_fold[j]), "Disk leakage between folds!"

    for f in range(n_folds):
        n_d = sum(1 for d in disks if d["fold"] == f)
        n_a = sum(d["n_annotations"] for d in disks if d["fold"] == f)
        log_msg(f"Fold {f}: {n_d} disks, {len(dates_by_fold[f])} unique dates, {n_a} annotator records")
    return disks


def build_magfilo_dataset_and_cache(
    train_json_path: str,
    train_img_dir: str,
    cache_dir: str,
    n_folds: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse MAGFiLO COCO JSON, group by physical disk & observation date, build
    multi-annotator consensus targets [y_fg, y_spine, y_sep], and cache compact PNGs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    log_msg(f"Parsing MAGFiLO JSON: {train_json_path}")
    with open(train_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    stem_index: Dict[str, str] = {}
    for fn in sorted(os.listdir(train_img_dir)):
        if fn.lower().endswith((".jpeg", ".jpg", ".png")):
            stem_index[os.path.splitext(fn)[0]] = os.path.join(train_img_dir, fn)

    anns_by_image_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image_id[str(ann["image_id"])].append(ann)

    records_by_stem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for img_rec in coco.get("images", []):
        fn = img_rec["file_name"]
        stem = os.path.splitext(os.path.basename(fn))[0]
        img_id_str = str(img_rec["id"])
        records_by_stem[stem].append(
            {
                "annotator_image": img_id_str,
                "file_name": fn,
                "width": int(img_rec.get("width", IMG_SIZE)),
                "height": int(img_rec.get("height", IMG_SIZE)),
                "annotations": anns_by_image_id.get(img_id_str, []),
            }
        )

    disks: List[Dict[str, Any]] = []
    gt_records: List[Dict[str, Any]] = []
    sorted_stems = sorted(records_by_stem.keys())
    log_msg(
        f"Found {len(coco.get('images', []))} annotator records across "
        f"{len(sorted_stems)} unique physical solar disks."
    )

    t0 = time.time()
    for idx, stem in enumerate(sorted_stems):
        recs = records_by_stem[stem]
        img_path = resolve_disk_image_path(train_img_dir, recs[0]["file_name"], stem_index)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = gray.shape[:2]

        cx, cy, R = estimate_solar_disk(gray)
        target_png_path = os.path.join(cache_dir, f"{stem}_targets.png")

        fg_sum = np.zeros((h, w), dtype=np.float32)
        spine_sum = np.zeros((h, w), dtype=np.float32)
        sep_max = np.zeros((h, w), dtype=np.uint8)
        n_ann = len(recs)
        total_inst = 0

        for rec in recs:
            inst_masks: List[np.ndarray] = []
            inst_rles: List[Dict[str, Any]] = []
            ann_spine_union = np.zeros((h, w), dtype=np.uint8)
            ann_fg_union = np.zeros((h, w), dtype=np.uint8)

            for ann in rec["annotations"]:
                m = rasterize_annotation_mask(ann["segmentation"], h, w)
                if m.sum() == 0:
                    continue
                inst_masks.append(m)
                counts = encode_rle_counts(m)
                inst_rles.append(rle_dict_from_counts(counts, h, w))
                ann_fg_union = np.maximum(ann_fg_union, m)
                sp = draw_spine_mask(ann, m, h, w)
                ann_spine_union = np.maximum(ann_spine_union, sp)

            total_inst += len(inst_masks)
            fg_sum += ann_fg_union.astype(np.float32)
            spine_sum += ann_spine_union.astype(np.float32)
            if len(inst_masks) >= 2:
                sep_a = compute_gap_corridor_mask(inst_masks, h, w)
                sep_max = np.maximum(sep_max, sep_a)

            gt_records.append(
                {
                    "annotator_image": rec["annotator_image"],
                    "disk_stem": stem,
                    "gt_rles": inst_rles,
                }
            )

        y_fg = fg_sum / float(max(n_ann, 1))
        y_spine = spine_sum / float(max(n_ann, 1))
        y_sep = sep_max.astype(np.float32)

        packed = np.stack(
            [
                np.clip(np.round(y_fg * 255.0), 0, 255).astype(np.uint8),
                np.clip(np.round(y_spine * 255.0), 0, 255).astype(np.uint8),
                np.clip(np.round(y_sep * 255.0), 0, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        cv2.imwrite(target_png_path, packed, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        disks.append(
            {
                "disk_stem": stem,
                "obs_date": extract_obs_date(stem),
                "img_path": img_path,
                "target_path": target_png_path,
                "height": h,
                "width": w,
                "disk_params": (float(cx), float(cy), float(R)),
                "n_annotations": n_ann,
                "n_instances": total_inst,
                "fold": -1,
            }
        )
        if (idx + 1) % 200 == 0 or (idx + 1) == len(sorted_stems):
            log_msg(f"  Target cache built: {idx + 1}/{len(sorted_stems)} disks ({time.time() - t0:.1f}s)")

    disks = assign_date_grouped_folds(disks, n_folds=n_folds)
    return disks, gt_records


# ---------------------------------------------------------------------------
# Stage N: 3-Head MagfiloPanopticUNet + Boundary-Upweighted Loss + soft_clDice
# ---------------------------------------------------------------------------


class SCSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation (Roy et al., MICCAI 2018)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.c_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.c_se(x) + x * self.s_se(x)


class PPMContext(nn.Module):
    """Lightweight Multi-Scale Atrous + Global Context Bottleneck."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        b_ch = max(out_ch // 4, 16)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, b_ch, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(b_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), p], dim=1)
        return self.proj(out)


class DecoderBlock(nn.Module):
    """U-Net skip decoder stage with 1x1 compression, depthwise-separable 3x3 refinement, and scSE."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dw_sep: bool = False) -> None:
        super().__init__()
        if dw_sep:
            refine: List[nn.Module] = [
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 1, bias=False),
            ]
        else:
            refine = [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)]
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            *refine,
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.scse = SCSEBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.scse(self.conv(x))


class TorchvisionResNet18Features(nn.Module):
    """Pure torchvision ResNet18 multi-scale feature extractor fallback."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained and HAS_TORCHVISION:
            try:
                weights = tv_models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        net = tv_models.resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c1, c2, c3, c4, c5]


class MagfiloPanopticUNet(nn.Module):
    """
    Lightweight 3-head (fg, spine, sep) Panoptic U-Net built from PyTorch + timm
    (with torchvision fallback). Uses a compact stride-1 detail stem to retain
    1-3 px filament barbs and 1-8 px inter-filament separation corridors.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        bottleneck_ch: int = 160,
        dec_widths: Sequence[int] = (96, 64, 40, 24),
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, enc_channels = self._build_encoder(backbone_name, pretrained)

        self.full_stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
        )

        self.ppm = PPMContext(enc_channels[-1], bottleneck_ch)

        n_skips = len(enc_channels) - 1
        widths = list(dec_widths[:n_skips])
        while len(widths) < n_skips:
            widths.append(max(widths[-1] // 2, 24))

        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for idx in range(n_skips):
            skip_ch = enc_channels[-2 - idx]
            out_ch = widths[idx]
            self.dec_blocks.append(DecoderBlock(prev_ch, skip_ch, out_ch, dw_sep=(idx >= 2)))
            prev_ch = out_ch

        head_in = prev_ch + 8
        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_in, 3, padding=1, groups=head_in, bias=False),
            nn.BatchNorm2d(head_in),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_in, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 1),
        )

    @staticmethod
    def _build_encoder(backbone_name: str, pretrained: bool) -> Tuple[nn.Module, List[int]]:
        if HAS_TIMM:
            for try_pretrained in ([True, False] if pretrained else [False]):
                try:
                    enc = timm.create_model(
                        backbone_name,
                        pretrained=try_pretrained,
                        features_only=True,
                        in_chans=3,
                    )
                    return enc, list(enc.feature_info.channels())
                except Exception as exc:
                    log_msg(
                        f"Warning: timm.create_model({backbone_name}, pretrained={try_pretrained}) failed: {exc}"
                    )
        if HAS_TORCHVISION:
            for try_pretrained in ([True, False] if pretrained else [False]):
                try:
                    enc = TorchvisionResNet18Features(pretrained=try_pretrained)
                    return enc, enc.channels
                except Exception:
                    pass
        raise RuntimeError("Neither timm nor torchvision encoder could be initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s_full = self.full_stem(x)
        feats = self.encoder(x)
        d = self.ppm(feats[-1])
        for idx, blk in enumerate(self.dec_blocks):
            d = blk(d, feats[-2 - idx])
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([d, s_full], dim=1)
        return self.head(out)


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 5) -> torch.Tensor:
    """Differentiable soft skeletonization (Shit et al., CVPR 2021)."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def masked_soft_dice_loss(
    prob: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    p = prob.float() * valid_mask
    t = target.float() * valid_mask
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1.0) / (denom + 1.0 + eps)
    return (1.0 - dice).mean()


class SolarMultiTaskLoss(nn.Module):
    """
    Boundary-upweighted BCE + SoftDice (3x weight near y_sep) + spine/sep losses
    + 0.25 * soft_clDice topology preservation (Shit et al., CVPR 2021).
    """

    def __init__(self, cl_iters: int = 5) -> None:
        super().__init__()
        self.cl_iters = cl_iters

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        fg_logit = logits[:, 0:1].float()
        sp_logit = logits[:, 1:2].float()
        sep_logit = logits[:, 2:3].float()

        y_fg = targets[:, 0:1].float()
        y_sp = targets[:, 1:2].float()
        y_sep = targets[:, 2:3].float()
        valid = valid_mask.float()

        fg_prob = torch.sigmoid(fg_logit)
        sp_prob = torch.sigmoid(sp_logit)
        sep_prob = torch.sigmoid(sep_logit)

        sep_zone = F.max_pool2d(y_sep, kernel_size=5, stride=1, padding=2)
        w_fg = valid * (1.0 + 2.0 * sep_zone)
        bce_fg = (
            F.binary_cross_entropy_with_logits(fg_logit, y_fg, reduction="none") * w_fg
        ).sum() / w_fg.sum().clamp_min(1.0)
        dice_fg = masked_soft_dice_loss(fg_prob, y_fg, valid)

        bce_sp = (
            F.binary_cross_entropy_with_logits(sp_logit, y_sp, reduction="none") * valid
        ).sum() / valid.sum().clamp_min(1.0)
        dice_sp = masked_soft_dice_loss(sp_prob, y_sp, valid)

        sep_pos_w = torch.tensor([4.0], device=logits.device, dtype=torch.float32)
        bce_sep = (
            F.binary_cross_entropy_with_logits(sep_logit, y_sep, pos_weight=sep_pos_w, reduction="none")
            * valid
        ).sum() / valid.sum().clamp_min(1.0)
        dice_sep = masked_soft_dice_loss(sep_prob, y_sep, valid)

        fg_masked = fg_prob * valid
        y_fg_masked = y_fg * valid
        skel_pred = soft_skel(fg_masked, iter_=self.cl_iters)
        skel_gt = torch.maximum(y_sp * valid, soft_skel(y_fg_masked, iter_=self.cl_iters))
        tprec = (skel_pred * y_fg_masked).sum() / skel_pred.sum().clamp_min(1e-4)
        tsens = (skel_gt * fg_masked).sum() / skel_gt.sum().clamp_min(1e-4)
        cl_fg = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + 1e-4)

        sp_masked = sp_prob * valid
        tprec_sp = (sp_masked * fg_masked).sum() / sp_masked.sum().clamp_min(1e-4)
        tsens_sp = (skel_pred * sp_masked).sum() / skel_pred.sum().clamp_min(1e-4)
        cl_fg_sp = 1.0 - (2.0 * tprec_sp * tsens_sp) / (tprec_sp + tsens_sp + 1e-4)

        return (
            (bce_fg + dice_fg)
            + 0.60 * (bce_sp + dice_sp)
            + 0.50 * (0.5 * bce_sep + 0.5 * dice_sep)
            + 0.25 * (0.75 * cl_fg + 0.25 * cl_fg_sp)
        )


# ---------------------------------------------------------------------------
# Datasets for GPU (1536x1536 crop) and CPU (1024x1024-matched 384x384 patches)
# ---------------------------------------------------------------------------


def _apply_geometric_aug(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact rotation/flip equivariant augmentation for solar H-alpha imagery."""
    k = int(rng.integers(0, 4))
    if k > 0:
        x = np.rot90(x, k, axes=(1, 2)).copy()
        y = np.rot90(y, k, axes=(1, 2)).copy()
        valid = np.rot90(valid, k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        x = x[:, :, ::-1].copy()
        y = y[:, :, ::-1].copy()
        valid = valid[:, :, ::-1].copy()
    if rng.random() < 0.5:
        x = x[:, ::-1, :].copy()
        y = y[:, ::-1, :].copy()
        valid = valid[:, ::-1, :].copy()
    if rng.random() < 0.7:
        g0 = float(rng.uniform(0.92, 1.08))
        b0 = float(rng.uniform(-0.04, 0.04))
        g1 = float(rng.uniform(0.90, 1.10))
        x[0] = np.clip(x[0] * g0 + b0, 0.0, 1.0) * valid[0]
        x[1] = np.clip((x[1] - 0.3) * g1 + 0.3, 0.0, 1.0) * valid[0]
    return x, y, valid


class GpuFullDiskDataset(Dataset):
    """Dataset returning tight solar-disk crops resized to gpu_img_size (1536x1536)."""

    def __init__(
        self, disks: List[Dict[str, Any]], target_size: int = 1536, train: bool = True, seed: int = 42
    ) -> None:
        self.disks = disks
        self.target_size = target_size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.disks)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.disks[index]
        gray = cv2.imread(d["img_path"], cv2.IMREAD_GRAYSCALE)
        prep = preprocess_solar_disk(gray, disk_params=d["disk_params"])
        packed = cv2.imread(d["target_path"], cv2.IMREAD_UNCHANGED)
        y_raw = packed.astype(np.float32) / 255.0

        h, w = gray.shape[:2]
        cx, cy, R = d["disk_params"]
        x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

        x_crop = prep["x_3ch"][:, y0:y1, x0:x1].transpose(1, 2, 0)
        m_crop = prep["disk_mask"][y0:y1, x0:x1].astype(np.float32)
        y_crop = y_raw[y0:y1, x0:x1, :]

        sz = self.target_size
        x_res = cv2.resize(x_crop, (sz, sz), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        m_res = (cv2.resize(m_crop, (sz, sz), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.float32)[
            None, :, :
        ]
        fg_res = cv2.resize(y_crop[:, :, 0], (sz, sz), interpolation=cv2.INTER_AREA)
        sp_res = np.clip(cv2.resize(y_crop[:, :, 1], (sz, sz), interpolation=cv2.INTER_AREA) * 1.15, 0.0, 1.0)
        sep_res = (cv2.resize(y_crop[:, :, 2], (sz, sz), interpolation=cv2.INTER_AREA) > 0.12).astype(
            np.float32
        )
        y_res = np.stack([fg_res, sp_res, sep_res], axis=0).astype(np.float32)

        if self.train:
            rng = np.random.default_rng(self.seed + index + int(time.time() * 1000) % 100000)
            x_res, y_res, m_res = _apply_geometric_aug(x_res, y_res, m_res, rng)

        return torch.from_numpy(x_res), torch.from_numpy(y_res), torch.from_numpy(m_res)


def sample_cpu_patches_from_disks(
    disks: List[Dict[str, Any]],
    total_patches: int,
    patch_size: int = 384,
    disk_scale_size: int = 1024,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample 384x384 patches from tight solar-disk crops resampled to disk_scale_size
    (matching 1024x1024 CPU single-pass inference scale 1:1) with a
    70% spine/filament, 20% hard-negative plage/sunspot, 10% quiet-disk mix.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_disks = len(disks)
    patches_per_disk = max(3, int(math.ceil(total_patches / float(min(max(n_disks, 1), 36)))))
    order = rng.permutation(n_disks)

    xs_list: List[np.ndarray] = []
    ys_list: List[np.ndarray] = []
    ms_list: List[np.ndarray] = []
    half = patch_size // 2
    sz = disk_scale_size

    for idx in order:
        if len(xs_list) >= total_patches:
            break
        d = disks[int(idx)]
        gray = cv2.imread(d["img_path"], cv2.IMREAD_GRAYSCALE)
        prep = preprocess_solar_disk(gray, disk_params=d["disk_params"])
        packed = cv2.imread(d["target_path"], cv2.IMREAD_UNCHANGED)
        y_raw = packed.astype(np.float32) / 255.0

        h, w = gray.shape[:2]
        cx, cy, R = d["disk_params"]
        x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

        x_crop = prep["x_3ch"][:, y0:y1, x0:x1].transpose(1, 2, 0)
        m_crop = prep["disk_mask"][y0:y1, x0:x1].astype(np.float32)
        y_crop = y_raw[y0:y1, x0:x1, :]
        con_crop = prep["contrast"][y0:y1, x0:x1]
        clv_crop = prep["norm_clv"][y0:y1, x0:x1]

        x_res = cv2.resize(x_crop, (sz, sz), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        m_res = cv2.resize(m_crop, (sz, sz), interpolation=cv2.INTER_NEAREST) > 0.5
        fg_res = cv2.resize(y_crop[:, :, 0], (sz, sz), interpolation=cv2.INTER_AREA)
        sp_res = np.clip(cv2.resize(y_crop[:, :, 1], (sz, sz), interpolation=cv2.INTER_AREA) * 1.25, 0.0, 1.0)
        sep_res = (cv2.resize(y_crop[:, :, 2], (sz, sz), interpolation=cv2.INTER_AREA) > 0.10).astype(
            np.float32
        )
        con_res = cv2.resize(con_crop, (sz, sz), interpolation=cv2.INTER_AREA)
        clv_res = cv2.resize(clv_crop, (sz, sz), interpolation=cv2.INTER_AREA)

        ds_sz = sz // 4
        ds_sp = cv2.resize((sp_res > 0.25).astype(np.uint8), (ds_sz, ds_sz), interpolation=cv2.INTER_NEAREST)
        ds_fg = cv2.resize((fg_res > 0.25).astype(np.uint8), (ds_sz, ds_sz), interpolation=cv2.INTER_NEAREST)
        ds_dm = cv2.resize(m_res.astype(np.uint8), (ds_sz, ds_sz), interpolation=cv2.INTER_NEAREST) > 0
        ds_con = cv2.resize(con_res, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA)
        ds_clv = cv2.resize(clv_res, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA)

        pos_pts = np.argwhere(ds_sp > 0)
        if pos_pts.shape[0] == 0:
            pos_pts = np.argwhere(ds_fg > 0)
        hard_pts = np.argwhere(
            ds_dm & (ds_fg == 0) & ((np.abs(ds_con) > 0.08) | (ds_clv < 0.78))
        )
        quiet_pts = np.argwhere(ds_dm & (ds_fg == 0))

        for _ in range(patches_per_disk):
            if len(xs_list) >= total_patches:
                break
            u = rng.random()
            if u < 0.70 and pos_pts.shape[0] > 0:
                pt = pos_pts[int(rng.integers(0, pos_pts.shape[0]))]
                cy_c = int(pt[0]) * 4 + int(rng.integers(-48, 49))
                cx_c = int(pt[1]) * 4 + int(rng.integers(-48, 49))
            elif u < 0.90 and hard_pts.shape[0] > 0:
                pt = hard_pts[int(rng.integers(0, hard_pts.shape[0]))]
                cy_c = int(pt[0]) * 4 + int(rng.integers(-24, 25))
                cx_c = int(pt[1]) * 4 + int(rng.integers(-24, 25))
            elif quiet_pts.shape[0] > 0:
                pt = quiet_pts[int(rng.integers(0, quiet_pts.shape[0]))]
                cy_c = int(pt[0]) * 4
                cx_c = int(pt[1]) * 4
            else:
                cy_c = sz // 2
                cx_c = sz // 2

            py0 = int(np.clip(cy_c - half, 0, max(0, sz - patch_size)))
            px0 = int(np.clip(cx_c - half, 0, max(0, sz - patch_size)))
            py1 = py0 + patch_size
            px1 = px0 + patch_size

            xp = x_res[:, py0:py1, px0:px1].copy()
            yp = np.stack(
                [fg_res[py0:py1, px0:px1], sp_res[py0:py1, px0:px1], sep_res[py0:py1, px0:px1]], axis=0
            ).astype(np.float32)
            mp = m_res[py0:py1, px0:px1].astype(np.float32)[None, :, :]
            xp, yp, mp = _apply_geometric_aug(xp, yp, mp, rng)

            xs_list.append(xp)
            ys_list.append(yp)
            ms_list.append(mp)

    return np.stack(xs_list, axis=0), np.stack(ys_list, axis=0), np.stack(ms_list, axis=0)


# ---------------------------------------------------------------------------
# Stage N Inference (GPU 1536 + 4-View TTA; CPU 1024 Single-Pass channels_last)
# ---------------------------------------------------------------------------


def infer_disk_gpu(
    models: List[nn.Module],
    prep: Dict[str, Any],
    device: torch.device,
    img_size: int = 1536,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer 3-head probabilities on GPU using tight disk crop + 4-view flip TTA."""
    x_3ch = prep["x_3ch"]
    dmask = prep["disk_mask"]
    h, w = dmask.shape
    cx, cy, R = prep["disk_params"]
    x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

    crop = x_3ch[:, y0:y1, x0:x1].transpose(1, 2, 0)
    crop_res = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    xt = torch.from_numpy(crop_res).unsqueeze(0).to(device)

    prob_acc = np.zeros((3, img_size, img_size), dtype=np.float32)
    n_views = 4 if use_tta else 1

    with torch.inference_mode():
        for model in models:
            model.eval()
            for v in range(n_views):
                xv = xt
                if v == 1:
                    xv = torch.flip(xt, dims=[3])
                elif v == 2:
                    xv = torch.flip(xt, dims=[2])
                elif v == 3:
                    xv = torch.flip(xt, dims=[2, 3])

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pv = torch.sigmoid(model(xv).float())

                if v == 1:
                    pv = torch.flip(pv, dims=[3])
                elif v == 2:
                    pv = torch.flip(pv, dims=[2])
                elif v == 3:
                    pv = torch.flip(pv, dims=[2, 3])
                prob_acc += pv.squeeze(0).cpu().numpy()

    prob_acc /= float(len(models) * n_views)
    ch, cw = y1 - y0, x1 - x0
    out_full = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        resized = cv2.resize(prob_acc[c], (cw, ch), interpolation=cv2.INTER_LINEAR)
        out_full[c, y0:y1, x0:x1] = resized * dmask[y0:y1, x0:x1]
    return out_full[0], out_full[1], out_full[2]


def infer_disk_cpu_tight(
    models: List[nn.Module],
    prep: Dict[str, Any],
    device: torch.device,
    infer_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-pass CPU inference on tight 1024x1024 solar-disk crop using
    torch.inference_mode() and channels_last layout, upsampled bilinearly back
    to native 2048x2048 coordinates inside the solar limb.
    """
    x_3ch = prep["x_3ch"]
    dmask = prep["disk_mask"]
    h, w = dmask.shape
    cx, cy, R = prep["disk_params"]
    x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

    crop = x_3ch[:, y0:y1, x0:x1].transpose(1, 2, 0)
    crop_res = cv2.resize(crop, (infer_size, infer_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    xt = torch.from_numpy(np.ascontiguousarray(crop_res)).unsqueeze(0).to(device)
    xt = xt.to(memory_format=torch.channels_last)

    prob_acc = np.zeros((3, infer_size, infer_size), dtype=np.float32)
    with torch.inference_mode():
        for model in models:
            model.eval()
            pv = torch.sigmoid(model(xt).float())
            prob_acc += pv.squeeze(0).cpu().numpy()

    prob_acc /= float(max(len(models), 1))
    ch, cw = y1 - y0, x1 - x0
    out_full = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        resized = cv2.resize(prob_acc[c], (cw, ch), interpolation=cv2.INTER_LINEAR)
        out_full[c, y0:y1, x0:x1] = resized * dmask[y0:y1, x0:x1]
    return out_full[0], out_full[1], out_full[2]


# ---------------------------------------------------------------------------
# Stage H: Component Tree, Physics Features, Bridging, LGBM, Tree DP
# ---------------------------------------------------------------------------


def _compute_candidate_features(
    comp_mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    level_thresh: float,
    n_children: int,
    p_fg: np.ndarray,
    p_spine: np.ndarray,
    p_sep: np.ndarray,
    contrast: np.ndarray,
    norm_clv: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Compute morphological, topological, and solar-physics features on tight bbox."""
    h, w = comp_mask.shape
    pad = 4
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad)
    y1 = min(h, by + bh + pad)

    sub_m = comp_mask[y0:y1, x0:x1]
    area = float(sub_m.sum())
    area_safe = max(area, 1.0)

    contours, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    perim_safe = max(perim, 1.0)

    hull_area = area_safe
    if contours:
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        hull_area = max(float(cv2.contourArea(hull)), area_safe)

    circularity = (4.0 * math.pi * area_safe) / (perim_safe * perim_safe)
    solidity = area_safe / hull_area
    bbox_fill = area_safe / max(float(bw * bh), 1.0)
    skel_len_ratio = perim_safe / (2.0 * math.sqrt(area_safe))
    mean_width = (2.0 * area_safe) / perim_safe

    M = cv2.moments(sub_m)
    m00 = max(float(M["m00"]), 1.0)
    mu20 = float(M["mu20"]) / m00
    mu02 = float(M["mu02"]) / m00
    mu11 = float(M["mu11"]) / m00
    diff = mu20 - mu02
    common = math.sqrt(max(0.0, diff * diff + 4.0 * mu11 * mu11))
    lam1 = max(0.5 * (mu20 + mu02 + common), 1e-6)
    lam2 = max(0.5 * (mu20 + mu02 - common), 1e-6)
    eccentricity = math.sqrt(max(0.0, 1.0 - lam2 / lam1))
    elongation = math.sqrt(lam1 / lam2)

    fg_vals = p_fg[y0:y1, x0:x1][sub_m > 0]
    sp_vals = p_spine[y0:y1, x0:x1][sub_m > 0]
    sep_vals = p_sep[y0:y1, x0:x1][sub_m > 0]
    con_vals = contrast[y0:y1, x0:x1][sub_m > 0]
    clv_vals = norm_clv[y0:y1, x0:x1][sub_m > 0]
    mu_vals = mu[y0:y1, x0:x1][sub_m > 0]

    dil = cv2.dilate(sub_m, np.ones((7, 7), dtype=np.uint8), iterations=1)
    ring = (dil > 0) & (sub_m == 0)
    ring_sep = p_sep[y0:y1, x0:x1][ring]
    boundary_p_sep = float(ring_sep.mean()) if ring_sep.size > 0 else 0.0

    umbral_dark_frac = float((clv_vals < 0.45).mean()) if clv_vals.size > 0 else 0.0
    sunspot_score = umbral_dark_frac * circularity * solidity

    return np.array(
        [
            math.log1p(area_safe),
            area_safe,
            perim_safe,
            float(bw),
            float(bh),
            bbox_fill,
            circularity,
            eccentricity,
            elongation,
            solidity,
            skel_len_ratio,
            mean_width,
            float(level_thresh),
            float(n_children),
            float(mu_vals.mean()),
            float(mu_vals.min()),
            float(fg_vals.mean()),
            float(np.percentile(fg_vals, 90)),
            float(np.percentile(fg_vals, 10)),
            float(sp_vals.mean()),
            float(np.percentile(sp_vals, 90)),
            float((sp_vals > 0.25).mean()),
            float(sep_vals.mean()),
            float(np.percentile(sep_vals, 90)),
            boundary_p_sep,
            float(con_vals.mean()),
            float(np.percentile(con_vals, 90)),
            float(clv_vals.mean()),
            float(clv_vals.min()),
            umbral_dark_frac,
            sunspot_score,
        ],
        dtype=np.float32,
    )


def _is_astrophysical_sunspot(feat: np.ndarray) -> bool:
    """Astrophysical morphological gate rejecting compact umbral sunspot cores."""
    circ = float(feat[6])
    elong = float(feat[8])
    solidity = float(feat[9])
    mean_sp = float(feat[19])
    umbral = float(feat[29])
    return umbral > 0.28 and circ > 0.62 and solidity > 0.85 and elong < 1.8 and mean_sp < 0.22


def _local_endpoint_tangent(pts_yx: np.ndarray, anchor_yx: np.ndarray) -> np.ndarray:
    """Estimate unit tangent vector pointing along the filament near anchor_yx."""
    d2 = np.sum((pts_yx - anchor_yx[None, :]) ** 2, axis=1)
    near = pts_yx[d2 <= 18.0 * 18.0]
    if near.shape[0] < 3:
        near = pts_yx[np.argsort(d2)[: min(8, pts_yx.shape[0])]]
    centered = near - near.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, -1]
    return tangent / max(float(np.linalg.norm(tangent)), 1e-6)


def _find_spine_bridge_candidates(
    base_nodes: List[Dict[str, Any]],
    p_fg: np.ndarray,
    p_spine: np.ndarray,
    p_sep: np.ndarray,
    contrast: np.ndarray,
    norm_clv: np.ndarray,
    mu: np.ndarray,
    disk_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Find pairs of filament fragments (gap <= 14 px, low P_sep, dark H-alpha /
    spine bridge, collinear spine-tangent vectors) and construct merged RLE candidates.
    """
    if len(base_nodes) < 2:
        return []
    h, w = disk_mask.shape
    n = len(base_nodes)

    pts_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    for node in base_nodes:
        m = decode_rle_counts(node["rle_counts"], h, w)
        masks_list.append(m)
        bx, by, bw, bh = node["bbox"]
        sub = m[by : by + bh, bx : bx + bw]
        er = cv2.erode(sub, np.ones((3, 3), dtype=np.uint8), iterations=1)
        b_yx = np.argwhere((sub > 0) & (er == 0))
        if b_yx.shape[0] == 0:
            b_yx = np.argwhere(sub > 0)
        b_yx[:, 0] += by
        b_yx[:, 1] += bx
        if b_yx.shape[0] > 400:
            step = max(1, b_yx.shape[0] // 400)
            b_yx = b_yx[::step]
        pts_list.append(b_yx.astype(np.float32))

    bridge_candidates: List[Dict[str, Any]] = []
    for i in range(n):
        ax0, ay0, aw, ah = base_nodes[i]["bbox"]
        ax1, ay1 = ax0 + aw, ay0 + ah
        tree_i = cKDTree(pts_list[i])
        for j in range(i + 1, n):
            bx0, by0, bw, bh = base_nodes[j]["bbox"]
            bx1, by1 = bx0 + bw, by0 + bh
            dx = max(0, max(ax0 - bx1, bx0 - ax1))
            dy = max(0, max(ay0 - by1, by0 - ay1))
            if dx * dx + dy * dy > BRIDGE_MAX_GAP_PX * BRIDGE_MAX_GAP_PX:
                continue

            dists, idxs = tree_i.query(pts_list[j], k=1, distance_upper_bound=BRIDGE_MAX_GAP_PX)
            best_j_idx = int(np.argmin(dists))
            best_dist = float(dists[best_j_idx])
            if not np.isfinite(best_dist) or best_dist > BRIDGE_MAX_GAP_PX or best_dist < 1.0:
                continue
            best_i_idx = int(idxs[best_j_idx])
            p_a = pts_list[i][best_i_idx]
            p_b = pts_list[j][best_j_idx]

            n_steps = max(4, int(math.ceil(best_dist * 2.0)))
            ts = np.linspace(0.0, 1.0, n_steps)[:, None]
            line_yx = np.round((1.0 - ts) * p_a[None, :] + ts * p_b[None, :]).astype(int)
            ly = np.clip(line_yx[:, 0], 0, h - 1)
            lx = np.clip(line_yx[:, 1], 0, w - 1)

            mean_sep_gap = float(p_sep[ly, lx].mean())
            mean_con_gap = float(contrast[ly, lx].mean())
            mean_sp_gap = float(p_spine[ly, lx].mean())
            if mean_sep_gap > BRIDGE_MAX_P_SEP:
                continue
            if mean_con_gap < 0.03 and mean_sp_gap < 0.14:
                continue

            vec = p_b - p_a
            vec_unit = vec / max(float(np.linalg.norm(vec)), 1e-6)
            tan_a = _local_endpoint_tangent(pts_list[i], p_a)
            tan_b = _local_endpoint_tangent(pts_list[j], p_b)
            if abs(float(np.dot(tan_a, vec_unit))) < BRIDGE_MIN_DOT:
                continue
            if abs(float(np.dot(tan_b, vec_unit))) < BRIDGE_MIN_DOT:
                continue

            merged = np.maximum(masks_list[i], masks_list[j]).copy()
            cv2.line(
                merged,
                (int(round(p_a[1])), int(round(p_a[0]))),
                (int(round(p_b[1])), int(round(p_b[0]))),
                color=1,
                thickness=3,
            )
            merged &= disk_mask.astype(np.uint8)
            ys_m, xs_m = np.where(merged > 0)
            if ys_m.size == 0:
                continue
            mbx, mby = int(xs_m.min()), int(ys_m.min())
            mbw, mbh = int(xs_m.max() - mbx + 1), int(ys_m.max() - mby + 1)
            feat = _compute_candidate_features(
                merged,
                mbx,
                mby,
                mbw,
                mbh,
                min(base_nodes[i]["level"], base_nodes[j]["level"]),
                0,
                p_fg,
                p_spine,
                p_sep,
                contrast,
                norm_clv,
                mu,
            )
            if _is_astrophysical_sunspot(feat):
                continue
            bridge_candidates.append(
                {
                    "node_a": i,
                    "node_b": j,
                    "rle_counts": encode_rle_counts(merged),
                    "bbox": (mbx, mby, mbw, mbh),
                    "area": int(ys_m.size),
                    "features": feat,
                    "gap_px": best_dist,
                }
            )
    return bridge_candidates


def extract_disk_component_tree(
    disk_stem: str,
    p_fg: np.ndarray,
    p_spine: np.ndarray,
    p_sep: np.ndarray,
    prep: Dict[str, Any],
    levels: Sequence[float] = TREE_LEVELS,
    min_area: int = MIN_COMPONENT_AREA,
    gt_annotators: Optional[List[List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Build laminar component tree over superlevels of P_eff, extract physics
    features, compute multi-annotator expected PQ u_gt(c), and collect spine
    corridor bridge options.
    """
    dmask = prep["disk_mask"]
    contrast = prep["contrast"]
    norm_clv = prep["norm_clv"]
    mu = prep["mu"]
    h, w = dmask.shape

    p_eff = np.clip(0.75 * p_fg + 0.25 * p_spine - 0.35 * p_sep, 0.0, 1.0) * dmask

    nodes: List[Dict[str, Any]] = []
    prev_labels: Optional[np.ndarray] = None
    prev_id_to_node: Dict[int, int] = {}
    nodes_at_036: List[int] = []

    def _run_levels(lvl_seq: Sequence[float], area_thr: int) -> None:
        nonlocal prev_labels, prev_id_to_node
        for lvl in lvl_seq:
            bw_mask = ((p_eff >= lvl) & dmask).astype(np.uint8)
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bw_mask, connectivity=8)
            curr_id_to_node: Dict[int, int] = {}

            for cid in range(1, n_lbl):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < area_thr:
                    continue
                bx = int(stats[cid, cv2.CC_STAT_LEFT])
                by = int(stats[cid, cv2.CC_STAT_TOP])
                bw = int(stats[cid, cv2.CC_STAT_WIDTH])
                bh = int(stats[cid, cv2.CC_STAT_HEIGHT])

                parent_idx = -1
                if prev_labels is not None:
                    sub_c = labels[by : by + bh, bx : bx + bw] == cid
                    sub_p = prev_labels[by : by + bh, bx : bx + bw]
                    vals = sub_p[sub_c]
                    if vals.size > 0:
                        pid = int(vals[0])
                        if pid in prev_id_to_node:
                            cand_parent = prev_id_to_node[pid]
                            if nodes[cand_parent]["area"] == area:
                                curr_id_to_node[cid] = cand_parent
                                continue
                            parent_idx = cand_parent

                comp_mask = (labels == cid).astype(np.uint8)
                node_idx = len(nodes)
                curr_id_to_node[cid] = node_idx
                nodes.append(
                    {
                        "node_idx": node_idx,
                        "level": float(lvl),
                        "area": area,
                        "bbox": (bx, by, bw, bh),
                        "parent": parent_idx,
                        "children": [],
                        "rle_counts": encode_rle_counts(comp_mask),
                        "is_sunspot": False,
                    }
                )
                if parent_idx >= 0:
                    nodes[parent_idx]["children"].append(node_idx)

            if abs(float(lvl) - 0.36) < 1e-4:
                nodes_at_036.extend(sorted(set(curr_id_to_node.values())))

            prev_labels = labels
            prev_id_to_node = curr_id_to_node

    _run_levels(levels, min_area)

    if len(nodes) == 0:
        prev_labels = None
        prev_id_to_node = {}
        _run_levels(FALLBACK_LEVELS, FALLBACK_MIN_AREA)
    if len(nodes) == 0:
        masked_eff = np.where(dmask, p_eff, -1.0)
        py, px = np.unravel_index(int(np.argmax(masked_eff)), masked_eff.shape)
        fb = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(fb, (int(px), int(py)), 8, 1, -1)
        fb &= dmask.astype(np.uint8)
        ys_f, xs_f = np.where(fb > 0)
        if ys_f.size == 0:
            fb[h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 1
            ys_f, xs_f = np.where(fb > 0)
        bx, by = int(xs_f.min()), int(ys_f.min())
        bw, bh = int(xs_f.max() - bx + 1), int(ys_f.max() - by + 1)
        nodes.append(
            {
                "node_idx": 0,
                "level": 0.01,
                "area": int(ys_f.size),
                "bbox": (bx, by, bw, bh),
                "parent": -1,
                "children": [],
                "rle_counts": encode_rle_counts(fb),
                "is_sunspot": False,
            }
        )

    feats_list: List[np.ndarray] = []
    for node in nodes:
        cm = decode_rle_counts(node["rle_counts"], h, w)
        bx, by, bw, bh = node["bbox"]
        feat = _compute_candidate_features(
            cm, bx, by, bw, bh, node["level"], len(node["children"]), p_fg, p_spine, p_sep, contrast, norm_clv, mu
        )
        node["is_sunspot"] = _is_astrophysical_sunspot(feat)
        feats_list.append(feat)

    feat_mat = np.stack(feats_list, axis=0)

    u_gt = np.zeros(len(nodes), dtype=np.float32)
    if gt_annotators is not None and len(gt_annotators) > 0:
        cand_rles = [rle_dict_from_counts(nd["rle_counts"], h, w) for nd in nodes]
        u_acc = np.zeros(len(nodes), dtype=np.float64)
        for ann_gt_rles in gt_annotators:
            if not ann_gt_rles:
                continue
            iou_m = mask_util.iou(cand_rles, ann_gt_rles, [0] * len(ann_gt_rles))
            best_iou = iou_m.max(axis=1)
            u_acc += np.where(best_iou > 0.5, best_iou, 0.0)
        u_gt = (u_acc / float(len(gt_annotators))).astype(np.float32)

    lvl_036 = [nodes[idx] for idx in nodes_at_036 if not nodes[idx]["is_sunspot"]]
    if len(lvl_036) < 2:
        lvl_036 = [nd for nd in nodes if nd["parent"] == -1 and not nd["is_sunspot"]]
    bridge_options = _find_spine_bridge_candidates(
        lvl_036, p_fg, p_spine, p_sep, contrast, norm_clv, mu, dmask
    )

    return {
        "disk_stem": disk_stem,
        "nodes": nodes,
        "features": feat_mat,
        "u_gt": u_gt,
        "base_bridge_nodes": [nd["node_idx"] for nd in lvl_036],
        "bridges": bridge_options,
    }


def solve_tree_dp_antichain(
    nodes: List[Dict[str, Any]],
    u_hat: np.ndarray,
    tau: float,
    allow_empty: bool = False,
) -> List[int]:
    """
    Exact bottom-up tree DP selecting a pairwise-disjoint antichain of
    candidates maximizing sum_{c in S} (u_hat(c) - tau).

    When allow_empty=False and every non-sunspot node falls below tau, falls back
    to the single highest-u_hat valid node so every test disk has >= 1 mask.
    """
    n = len(nodes)
    if n == 0:
        return []

    eff_u = np.array(
        [-1.0 if nodes[i].get("is_sunspot", False) else float(u_hat[i]) for i in range(n)],
        dtype=np.float64,
    )

    best_val = np.zeros(n, dtype=np.float64)
    take_self = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        children_val = sum(best_val[ch] for ch in nodes[i]["children"])
        self_val = eff_u[i] - tau
        if self_val >= children_val and self_val > 0.0:
            best_val[i] = self_val
            take_self[i] = True
        else:
            best_val[i] = children_val
            take_self[i] = False

    selected: List[int] = []

    def _collect(idx: int) -> None:
        if take_self[idx]:
            selected.append(idx)
        else:
            for ch in nodes[idx]["children"]:
                _collect(ch)

    for i in range(n):
        if nodes[i]["parent"] == -1:
            _collect(i)

    if not selected and not allow_empty:
        valid_idxs = [i for i in range(n) if not nodes[i].get("is_sunspot", False)]
        best_i = max(valid_idxs, key=lambda idx: float(u_hat[idx])) if valid_idxs else int(np.argmax(u_hat))
        selected = [best_i]

    return selected


def apply_spine_bridging_to_selection(
    disk_tree: Dict[str, Any],
    selected_idxs: List[int],
    u_hat: np.ndarray,
    bridge_u_hat: np.ndarray,
    tau: float,
) -> List[str]:
    """
    Build final RLE list with spine-tangent corridor bridging: map base nodes
    through exact match, ancestor, or unique selected descendant, merge across
    improving bridges (<= 14 px low-P_sep dark H-alpha corridor), and guarantee
    strict pairwise disjointness and non-emptiness.
    """
    nodes = disk_tree["nodes"]
    bridges = disk_tree.get("bridges", [])
    base_nodes = disk_tree.get("base_bridge_nodes", [])
    if not bridges or len(selected_idxs) < 2:
        return [nodes[i]["rle_counts"] for i in selected_idxs]

    sel_set = set(selected_idxs)

    def _selected_descendants(idx: int) -> List[int]:
        if idx in sel_set:
            return [idx]
        out: List[int] = []
        for ch in nodes[idx]["children"]:
            out.extend(_selected_descendants(ch))
        return out

    base_to_sel: Dict[int, int] = {}
    for bi, nidx in enumerate(base_nodes):
        if nidx in sel_set:
            base_to_sel[bi] = nidx
            continue
        cur = nodes[nidx]["parent"]
        anc_sel = -1
        while cur >= 0:
            if cur in sel_set:
                anc_sel = cur
                break
            cur = nodes[cur]["parent"]
        if anc_sel >= 0:
            base_to_sel[bi] = anc_sel
            continue
        descs = _selected_descendants(nidx)
        if len(descs) == 1:
            base_to_sel[bi] = descs[0]

    parent = {idx: idx for idx in selected_idxs}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_rles: Dict[Tuple[int, int], str] = {}
    for br_idx, br in enumerate(bridges):
        na, nb = br["node_a"], br["node_b"]
        if na not in base_to_sel or nb not in base_to_sel:
            continue
        sa, sb = base_to_sel[na], base_to_sel[nb]
        ra, rb = _find(sa), _find(sb)
        if ra == rb:
            continue
        gain_merged = float(bridge_u_hat[br_idx]) - tau
        gain_sep = (float(u_hat[sa]) - tau) + (float(u_hat[sb]) - tau)
        if gain_merged >= max(gain_sep - 0.02, 0.05):
            merged_rles[(min(sa, sb), max(sa, sb))] = br["rle_counts"]
            parent[rb] = ra

    if not merged_rles:
        return [nodes[i]["rle_counts"] for i in selected_idxs]

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in selected_idxs:
        groups[_find(idx)].append(idx)

    out_rles: List[str] = []
    occupied = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for _, members in groups.items():
        if len(members) == 1:
            m = decode_rle_counts(nodes[members[0]]["rle_counts"])
        else:
            m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for idx in members:
                m |= decode_rle_counts(nodes[idx]["rle_counts"])
            for (sa, sb), br_rle in merged_rles.items():
                if sa in members and sb in members:
                    m |= decode_rle_counts(br_rle)
        m = m & (1 - occupied)
        if m.sum() == 0:
            continue
        occupied |= m
        out_rles.append(encode_rle_counts(m))

    if not out_rles:
        out_rles = [nodes[selected_idxs[0]]["rle_counts"]]
    return out_rles


def train_and_evaluate_oof_lgbm(
    oof_trees: List[Dict[str, Any]],
    disk_folds: Dict[str, int],
    n_folds: int = 3,
    seed: int = 42,
) -> Tuple[List[lgb.Booster], lgb.Booster, Dict[str, np.ndarray], Dict[str, Any]]:
    """Train date-grouped OOF LightGBM regressors predicting expected PQ u_hat(c)."""
    X_all = np.vstack([t["features"] for t in oof_trees])
    y_all = np.concatenate([t["u_gt"] for t in oof_trees])
    f_all = np.concatenate(
        [np.full(len(t["nodes"]), disk_folds[t["disk_stem"]], dtype=np.int32) for t in oof_trees]
    )

    oof_u_hat = np.zeros_like(y_all, dtype=np.float32)
    fold_boosters: List[lgb.Booster] = []

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }

    for f in range(n_folds):
        tr_idx = f_all != f
        va_idx = f_all == f
        dtrain = lgb.Dataset(X_all[tr_idx], label=y_all[tr_idx], feature_name=FEATURE_NAMES)
        dval = lgb.Dataset(X_all[va_idx], label=y_all[va_idx], feature_name=FEATURE_NAMES, reference=dtrain)
        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=320,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        oof_u_hat[va_idx] = np.clip(booster.predict(X_all[va_idx]), 0.0, 1.0).astype(np.float32)
        fold_boosters.append(booster)

    best_iters = int(np.mean([b.best_iteration or 150 for b in fold_boosters]))
    dfull = lgb.Dataset(X_all, label=y_all, feature_name=FEATURE_NAMES)
    full_booster = lgb.train(lgb_params, dfull, num_boost_round=max(50, best_iters))

    u_hat_by_disk: Dict[str, np.ndarray] = {}
    bridge_u_by_disk: Dict[str, np.ndarray] = {}
    offset = 0
    for t in oof_trees:
        k = len(t["nodes"])
        stem = t["disk_stem"]
        u_hat_by_disk[stem] = oof_u_hat[offset : offset + k]
        offset += k
        if t["bridges"]:
            bfeats = np.stack([b["features"] for b in t["bridges"]], axis=0)
            f_id = disk_folds[stem]
            bridge_u_by_disk[stem] = np.clip(fold_boosters[f_id].predict(bfeats), 0.0, 1.0).astype(np.float32)
        else:
            bridge_u_by_disk[stem] = np.zeros(0, dtype=np.float32)

    return fold_boosters, full_booster, u_hat_by_disk, {"bridge_u_by_disk": bridge_u_by_disk}


# ---------------------------------------------------------------------------
# Submission Formatting & Strict Verification + Publication Diagnostic Plots
# ---------------------------------------------------------------------------


def write_and_validate_submission(
    csv_path: str,
    test_stems: List[str],
    preds_by_disk: Dict[str, List[str]],
    img_h: int = IMG_SIZE,
    img_w: int = IMG_SIZE,
) -> Dict[str, Any]:
    """
    Write submission CSV (Unix LF, no quotes) and enforce strict validation:
      - 100% test disk coverage
      - Zero empty masks
      - Zero overlapping masks per disk
      - Exact column header and Unix LF line endings
    """
    rows: List[Tuple[str, str]] = []
    for stem in sorted(test_stems):
        rles = preds_by_disk.get(stem, [])
        assert len(rles) >= 1, f"Test disk {stem} has 0 predicted masks!"
        occupied = np.zeros((img_h, img_w), dtype=np.uint8)
        kept = 0
        for counts in rles:
            assert '"' not in counts and "'" not in counts, "RLE counts must not contain quotes!"
            m = decode_rle_counts(counts, img_h, img_w)
            assert m.shape == (img_h, img_w), f"Invalid mask shape {m.shape}"
            assert int(m.sum()) > 0, f"Empty mask found on {stem}"
            overlap = int((occupied & m).sum())
            assert overlap == 0, f"Overlapping masks ({overlap} px) found on {stem}"
            occupied |= m
            kept += 1
            rows.append((f"{stem}_{kept}", counts))

    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            f.write(f"{fid},{rle}\n")

    raw_bytes = open(csv_path, "rb").read()
    assert b"\r" not in raw_bytes, f"CR byte detected in {csv_path}"
    return {"path": csv_path, "n_rows": len(rows), "n_disks": len(test_stems)}


def save_diagnostic_plots(
    out_dir: str,
    metrics_opt: Dict[str, Any],
    metrics_raw: Dict[str, Any],
    tau_curve: List[Dict[str, float]],
    tau_opt: float,
    qual_samples: List[Dict[str, Any]],
) -> None:
    """Save quantitative_rubric_histograms.png and qualitative_morphology_panels.png."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=180)

    # Top-left: IoU > 0 Distribution (matching self-evaluation-notebook.ipynb)
    ax = axes[0, 0]
    ious = metrics_opt["nonzero_ious"]
    bins = np.linspace(0.0, 1.0, 51)
    if ious.size > 0:
        counts, edges = np.histogram(ious, bins=bins)
        probs = counts / max(counts.sum(), 1)
        ax.bar(0.5 * (edges[:-1] + edges[1:]), probs, width=0.018, color="#1f4e79", edgecolor="black", alpha=0.8)
        ax.axvline(metrics_opt["iou_mean"], color="crimson", ls="--", lw=1.6, label=f"Mean={metrics_opt['iou_mean']:.3f}")
        ax.axvline(metrics_opt["iou_p50"], color="darkorange", ls=":", lw=1.6, label=f"Median={metrics_opt['iou_p50']:.3f}")
        ax.axvline(0.5, color="green", ls="-.", lw=1.2, label="TP IoU=0.50")
        ax.legend(fontsize=8)
    ax.set_title("OOF Pairwise IoU Distribution (IoU > 0)")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", ls="--", alpha=0.5)

    # Top-right: Dice > 0 Distribution (matching self-evaluation-notebook.ipynb)
    ax = axes[0, 1]
    dices = metrics_opt["nonzero_dices"]
    if dices.size > 0:
        counts, edges = np.histogram(dices, bins=bins)
        probs = counts / max(counts.sum(), 1)
        ax.bar(0.5 * (edges[:-1] + edges[1:]), probs, width=0.018, color="#2b6cb0", edgecolor="black", alpha=0.8)
        ax.axvline(metrics_opt["dice_mean"], color="crimson", ls="--", lw=1.6, label=f"Mean={metrics_opt['dice_mean']:.3f}")
        ax.axvline(metrics_opt["dice_p50"], color="darkorange", ls=":", lw=1.6, label=f"Median={metrics_opt['dice_p50']:.3f}")
        ax.legend(fontsize=8)
    ax.set_title("OOF Pairwise Dice Distribution (Dice > 0)")
    ax.set_xlabel("Dice")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", ls="--", alpha=0.5)

    # Bottom-left: 1:n and n:1 Matching Degree Distribution (IoU > 0)
    ax = axes[1, 0]
    offset = 0.5
    gt_hist = metrics_opt["gt_degree_hist"]
    pred_hist = metrics_opt["pred_degree_hist"]
    gt_bars = {deg + offset: cnt for deg, cnt in gt_hist.items() if deg <= 6}
    pred_bars = {-(deg + offset): cnt for deg, cnt in pred_hist.items() if deg <= 6}
    for pos, cnt in sorted(gt_bars.items()):
        deg = int(round(pos - offset))
        ax.bar(pos, cnt, width=0.8, color="crimson" if deg == 0 else "forestgreen", edgecolor="black", alpha=0.75)
    for pos, cnt in sorted(pred_bars.items()):
        deg = int(round(-pos - offset))
        ax.bar(pos, cnt, width=0.8, color="darkorange" if deg == 0 else "steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(0.0, color="black", lw=0.9)
    ticks = sorted(list(gt_bars.keys()) + list(pred_bars.keys()))
    labels = [f"{abs(int(round(v + offset)))}:1" if v < 0 else f"1:{int(round(v - offset))}" for v in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_title("Many-to-One (n:1) and One-to-Many (1:n) Relations")
    ax.set_xlabel("Matching Case")
    ax.set_ylabel("Count")
    ax.grid(axis="y", ls="--", alpha=0.5)
    handles = [
        Patch(facecolor="forestgreen", edgecolor="black", label=f"GT Hit (1:n): {metrics_opt['gt_1_to_1'] + metrics_opt['gt_1_to_many']}"),
        Patch(facecolor="crimson", edgecolor="black", label=f"GT Miss (1:0): {metrics_opt['gt_1_to_0_miss']}"),
        Patch(facecolor="darkorange", edgecolor="black", label=f"Pred Miss (0:1): {metrics_opt['pred_0_to_1_fp']}"),
        Patch(facecolor="steelblue", edgecolor="black", label=f"Pred Hit (n:1): {metrics_opt['pred_1_to_1'] + metrics_opt['pred_many_to_1']}"),
    ]
    ax.legend(handles=handles, fontsize=8)

    # Bottom-right: Tau Sweep PQ / SQ / RQ + Raw Baseline Ablation
    ax = axes[1, 1]
    if tau_curve:
        taus = [r["tau"] for r in tau_curve]
        ax.plot(taus, [r["pq"] for r in tau_curve], "o-", color="#111111", lw=2, ms=3, label="Tree-DP PQ")
        ax.plot(taus, [r["sq"] for r in tau_curve], "--", color="#2b6cb0", lw=1.5, label="SQ")
        ax.plot(taus, [r["rq"] for r in tau_curve], ":", color="#2f855a", lw=1.5, label="RQ")
        ax.axvline(tau_opt, color="crimson", ls="--", lw=1.4, label=f"tau*={tau_opt:.3f} (PQ={metrics_opt['pq']:.4f})")
        ax.axhline(metrics_raw["pq"], color="gray", ls="-.", lw=1.2, label=f"Raw P_eff>=0.40 PQ={metrics_raw['pq']:.4f}")
    ax.set_title("Exact OOF Panoptic Quality vs Tree-DP Threshold tau")
    ax.set_xlabel("tau")
    ax.set_ylabel("Score")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "quantitative_rubric_histograms.png"), bbox_inches="tight")
    plt.close(fig)

    if not qual_samples:
        return
    n_rows = min(4, len(qual_samples))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.0 * n_rows), dpi=160)
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = [
        "Raw H-alpha & Solar Limb",
        "CLV Contrast + P_fg(R) / P_spine(G) / P_sep(B)",
        "MAGFiLO Multi-Annotator GT",
        "MAGFiLO Tree-DP Predictions",
    ]
    for r_idx in range(n_rows):
        s = qual_samples[r_idx]
        axes[r_idx, 0].imshow(s["gray_ds"], cmap="gray")
        axes[r_idx, 0].set_title(f"{s['disk_stem']} (Raw)", fontsize=9)
        axes[r_idx, 1].imshow(s["prob_rgb"])
        axes[r_idx, 1].set_title(col_titles[1] if r_idx == 0 else "P_fg/P_spine/P_sep", fontsize=9)
        axes[r_idx, 2].imshow(s["gt_rgb"])
        axes[r_idx, 2].set_title(col_titles[2] if r_idx == 0 else f"GT ({s['n_gt']} inst)", fontsize=9)
        axes[r_idx, 3].imshow(s["pred_rgb"])
        axes[r_idx, 3].set_title(f"Pred ({s['n_pred']} inst, PQ={s['disk_pq']:.3f})", fontsize=9)
        for c_idx in range(4):
            axes[r_idx, c_idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qualitative_morphology_panels.png"), bbox_inches="tight")
    plt.close(fig)


def _colorize_instances(gray_ds: np.ndarray, rle_list: List[str], ds_size: int = 512) -> np.ndarray:
    """Overlay distinct deterministic pastel colors for instances onto grayscale H-alpha."""
    base = np.stack([gray_ds, gray_ds, gray_ds], axis=-1).astype(np.float32) / 255.0
    palette = [
        (0.95, 0.25, 0.25),
        (0.20, 0.85, 0.35),
        (0.25, 0.65, 0.98),
        (0.95, 0.78, 0.15),
        (0.85, 0.30, 0.90),
        (0.15, 0.90, 0.85),
        (0.98, 0.50, 0.15),
    ]
    for idx, counts in enumerate(rle_list):
        m = decode_rle_counts(counts)
        m_ds = cv2.resize(m, (ds_size, ds_size), interpolation=cv2.INTER_NEAREST) > 0
        col = np.asarray(palette[idx % len(palette)], dtype=np.float32)
        base[m_ds] = 0.45 * base[m_ds] + 0.55 * col[None, :]
    return np.clip(base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Synthetic Dry-Run Generator (Self-Contained Verification at Full 2048x2048)
# ---------------------------------------------------------------------------


def create_synthetic_magfilo_suite(base_dir: str, n_train: int = 6, n_test: int = 2) -> Tuple[str, str, str]:
    """
    Generate synthetic 2048x2048 GONG H-alpha solar disks + MAGFiLO COCO JSON
    featuring limb darkening, curved filaments, closely spaced parallel
    filaments, collinear fragments (spine bridging test), and dark circular
    sunspot umbrae for full end-to-end pipeline verification.
    """
    train_dir = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "train", "train_images")
    test_dir = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "test", "test_images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    json_path = os.path.join(base_dir, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    images_json: List[Dict[str, Any]] = []
    anns_json: List[Dict[str, Any]] = []
    ann_uid = 1

    def _draw_synthetic_disk(seed_offset: int) -> Tuple[np.ndarray, List[Tuple[np.ndarray, List[float]]]]:
        local_rng = np.random.default_rng(1000 + seed_offset)
        cx = 1024.0 + float(local_rng.uniform(-20, 20))
        cy = 1024.0 + float(local_rng.uniform(-20, 20))
        R = 880.0 + float(local_rng.uniform(-25, 25))
        rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(rho, 1.0) ** 2))
        disk_bool = rho <= 1.0

        bg = np.where(disk_bool, 75.0 + 95.0 * (0.35 + 0.65 * mu), 0.0)
        bg += np.where(disk_bool, local_rng.normal(0.0, 4.0, (IMG_SIZE, IMG_SIZE)), 0.0)

        fils: List[Tuple[np.ndarray, List[float]]] = []
        t = np.linspace(0.0, 1.0, 40)
        sx = cx - 350.0 + 420.0 * t + float(local_rng.uniform(-50, 50))
        sy = cy - 220.0 + 180.0 * np.sin(t * 2.5) + float(local_rng.uniform(-40, 40))
        m1 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        pts1 = np.stack([sx, sy], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m1, [pts1], False, 1, thickness=18)
        fils.append((m1, np.column_stack([sx, sy]).ravel().tolist()))

        for _, dy_off in [(1, 0), (-1, 24)]:
            m_par = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            px = cx + 120.0 + np.linspace(0, 220, 25)
            py = cy + 200.0 + dy_off + 20.0 * np.sin(np.linspace(0, 2.0, 25))
            pts_p = np.stack([px, py], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(m_par, [pts_p], False, 1, thickness=16)
            fils.append((m_par, np.column_stack([px, py]).ravel().tolist()))

        for x_a, x_b in [(cx - 420, cx - 260), (cx - 246, cx - 100)]:
            m_fr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            fx = np.linspace(x_a, x_b, 16)
            fy = np.full_like(fx, cy + 260.0)
            pts_f = np.stack([fx, fy], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(m_fr, [pts_f], False, 1, thickness=14)
            fils.append((m_fr, np.column_stack([fx, fy]).ravel().tolist()))

        for fm, _ in fils:
            bg = np.where(fm > 0, bg * 0.52, bg)

        cv2.circle(bg, (int(cx + 260), int(cy - 240)), 22, 20.0, -1)
        cv2.circle(bg, (int(cx - 100), int(cy - 300)), 38, 205.0, -1)
        return np.clip(np.round(bg), 0, 255).astype(np.uint8), fils

    dates = ["20250101", "20250102", "20250103"]
    for i in range(n_train):
        dt = dates[i % len(dates)]
        stem = f"{dt}12000{i}Bh"
        fn = f"{stem}.jpeg"
        img_u8, fils = _draw_synthetic_disk(i)
        cv2.imwrite(os.path.join(train_dir, fn), img_u8)

        for ann_id in ["010101", "010102"]:
            img_uid = f"{ann_id}-{stem}"
            images_json.append(
                {
                    "id": img_uid,
                    "width": IMG_SIZE,
                    "height": IMG_SIZE,
                    "file_name": fn,
                    "license": 1,
                    "date_captured": dt,
                }
            )
            for fm, sp in fils:
                contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                poly = cnt.reshape(-1, 2).astype(float).ravel().tolist()
                if len(poly) >= 6:
                    poly.extend([poly[0], poly[1]])
                    x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                    anns_json.append(
                        {
                            "id": f"ann_{ann_uid}",
                            "image_id": img_uid,
                            "category_id": 1,
                            "segmentation": [poly],
                            "area": float(fm.sum()),
                            "spine": sp,
                            "bbox": [float(x_b), float(y_b), float(w_b), float(h_b)],
                            "iscrowd": 0,
                        }
                    )
                    ann_uid += 1

    for j in range(n_test):
        stem = f"2026020{j + 1}120000Ch"
        img_u8, _ = _draw_synthetic_disk(100 + j)
        cv2.imwrite(os.path.join(test_dir, f"{stem}.jpeg"), img_u8)

    coco = {
        "info": {"description": "Synthetic MAGFiLO Dry-Run"},
        "images": images_json,
        "annotations": anns_json,
        "licenses": [{"id": 1, "name": "CC0", "url": ""}],
        "categories": [{"id": 1, "name": "Left", "supercategory": "filament"}],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return json_path, train_dir, test_dir


# ---------------------------------------------------------------------------
# Main End-to-End Training & Evaluation Orchestration
# ---------------------------------------------------------------------------


def benchmark_cpu_inference_latency(
    backbone_name: str, device: torch.device, infer_size: int = 1024
) -> float:
    """Measure single-pass channels_last CPU inference latency for 1 disk."""
    model = MagfiloPanopticUNet(backbone_name=backbone_name, pretrained=False).to(device)
    model = model.to(memory_format=torch.channels_last).eval()
    x = torch.randn(1, 3, infer_size, infer_size, device=device).to(memory_format=torch.channels_last)
    with torch.inference_mode():
        _ = model(x)
        t0 = time.time()
        _ = model(x)
        dt = time.time() - t0
    return float(max(dt, 0.05))


def train_one_fold(
    fold: int,
    train_disks: List[Dict[str, Any]],
    val_disks: List[Dict[str, Any]],
    gt_by_disk: Dict[str, List[List[Dict[str, Any]]]],
    is_gpu: bool,
    backbone_name: str,
    pretrained: bool,
    epochs: int,
    ckpt_dir: str,
    dry_run: bool,
    cpu_fold_budget: float,
    cpu_infer_size: int,
    t_pipeline0: float,
    reserved_post_train_sec: float,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train Stage N MagfiloPanopticUNet on one date-grouped fold, save compact
    float16 weights, and extract Stage H component trees on OOF validation disks.
    """
    device = torch.device("cuda" if is_gpu else "cpu")
    model = MagfiloPanopticUNet(backbone_name=backbone_name, pretrained=pretrained).to(device)
    if not is_gpu:
        model = model.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters())
    fp16_mb = (n_params * 2) / (1024.0 * 1024.0)
    log_msg(f"Fold {fold} model={backbone_name} params={n_params/1e6:.2f}M est_fp16={fp16_mb:.2f} MB")
    assert fp16_mb < 28.5, f"Model fp16 size {fp16_mb:.2f} MB risks >= 30 MB repository limit!"

    criterion = SolarMultiTaskLoss(cl_iters=3 if dry_run else 5).to(device)
    lr = 5e-4 if is_gpu else 6e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if is_gpu:
        gpu_sz = 512 if dry_run else 1536
        ds = GpuFullDiskDataset(train_disks, target_size=gpu_sz, train=True, seed=42 + fold * 100)
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        grad_accum = 4
        for ep in range(epochs):
            t_ep = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for step, (xb, yb, mb) in enumerate(loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb, mb) / float(grad_accum)
                scaler.scale(loss).backward()
                loss_sum += float(loss.item() * grad_accum)
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            log_msg(f"  [GPU Fold {fold}] Ep {ep+1}/{epochs} loss={loss_sum/max(len(loader),1):.4f} ({time.time()-t_ep:.1f}s)")
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        patch_sz = 384
        batch_sz = 4 if dry_run else 6
        target_epochs = 2 if dry_run else epochs
        rng_ep = np.random.default_rng(1000 * fold + 7)

        # Measure Batch-1 forward+backward speed to dynamically cap steps/epoch
        t_b0 = time.time()
        bx, by, bm = sample_cpu_patches_from_disks(
            train_disks, batch_sz, patch_size=patch_sz, disk_scale_size=cpu_infer_size, rng=rng_ep
        )
        t_sample_batch1 = time.time() - t_b0
        model.train()
        t_fw0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        xb_t = torch.from_numpy(bx).to(memory_format=torch.channels_last)
        logits = model(xb_t)
        loss = criterion(logits, torch.from_numpy(by), torch.from_numpy(bm))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        t_batch1 = time.time() - t_fw0

        if dry_run:
            patches_per_ep = 8
        else:
            avail_global = MAX_CPU_RUNTIME_SEC - (time.time() - t_pipeline0) - reserved_post_train_sec
            eff_fold_budget = max(120.0, min(cpu_fold_budget, avail_global))
            max_batches_total = max(
                target_epochs * 4,
                int((eff_fold_budget * 0.85) / max(t_batch1 + 0.10, 0.25)),
            )
            steps_per_ep = int(np.clip(max_batches_total // max(target_epochs, 1), 4, 48))
            patches_per_ep = steps_per_ep * batch_sz
        log_msg(
            f"  [CPU Fold {fold}] Batch-1 ({batch_sz}x{patch_sz})={t_batch1:.2f}s "
            f"(sample={t_sample_batch1:.2f}s) -> {patches_per_ep} patches/ep ({patches_per_ep//batch_sz} steps) "
            f"x {target_epochs} epochs (budget={cpu_fold_budget:.0f}s)"
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(target_epochs, 1), eta_min=1e-5
        )
        t_fold_start = time.time()
        for ep in range(target_epochs):
            rem_global = MAX_CPU_RUNTIME_SEC - (time.time() - t_pipeline0) - reserved_post_train_sec
            if not dry_run and ((time.time() - t_fold_start) > cpu_fold_budget or rem_global < 60.0):
                log_msg(f"  [CPU Fold {fold}] Dynamic runtime cap reached at ep {ep}; stopping training cleanly.")
                break
            t_ep = time.time()
            px, py, pm = sample_cpu_patches_from_disks(
                train_disks,
                patches_per_ep,
                patch_size=patch_sz,
                disk_scale_size=cpu_infer_size,
                rng=rng_ep,
            )
            perm = rng_ep.permutation(px.shape[0])
            model.train()
            loss_sum = 0.0
            steps = 0
            for i in range(0, px.shape[0], batch_sz):
                rem_step = MAX_CPU_RUNTIME_SEC - (time.time() - t_pipeline0) - reserved_post_train_sec
                if not dry_run and steps > 0 and ((time.time() - t_fold_start) > cpu_fold_budget or rem_step < 30.0):
                    break
                sel = perm[i : i + batch_sz]
                optimizer.zero_grad(set_to_none=True)
                xb_s = torch.from_numpy(px[sel]).to(memory_format=torch.channels_last)
                logits = model(xb_s)
                loss = criterion(logits, torch.from_numpy(py[sel]), torch.from_numpy(pm[sel]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.item())
                steps += 1
            scheduler.step()
            log_msg(
                f"  [CPU Fold {fold}] Ep {ep+1}/{target_epochs} steps={steps} "
                f"loss={loss_sum/max(steps,1):.4f} ({time.time()-t_ep:.1f}s)"
            )

    ckpt_path = os.path.join(ckpt_dir, f"unet_fold{fold}_fp16.pt")
    fp16_state = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "backbone": backbone_name,
            "fold": fold,
            "state_dict_fp16": fp16_state,
        },
        ckpt_path,
    )
    ckpt_mb = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
    log_msg(f"Saved Fold {fold} checkpoint: {ckpt_path} ({ckpt_mb:.2f} MB)")

    # OOF Validation Inference + Immediate Stage H Tree Extraction
    oof_trees: List[Dict[str, Any]] = []
    qual_snapshots: List[Dict[str, Any]] = []
    t_val0 = time.time()
    for vi, d in enumerate(val_disks):
        gray = cv2.imread(d["img_path"], cv2.IMREAD_GRAYSCALE)
        prep = preprocess_solar_disk(gray, disk_params=d["disk_params"])
        if is_gpu:
            p_fg, p_sp, p_sep = infer_disk_gpu(
                [model], prep, device, img_size=512 if dry_run else 1536, use_tta=True
            )
        else:
            p_fg, p_sp, p_sep = infer_disk_cpu_tight(
                [model], prep, device, infer_size=cpu_infer_size
            )

        tree = extract_disk_component_tree(
            d["disk_stem"],
            p_fg,
            p_sp,
            p_sep,
            prep,
            gt_annotators=gt_by_disk.get(d["disk_stem"], []),
        )
        p_eff = np.clip(0.75 * p_fg + 0.25 * p_sp - 0.35 * p_sep, 0.0, 1.0) * prep["disk_mask"]
        raw_bw = (p_eff >= 0.40).astype(np.uint8)
        n_rl, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(raw_bw, connectivity=8)
        raw_rles: List[str] = []
        for cid in range(1, n_rl):
            if int(r_stats[cid, cv2.CC_STAT_AREA]) >= MIN_COMPONENT_AREA:
                raw_rles.append(encode_rle_counts((r_labels == cid).astype(np.uint8)))
        tree["raw_baseline_rles"] = raw_rles
        oof_trees.append(tree)

        if len(qual_snapshots) < 2:
            ds_sz = 512
            gray_ds = cv2.resize(gray, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA)
            clv_ds = cv2.resize(prep["x_3ch"][0], (ds_sz, ds_sz), interpolation=cv2.INTER_AREA)
            prob_rgb = np.stack(
                [
                    cv2.resize(p_fg, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA),
                    cv2.resize(p_sp, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA),
                    cv2.resize(p_sep, (ds_sz, ds_sz), interpolation=cv2.INTER_AREA),
                ],
                axis=-1,
            )
            prob_overlay = np.clip(0.42 * clv_ds[:, :, None] + 0.75 * prob_rgb, 0.0, 1.0)
            gt0_rles = []
            anns0 = gt_by_disk.get(d["disk_stem"], [])
            if anns0:
                gt0_rles = [
                    r["counts"].decode("utf-8") if isinstance(r["counts"], bytes) else r["counts"]
                    for r in anns0[0]
                ]
            qual_snapshots.append(
                {
                    "disk_stem": d["disk_stem"],
                    "gray_ds": gray_ds,
                    "prob_rgb": prob_overlay,
                    "gt_rgb": _colorize_instances(gray_ds, gt0_rles, ds_sz),
                    "n_gt": len(gt0_rles),
                }
            )

        if (vi + 1) % 50 == 0 or (vi + 1) == len(val_disks):
            log_msg(f"  Fold {fold} OOF extracted: {vi+1}/{len(val_disks)} disks ({time.time()-t_val0:.1f}s)")

    return ckpt_path, oof_trees, qual_snapshots


def load_fp16_model_for_eval(ckpt_path: str, device: torch.device) -> nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MagfiloPanopticUNet(backbone_name=payload["backbone"], pretrained=False)
    fp32_state = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in payload["state_dict_fp16"].items()
    }
    model.load_state_dict(fp32_state, strict=True)
    model = model.to(device)
    if device.type == "cpu":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def run_test_inference_and_write_submissions(
    test_files: List[str],
    active_test_models: List[nn.Module],
    fold_boosters: List[lgb.Booster],
    out_dir: str,
    device: torch.device,
    is_gpu: bool,
    infer_size: int,
    best_tau: float,
    tau_hp: float,
) -> Tuple[List[str], Dict[str, Any]]:
    """Run multi-fold Stage P/N/H ensemble inference on test disks and write validated CSVs."""
    log_msg(
        f"Running {len(active_test_models)}-fold ensemble test inference "
        f"({infer_size}x{infer_size}, gpu={is_gpu}) on {len(test_files)} test disks..."
    )
    os.makedirs(out_dir, exist_ok=True)

    test_stems: List[str] = []
    test_preds_opt: Dict[str, List[str]] = {}
    test_preds_hp: Dict[str, List[str]] = {}
    test_preds_bridged: Dict[str, List[str]] = {}

    t_test0 = time.time()
    for ti, tpath in enumerate(test_files):
        stem = os.path.splitext(os.path.basename(tpath))[0]
        test_stems.append(stem)
        gray = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE)
        prep = preprocess_solar_disk(gray)

        if is_gpu:
            p_fg, p_sp, p_sep = infer_disk_gpu(
                active_test_models, prep, device, img_size=infer_size, use_tta=True
            )
        else:
            p_fg, p_sp, p_sep = infer_disk_cpu_tight(
                active_test_models, prep, device, infer_size=infer_size
            )

        ttree = extract_disk_component_tree(stem, p_fg, p_sp, p_sep, prep, gt_annotators=None)
        u_test = np.mean([b.predict(ttree["features"]) for b in fold_boosters], axis=0)
        u_test = np.clip(u_test, 0.0, 1.0).astype(np.float32)

        if ttree["bridges"]:
            bfeats = np.stack([b["features"] for b in ttree["bridges"]], axis=0)
            bu_test = np.clip(np.mean([b.predict(bfeats) for b in fold_boosters], axis=0), 0.0, 1.0)
        else:
            bu_test = np.zeros(0, dtype=np.float32)

        sel_opt = solve_tree_dp_antichain(ttree["nodes"], u_test, best_tau, allow_empty=False)
        sel_hp = solve_tree_dp_antichain(ttree["nodes"], u_test, tau_hp, allow_empty=False)
        test_preds_opt[stem] = [ttree["nodes"][i]["rle_counts"] for i in sel_opt]
        test_preds_hp[stem] = [ttree["nodes"][i]["rle_counts"] for i in sel_hp]
        test_preds_bridged[stem] = apply_spine_bridging_to_selection(
            ttree, sel_opt, u_test, bu_test, best_tau
        )

        if (ti + 1) % 45 == 0 or (ti + 1) == len(test_files):
            log_msg(f"  Test inferred: {ti+1}/{len(test_files)} disks ({time.time()-t_test0:.1f}s)")

    sub_reports = {
        "submission_pq_optimal.csv": write_and_validate_submission(
            os.path.join(out_dir, "submission_pq_optimal.csv"), test_stems, test_preds_opt
        ),
        "submission_high_precision.csv": write_and_validate_submission(
            os.path.join(out_dir, "submission_high_precision.csv"), test_stems, test_preds_hp
        ),
        "submission_spine_bridged.csv": write_and_validate_submission(
            os.path.join(out_dir, "submission_spine_bridged.csv"), test_stems, test_preds_bridged
        ),
    }
    return test_stems, sub_reports


def _resolve_weights_dir(args: argparse.Namespace) -> str:
    """Resolve directory containing trained fp16 UNet weights, LightGBM scorers, and postproc config."""
    candidates = []
    if args.weights_dir:
        candidates.append(args.weights_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(script_dir, "weights"),
            os.path.join(args.out_dir, "checkpoints"),
        ]
    )
    for cand in candidates:
        if cand and os.path.isdir(cand) and os.path.exists(os.path.join(cand, "postproc_config.json")):
            return cand
    raise FileNotFoundError(f"Could not find weights dir with postproc_config.json in {candidates}")


def run_predict_only(args: argparse.Namespace) -> None:
    """Reproduce test CSV predictions directly from stored fp16 UNet + LightGBM weights without retraining."""
    t0 = time.time()
    set_seed(42)

    is_gpu = torch.cuda.is_available() and (not args.force_cpu)
    device = torch.device("cuda" if is_gpu else "cpu")
    if not is_gpu:
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    weights_dir = _resolve_weights_dir(args)
    cfg_path = os.path.join(weights_dir, "postproc_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        postproc_cfg = json.load(f)

    ckpt_paths = sorted(
        os.path.join(weights_dir, fn)
        for fn in os.listdir(weights_dir)
        if fn.startswith("unet_fold") and fn.endswith(".pt")
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No unet_fold*.pt checkpoints found in {weights_dir}")

    booster_paths = sorted(
        os.path.join(weights_dir, fn)
        for fn in os.listdir(weights_dir)
        if fn.startswith("lgb_scorer_fold") and fn.endswith(".txt")
    )
    if not booster_paths:
        fallback_lgb = os.path.join(weights_dir, "lgb_scorer.txt")
        if not os.path.exists(fallback_lgb):
            fallback_lgb = os.path.join(os.path.dirname(weights_dir), "scaler", "lgb_scorer.txt")
        if os.path.exists(fallback_lgb):
            booster_paths = [fallback_lgb]
    if not booster_paths:
        raise FileNotFoundError(f"No lgb_scorer*.txt models found in {weights_dir}")

    log_msg(
        f"PREDICT-ONLY MODE: weights_dir={weights_dir} | device={device} | "
        f"unet_folds={len(ckpt_paths)} | lgb_boosters={len(booster_paths)}"
    )
    eval_models = [load_fp16_model_for_eval(p, device) for p in ckpt_paths]
    fold_boosters = [lgb.Booster(model_file=p) for p in booster_paths]

    tmp_work = os.path.join(out_dir, "_tmp_predict_work") if args.dry_run else None
    if args.dry_run:
        log_msg("PREDICT-ONLY DRY-RUN: Generating synthetic 2048x2048 test solar disks...")
        _, _, test_dir = create_synthetic_magfilo_suite(os.path.join(tmp_work, "syn_data"), n_train=2, n_test=2)
    else:
        test_dir = discover_test_dir(args.data_root, args.test_dir)

    test_files = sorted(
        os.path.join(test_dir, fn)
        for fn in os.listdir(test_dir)
        if fn.lower().endswith((".jpeg", ".jpg", ".png"))
    )
    if not test_files:
        raise FileNotFoundError(f"No image files found in {test_dir}")

    cfg_infer_size = int(postproc_cfg.get("infer_size", 1536))
    infer_size = int(args.cpu_infer_size) if args.cpu_infer_size else (512 if args.dry_run else cfg_infer_size)
    best_tau = float(postproc_cfg["tau_optimal"])
    tau_hp = float(postproc_cfg["tau_high_precision"])

    test_stems, sub_reports = run_test_inference_and_write_submissions(
        test_files=test_files,
        active_test_models=eval_models,
        fold_boosters=fold_boosters,
        out_dir=out_dir,
        device=device,
        is_gpu=is_gpu,
        infer_size=infer_size,
        best_tau=best_tau,
        tau_hp=tau_hp,
    )

    predict_summary = {
        "mode": "predict_only",
        "accelerator": "gpu" if is_gpu else "cpu",
        "weights_dir": os.path.abspath(weights_dir),
        "backbone": postproc_cfg.get("backbone", "resnet18"),
        "infer_size": infer_size,
        "n_unet_folds": len(eval_models),
        "n_lgbm_boosters": len(fold_boosters),
        "tau_optimal": best_tau,
        "tau_high_precision": tau_hp,
        "n_test_disks": len(test_stems),
        "runtime_sec": round(time.time() - t0, 2),
        "submissions": sub_reports,
    }
    with open(os.path.join(out_dir, "predict_summary.json"), "w", encoding="utf-8") as f:
        json.dump(predict_summary, f, indent=2)

    if tmp_work and os.path.exists(tmp_work):
        shutil.rmtree(tmp_work, ignore_errors=True)

    log_msg(f"PREDICT-ONLY DONE in {time.time() - t0:.1f}s. Submissions: {sub_reports}")


def run_pipeline(args: argparse.Namespace) -> None:
    if args.predict_only:
        run_predict_only(args)
        return
    t_pipeline0 = time.time()
    set_seed(42)

    is_gpu = torch.cuda.is_available() and (not args.force_cpu)
    device = torch.device("cuda" if is_gpu else "cpu")
    out_dir = args.out_dir
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    scaler_dir = os.path.join(out_dir, "scaler")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    tmp_work = os.path.join(out_dir, "_tmp_work") if args.dry_run else "/tmp/magfilo_panoptic_work"
    os.makedirs(tmp_work, exist_ok=True)

    if args.dry_run:
        log_msg("DRY-RUN MODE: Generating synthetic 2048x2048 MAGFiLO suite...")
        syn_dir = os.path.join(tmp_work, "syn_data")
        train_json, train_dir, test_dir = create_synthetic_magfilo_suite(syn_dir)
    else:
        train_json, train_dir, test_dir = discover_competition_paths(
            args.data_root, args.train_json, args.train_dir, args.test_dir
        )

    test_files = sorted(
        [
            os.path.join(test_dir, fn)
            for fn in os.listdir(test_dir)
            if fn.lower().endswith((".jpeg", ".jpg", ".png"))
        ]
    )
    log_msg(
        f"Environment: device={device}, is_gpu={is_gpu}, timm={HAS_TIMM}, "
        f"torchvision={HAS_TORCHVISION}, test_files={len(test_files)}"
    )

    # Stage 0: Parse JSON, cache target masks, assign 3 date-grouped folds
    disks, gt_records = build_magfilo_dataset_and_cache(
        train_json, train_dir, os.path.join(tmp_work, "targets"), n_folds=3
    )
    disk_folds = {d["disk_stem"]: d["fold"] for d in disks}
    gt_by_disk: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    for rec in gt_records:
        gt_by_disk[rec["disk_stem"]].append(rec["gt_rles"])

    if is_gpu:
        backbone_name = "resnet18"  # ~24 MB fp16/fold -> ~72 MB total
        epochs = 2 if args.dry_run else 18
        cpu_fold_budget = 0.0
        cpu_infer_size = 1536
        reserved_post_train_sec = 0.0
        test_ensemble_k = 3
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        backbone_name = "mobilenetv3_large_100" if HAS_TIMM else "resnet18"
        epochs = 2 if args.dry_run else 15
        cpu_infer_size = 1024
        t_inf_one = benchmark_cpu_inference_latency(backbone_name, device, infer_size=cpu_infer_size)
        n_tr = len(disks)
        n_te = len(test_files)
        est_stage_h = (n_tr + n_te) * 0.45 + 150.0
        est_oof_inf = n_tr * t_inf_one * 1.12
        est_test_3 = (n_te * 3) * t_inf_one * 1.12
        elapsed_now = time.time() - t_pipeline0

        test_ensemble_k = 3
        if not args.dry_run and (elapsed_now + est_oof_inf + est_test_3 + est_stage_h + 1200.0) > MAX_CPU_RUNTIME_SEC:
            test_ensemble_k = 1
        est_test_inf = (n_te * test_ensemble_k) * t_inf_one * 1.12
        reserved_post_train_sec = est_oof_inf + est_test_inf + est_stage_h
        avail_train_total = max(360.0, MAX_CPU_RUNTIME_SEC - elapsed_now - reserved_post_train_sec)
        cpu_fold_budget = avail_train_total / 3.0
        log_msg(
            f"CPU Budget Guard: inf_{cpu_infer_size}={t_inf_one:.2f}s/pass | "
            f"est_oof={est_oof_inf/60:.1f}m + est_test(k={test_ensemble_k})={est_test_inf/60:.1f}m "
            f"+ stageH={est_stage_h/60:.1f}m -> train_budget={cpu_fold_budget/60:.1f}m/fold"
        )

    # Stage N Training + OOF Inference + Stage H Tree Extraction
    ckpt_paths: List[str] = []
    oof_trees: List[Dict[str, Any]] = []
    qual_snapshots: List[Dict[str, Any]] = []

    for f in range(3):
        tr_disks = [d for d in disks if d["fold"] != f]
        va_disks = [d for d in disks if d["fold"] == f]
        if not is_gpu and not args.dry_run:
            rem_f = 3 - f
            n_oof_rem = sum(len([d for d in disks if d["fold"] == ff]) for ff in range(f, 3))
            post_rem = (n_oof_rem + len(test_files) * test_ensemble_k) * t_inf_one * 1.12 + (
                len(disks) + len(test_files)
            ) * 0.45 + 120.0
            avail_tr_rem = max(120.0, MAX_CPU_RUNTIME_SEC - (time.time() - t_pipeline0) - post_rem)
            fold_budget_cur = avail_tr_rem / float(rem_f)
            post_reserve_cur = post_rem + fold_budget_cur * float(rem_f - 1)
        else:
            fold_budget_cur = cpu_fold_budget
            post_reserve_cur = reserved_post_train_sec

        ckpt_path, fold_trees, fold_snaps = train_one_fold(
            fold=f,
            train_disks=tr_disks,
            val_disks=va_disks,
            gt_by_disk=gt_by_disk,
            is_gpu=is_gpu,
            backbone_name=backbone_name,
            pretrained=(not args.dry_run),
            epochs=epochs,
            ckpt_dir=ckpt_dir,
            dry_run=args.dry_run,
            cpu_fold_budget=fold_budget_cur,
            cpu_infer_size=cpu_infer_size,
            t_pipeline0=t_pipeline0,
            reserved_post_train_sec=post_reserve_cur,
        )
        ckpt_paths.append(ckpt_path)
        oof_trees.extend(fold_trees)
        qual_snapshots.extend(fold_snaps[:2])

    ckpt_sizes_mb = {os.path.basename(p): round(os.path.getsize(p) / (1024.0 * 1024.0), 3) for p in ckpt_paths}
    total_ckpt_mb = sum(ckpt_sizes_mb.values())
    log_msg(f"Checkpoint fp16 sizes (MB): {ckpt_sizes_mb}, total={total_ckpt_mb:.2f} MB")
    assert all(sz < 30.0 for sz in ckpt_sizes_mb.values()), "Checkpoint file >= 30 MB!"
    assert total_ckpt_mb < 80.0, "Total checkpoint size >= 80 MB!"

    # Stage H: Grouped-OOF LightGBM expected-PQ scorer + Tree DP tau* tuning
    fold_boosters, full_booster, u_hat_by_disk, extra_oof = train_and_evaluate_oof_lgbm(
        oof_trees, disk_folds, n_folds=3, seed=42
    )
    for f, booster in enumerate(fold_boosters):
        booster.save_model(os.path.join(ckpt_dir, f"lgb_scorer_fold{f}.txt"))
    full_booster.save_model(os.path.join(scaler_dir, "lgb_scorer.txt"))

    tau_grid = np.linspace(0.05, 0.45, 81)
    tau_curve: List[Dict[str, float]] = []
    best_tau = 0.22
    best_pq = -1.0
    best_metrics: Dict[str, Any] = {}
    tree_by_stem = {t["disk_stem"]: t for t in oof_trees}

    for tau in tau_grid:
        preds_tau: Dict[str, List[str]] = {}
        for t in oof_trees:
            stem = t["disk_stem"]
            sel = solve_tree_dp_antichain(t["nodes"], u_hat_by_disk[stem], float(tau), allow_empty=False)
            preds_tau[stem] = [t["nodes"][i]["rle_counts"] for i in sel]
        m = compute_exact_magfilo_metrics(gt_records, preds_tau)
        tau_curve.append({"tau": float(tau), "pq": m["pq"], "sq": m["sq"], "rq": m["rq"]})
        if m["pq"] > best_pq:
            best_pq = m["pq"]
            best_tau = float(tau)
            best_metrics = m

    tau_hp = min(0.60, best_tau + 0.035)
    preds_hp: Dict[str, List[str]] = {}
    preds_bridged: Dict[str, List[str]] = {}
    preds_raw: Dict[str, List[str]] = {}
    bridge_u_by_disk = extra_oof["bridge_u_by_disk"]

    for t in oof_trees:
        stem = t["disk_stem"]
        preds_raw[stem] = t["raw_baseline_rles"]
        sel_hp = solve_tree_dp_antichain(t["nodes"], u_hat_by_disk[stem], tau_hp, allow_empty=False)
        preds_hp[stem] = [t["nodes"][i]["rle_counts"] for i in sel_hp]
        sel_opt = solve_tree_dp_antichain(t["nodes"], u_hat_by_disk[stem], best_tau, allow_empty=False)
        preds_bridged[stem] = apply_spine_bridging_to_selection(
            t, sel_opt, u_hat_by_disk[stem], bridge_u_by_disk[stem], best_tau
        )

    metrics_hp = compute_exact_magfilo_metrics(gt_records, preds_hp)
    metrics_bridged = compute_exact_magfilo_metrics(gt_records, preds_bridged)
    metrics_raw = compute_exact_magfilo_metrics(gt_records, preds_raw)

    log_msg(
        f"OOF Exact PQ -> Raw Baseline: {metrics_raw['pq']:.4f} | "
        f"PQ-Optimal (tau*={best_tau:.3f}): {best_metrics['pq']:.4f} (SQ={best_metrics['sq']:.4f}, RQ={best_metrics['rq']:.4f}) | "
        f"High-Prec (tau={tau_hp:.3f}): {metrics_hp['pq']:.4f} | Spine-Bridged: {metrics_bridged['pq']:.4f}"
    )

    for snap in qual_snapshots:
        stem = snap["disk_stem"]
        t = tree_by_stem[stem]
        sel = solve_tree_dp_antichain(t["nodes"], u_hat_by_disk[stem], best_tau, allow_empty=False)
        pred_rles = [t["nodes"][i]["rle_counts"] for i in sel]
        snap["pred_rgb"] = _colorize_instances(snap["gray_ds"], pred_rles, 512)
        snap["n_pred"] = len(pred_rles)
        disk_gt = [rec for rec in gt_records if rec["disk_stem"] == stem]
        snap["disk_pq"] = compute_exact_magfilo_metrics(disk_gt, {stem: pred_rles})["pq"]

    save_diagnostic_plots(out_dir, best_metrics, metrics_raw, tau_curve, best_tau, qual_snapshots)

    # Check final test ensemble size dynamically on CPU so total runtime stays < 2.20h
    eval_models_all = [load_fp16_model_for_eval(p, device) for p in ckpt_paths]
    if is_gpu:
        active_test_models = eval_models_all
    else:
        rem_sec = MAX_CPU_RUNTIME_SEC - (time.time() - t_pipeline0) - 180.0
        n_te = max(len(test_files), 1)
        est_per_fold_test = n_te * t_inf_one * 1.12
        max_folds_fit = int(max(1, min(len(eval_models_all), rem_sec // max(est_per_fold_test, 1.0))))
        active_test_models = eval_models_all[:max_folds_fit]
        log_msg(
            f"Test Inference Config (CPU): {len(active_test_models)}-fold {cpu_infer_size}x{cpu_infer_size} "
            f"channels_last (rem_budget={rem_sec:.0f}s)"
        )

    postproc_cfg = {
        "backbone": backbone_name,
        "accelerator": "gpu_t4" if is_gpu else "cpu_4core",
        "infer_size": 1536 if is_gpu else cpu_infer_size,
        "test_ensemble_folds": len(active_test_models),
        "p_eff_weights": {"fg": 0.75, "spine": 0.25, "sep": -0.35},
        "tree_levels": list(TREE_LEVELS),
        "min_component_area": MIN_COMPONENT_AREA,
        "tau_optimal": best_tau,
        "tau_high_precision": tau_hp,
        "bridge_max_gap_px": BRIDGE_MAX_GAP_PX,
        "feature_names": FEATURE_NAMES,
    }
    for cfg_path in [
        os.path.join(out_dir, "postproc_config.json"),
        os.path.join(ckpt_dir, "postproc_config.json"),
    ]:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(postproc_cfg, f, indent=2)

    infer_size = 512 if args.dry_run else (1536 if is_gpu else cpu_infer_size)
    test_stems, sub_reports = run_test_inference_and_write_submissions(
        test_files=test_files,
        active_test_models=active_test_models,
        fold_boosters=fold_boosters,
        out_dir=out_dir,
        device=device,
        is_gpu=is_gpu,
        infer_size=infer_size,
        best_tau=best_tau,
        tau_hp=tau_hp,
    )

    def _strip_arrays(m: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in m.items() if k not in ("nonzero_ious", "nonzero_dices")}

    run_metrics = {
        "accelerator": "gpu_t4" if is_gpu else "cpu_4core",
        "backbone": backbone_name,
        "infer_size": 1536 if is_gpu else cpu_infer_size,
        "folds": 3,
        "epochs_target": epochs,
        "n_train_disks": len(disks),
        "n_gt_annotator_records": len(gt_records),
        "n_test_disks": len(test_stems),
        "runtime_sec": round(time.time() - t_pipeline0, 2),
        "checkpoint_sizes_mb": ckpt_sizes_mb,
        "total_checkpoint_mb": round(total_ckpt_mb, 3),
        "tau_optimal": best_tau,
        "tau_high_precision": tau_hp,
        "oof_metrics": {
            "pq_optimal": _strip_arrays(best_metrics),
            "high_precision": _strip_arrays(metrics_hp),
            "spine_bridged": _strip_arrays(metrics_bridged),
            "raw_baseline_p_eff_040": _strip_arrays(metrics_raw),
        },
        "submissions": sub_reports,
    }
    with open(os.path.join(out_dir, "run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    if not args.dry_run and os.path.exists(tmp_work):
        shutil.rmtree(tmp_work, ignore_errors=True)

    log_msg(f"ALL DONE in {time.time() - t_pipeline0:.1f}s. Submissions: {sub_reports}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGFiLO Panoptic Solar Filament Segmentation Pipeline")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-json", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="/kaggle/working")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--cpu-infer-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
