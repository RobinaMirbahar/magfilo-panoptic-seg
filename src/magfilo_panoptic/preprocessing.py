"""Solar-disk geometry estimation and CLV flat-fielding (Stage P)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .rle import IMG_SIZE


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
    Estimate solar-disk center and radius (cx, cy, R) on an H-alpha image.

    Uses 4x downsampled Otsu thresholding, largest connected component,
    moment initialization, 72-ray radial 50%-limb crossing refinement, MAD
    prominence outlier rejection, and Kasa least-squares circle fitting.
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

    Vectorized implementation (~50-90 ms per 2048x2048 image):
      - Single-argsort radial median CLV profile (128 bins on 256x256)
      - Bilinear CLV surface interpolation via 512x512 grid
      - Normalized-convolution B_sigma48 background estimation on 512x512
      - Heliocentric cosine mu = sqrt(max(0, 1 - rho^2)) inside r <= 0.995 R
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

    # Full-resolution off-limb mask and mu via squared radial distance (avoids full-res sqrt of r)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    rho2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (R * R)
    disk_mask = rho2 <= (0.995 * 0.995)

    raw_f = gray_u8.astype(np.float32)
    norm_clv = np.where(disk_mask, raw_f / clv_bg, 0.0).astype(np.float32)

    # Local chromospheric absorption background B_sigma48 via normalized convolution at 512x512
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
