"""Exact MAGFiLO multi-annotator Panoptic Quality (PQ) and rubric metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools import mask as mask_util

from .rle import IMG_SIZE, rle_dict_from_counts


def compute_exact_magfilo_metrics(
    gt_records: List[Dict[str, Any]],
    pred_by_disk: Dict[str, List[str]],
    img_h: int = IMG_SIZE,
    img_w: int = IMG_SIZE,
) -> Dict[str, Any]:
    """
    Compute exact MAGFiLO multi-annotator Panoptic Quality (PQ = SQ * RQ),
    pairwise positive IoU / Dice distributions (IoU > 0, Dice > 0), and 1:n / n:1
    matching cardinality statistics across all annotator records.
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
