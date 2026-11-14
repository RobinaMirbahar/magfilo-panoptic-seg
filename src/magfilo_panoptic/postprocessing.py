"""Stage H: Component-tree candidate generation, LightGBM PQ scorer, and Tree DP."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import lightgbm as lgb
import numpy as np
from pycocotools import mask as mask_util
from scipy.spatial import cKDTree

from .rle import IMG_SIZE, decode_rle_counts, encode_rle_counts, rle_dict_from_counts

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
    Build laminar component tree over superlevels of P_eff, compute physics features,
    multi-annotator expected PQ contribution u_gt(c), and candidate spine bridges.
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
    Exact bottom-up tree DP selecting a pairwise-disjoint antichain maximizing
    sum_{c in S} (u_hat(c) - tau). Guarantees >= 1 mask when allow_empty=False.
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
    Apply improving spine-tangent corridor bridges over the DP-selected antichain,
    mapping each base node cleanly through exact match, ancestor, or unique
    selected descendant, and preserving strict pairwise mask disjointness.
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
    """Train 3-fold date-grouped LightGBM regressors predicting expected PQ u_hat(c)."""
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
