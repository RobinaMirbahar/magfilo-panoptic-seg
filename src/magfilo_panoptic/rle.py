"""Compressed COCO RLE utilities for 2048x2048 solar filament masks."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pycocotools import mask as mask_util

IMG_SIZE = 2048


def encode_rle_counts(mask_u8: np.ndarray) -> str:
    """Encode a 2D binary mask into compressed Fortran-order COCO RLE counts."""
    rle = mask_util.encode(np.asfortranarray(mask_u8.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def decode_rle_counts(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    """Decode compressed COCO RLE counts string into a uint8 binary mask."""
    return mask_util.decode({"size": [h, w], "counts": counts}).astype(np.uint8)


def rle_dict_from_counts(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> Dict[str, Any]:
    """Build pycocotools-compatible RLE dictionary from compressed counts string."""
    return {
        "size": [h, w],
        "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
    }
