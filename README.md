# MAGFiLO Panoptic Solar Filament Segmentation

Three-stage astrophysical preprocessing, topology-aware multi-head U-Net (`MagfiloPanopticUNet`), and component-tree dynamic programming pipeline for full-disk GONG H-alpha solar filament panoptic segmentation (**Solar Filament Segmentation 2026**).

## Methodological Foundations

1. **MAGFiLO Dataset & Multi-Annotator Spine Protocol**
   * Ahmadzadeh et al., *"MAGFiLO: Manually Annotated GONG Filaments from H-alpha Observations"*, *Scientific Data* (2024), doi:[10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y).
2. **Centerline Topology Preservation (`soft-clDice`)**
   * Shit et al., *"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"*, *CVPR* 2021.
3. **Panoptic Quality (`PQ = SQ * RQ`)**
   * Kirillov et al., *"Panoptic Segmentation"*, *CVPR* 2019.

## Architecture Overview

* **Stage 0 (`train_magfilo_panoptic.py`)**: Groups 1,924 multi-annotator records by unique physical disk (`707` train disks) and `YYYYMMDD` observation date into 3 zero-leakage CV folds. Synthesizes soft consensus foreground (`y_fg in [0, 1]`), spine centerline (`y_spine` combining explicit `annotation["spine"]` and medial-axis skeletons), and same-annotator 1-8 px gap corridor (`y_sep in {0, 1}`).
* **Stage P (`src/magfilo_panoptic/preprocessing.py`)**: Regresses solar disk geometry `(cx, cy, R)` via 72-ray 50%-limb crossing and Kasa circle fit, computes 128-bin radial median Center-to-Limb Variation (`norm_CLV`), masks off-limb pixels (`r > 0.995 R`), estimates normalized-convolution local absorption contrast `(B_sigma48 - F) / B_sigma48`, and includes heliocentric cosine `mu`.
* **Stage N (`src/magfilo_panoptic/network.py`)**: 3-head (`fg`, `spine`, `sep`) `MagfiloPanopticUNet` with scSE attention, PPM bottleneck, stride-1 detail stem, boundary-upweighted `BCE + SoftDice` (`3x` near `y_sep`), and `0.25 * soft_clDice`. Exports float16 weights (`< 30 MB`/file).
* **Stage H (`src/magfilo_panoptic/postprocessing.py`)**: Builds an effective map `P_eff = clip(0.75 P_fg + 0.25 P_spine - 0.35 P_sep, 0, 1)`, extracts a 7-level laminar component tree (`area >= 120 px`), filters compact umbral sunspots via astrophysical shape/darkness features, predicts per-candidate expected PQ contribution `u_hat(c)` with date-grouped LightGBM, solves exact bottom-up Tree DP over pairwise-disjoint antichains maximizing `sum(u_hat(c) - tau)`, and optionally bridges collinear dark H-alpha fragments along spine tangents (`gap <= 14 px`).

## Repository Structure

```text
magfilo-panoptic-seg/
├── .gitignore
├── LICENSE                    # MIT License (Robina Mirbahar)
├── README.md
├── pipeline.ipynb             # End-to-end verification, inference & evaluation notebook
├── requirements.txt
├── train_magfilo_panoptic.py  # Full training, validation & --predict-only inference CLI
├── src/
│   └── magfilo_panoptic/
│       ├── __init__.py
│       ├── metrics.py         # Exact multi-annotator PQ, SQ, RQ, 1:n / n:1 evaluation
│       ├── network.py         # MagfiloPanopticUNet, SolarMultiTaskLoss, soft-clDice, inference
│       ├── postprocessing.py  # Component tree, LightGBM u_hat scorer, Tree DP, bridging
│       ├── preprocessing.py   # Solar disk circle fit, CLV flat-fielding, contrast, mu
│       └── rle.py             # Fortran-order 2048x2048 COCO RLE codec
└── weights/                   # Trained fp16 UNet (<30 MB/file), LightGBM scorers, postproc_config.json
```

## Reproducing Locally

```bash
# 1. Verify stored weights end-to-end without retraining (synthetic 2048x2048 test disks)
python train_magfilo_panoptic.py --predict-only --weights-dir ./weights --dry-run --out-dir ./dry_run_out

# 2. Reproduce competition test predictions directly from stored weights/ (no retraining)
python train_magfilo_panoptic.py --predict-only --weights-dir ./weights --data-root /path/to/filament-segmentation-2026 --out-dir ./out

# 3. Full 3-fold training, OOF calibration & test inference from scratch
python train_magfilo_panoptic.py --data-root /path/to/filament-segmentation-2026 --out-dir ./out
```

## Results

Measured out-of-fold (OOF) validation on Kaggle NVIDIA Tesla T4 (`1536x1536`, 3 observation-date-grouped folds across `707` solar disks / `1,154` multi-annotator records, `18` epochs/fold, 4-view flip TTA, total runtime `11,864.3s`):

| Method | PQ | SQ | RQ | TP / FP / FN | IoU (mean / p50) | Dice (mean / p50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw `P_eff >= 0.40` baseline | `0.2038` | `0.6152` | `0.3313` | `3187 / 7855 / 5012` | `0.4330 / 0.4453` | `0.5801 / 0.6162` |
| Tree DP (`tau* = 0.205`, `pq_optimal`) | `0.3252` | `0.6295` | `0.5166` | `3824 / 2782 / 4375` | `0.5163 / 0.5508` | `0.6611 / 0.7104` |
| Tree DP (`tau = 0.240`, `high_precision`) | `0.3231` | `0.6306` | `0.5124` | `3647 / 2390 / 4552` | `0.5210 / 0.5561` | `0.6655 / 0.7147` |
| Tree DP (`spine_bridged`) | `0.3252` | `0.6295` | `0.5166` | `3824 / 2782 / 4375` | `0.5163 / 0.5508` | `0.6611 / 0.7104` |

- **Gain over threshold baseline**: `+0.1214` absolute PQ (`+59.5%` relative) with `-64.6%` fewer false positives (`7,855 -> 2,782`).
- **OOF quartiles (`pq_optimal`)**: IoU `p25/p50/p75 = 0.4181 / 0.5508 / 0.6471`; Dice `p25/p50/p75 = 0.5896 / 0.7104 / 0.7858`.
- **OOF fragmentation/merging (`pq_optimal`)**: GT matching degrees `{0: 2321, 1: 5597, 2: 247, 3: 33, 4: 1}` (`281` `1->many` splits); predicted matching degrees `{0: 652, 1: 5734, 2: 197, 3: 22}` (`219` `many->1` merges).
- **Live Kaggle Public LB (`submission_pq_optimal.csv`, ref `58343725`)**: `0.30` (`1,011` predicted filament instances across `180/180` test solar disks).
- **Compact `float16` checkpoints**: `unet_fold0_fp16.pt`, `unet_fold1_fp16.pt`, `unet_fold2_fp16.pt` (`22.61 MB` each, `67.82 MB` total) stored in `weights/` alongside `lgb_scorer_fold{0,1,2}.txt`, `lgb_scorer.txt`, and `postproc_config.json`.
- **Outputs (`./out/`)**: `run_metrics.json`, `quantitative_rubric_histograms.png`, `qualitative_morphology_panels.png`, `submission_pq_optimal.csv` (`1,011` rows / `180` test disks), `submission_high_precision.csv` (`927` rows / `180` test disks), and `submission_spine_bridged.csv` (`1,011` rows / `180` test disks).
