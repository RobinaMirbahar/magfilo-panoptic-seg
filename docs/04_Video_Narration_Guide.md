# MAGFiLO Solar Filament 2026 — Video Narration & Kaggle Writeup Guide

## Timestamped Voice-Over Script (`03_Explainer_Video_MAGFiLO_Solar_Filament_1080p.mp4`, 10.0s @ 15 fps)

- **00:00 – 00:02.8 (Milestone 1 — The Solar Limb & Sunspot Trap):**
  "On 2048-by-2048 GONG H-alpha solar disks, radial limb darkening and circular sunspot umbrae cause standard connected-component thresholding at 0.40 to generate 7,855 false positives across 707 unique disks, stalling Panoptic Quality at 0.2038."
- **00:02.8 – 00:05.8 (Milestone 2 — Stage P Physics + Stage N 3-Head UNet):**
  "MAGFiLO first fits the solar limb via a 72-spoke gradient scan and Kasa circle fit, divides out a 128-bin Center-to-Limb Variation profile, and trains a compact 22.61-megabyte ResNet-18 panoptic U-Net with soft-clDice skeleton supervision to predict foreground, medial spine, and boundary separation heads."
- **00:05.8 – 00:08.4 (Milestone 3 — Sunspot Gate & Exact Bottom-Up Tree-DP):**
  "Next, an astrophysical sunspot gate rejects dark circular umbrae lacking ribbon spines, while a 31-feature LightGBM IoU regressor and exact linear-time Tree Dynamic Programming over a 7-level laminar component tree boost out-of-fold Panoptic Quality by 59.5 percent to 0.3252, slashing false positives by 64.6 percent."
- **00:08.4 – 00:10.0 (Milestone 4 — Calibrated Deployment):**
  "Raising tau to 0.240 further trims false positives to 2,390 at 0.6306 Segmentation Quality, producing 1,011 disjoint filament masks across all 180 held-out test disks."

## How to Attach Both Papers + Video Inside Your Single Kaggle Writeup
1. Open the **Solar Filament Segmentation Challenge 2026** Writeup editor (`https://www.kaggle.com/competitions/filament-segmentation-2026/writeups`) **before `2026-11-16 11:59:59 UTC`**.
2. Keep **one single official Writeup** (Kaggle allows one active writeup per entrant/team).
3. Under **Media Gallery**, upload `03_Explainer_Video_MAGFiLO_Solar_Filament_1080p.mp4` (and optionally the comic preview sheet PNGs as cover thumbnails).
4. Under **Project Files / Attachments**, attach **both PDFs**:
   - `01_Scientific_Paper_MAGFiLO_Solar_Filament_2026.pdf` *(Primary Formal Scientific Report)*
   - `02_Visual_Comic_Guide_MAGFiLO_Solar_Filament_2026.pdf` *(Illustrated Visual Graphic-Story Companion)*
