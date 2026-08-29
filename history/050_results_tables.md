# 050: Results tables (D-ViT 2411.07167 p.6 format; vitl-320 / vitt-256 / hg0-256 / vitt-096 / hg0-096)

Created 2026-08-29 (user request: tables in the same form as page 6 of docs/2411.07167v2.pdf (D-ViT's Table 1 / 2), covering
the evaluation on each dataset together with yaw / pitch / roll metrics). All values come from this repository's own instruments
(`hrffa.train.evaluate`).

## 0. How to read these tables (required reading)

- **Absolute values are fit measures**: since 2026-08-26 every model is trained on all splits of the real-image datasets (including
  WFLW test / 300W / COFW test) (036 §12, 042 §0). The official numbers in these tables therefore measure the fit to the training
  distribution and **must not be compared with published benchmark results (e.g. D-ViT's Table 1 / 2)**; only the table format is
  borrowed. **The `D-ViT (paper)` rows in Tables 1 / 2 are the published values of "Ours" in 2411.07167 v2 p.6 (unleaked test
  benchmarks), listed side by side at the user's request (2026-08-29). Their conditions differ from this repository's rows (fit
  measures), so no ranking is drawn from the numbers.** There are no D-ViT values corresponding to Tables 3–6.
- The only unleaked instrument is 300wlp_val (head-NME on a held-out 300W-LP subset): vitl-320 0.0033 / vitt-256 0.0071 /
  hg0-256 0.0158 / vitt-096 0.0100 / hg0-096 0.0185 (047 / 048).
- **Yaw / pitch / roll**: these models do not output head pose (the pose head is untrained and excluded from the export, 047 §2.2),
  so pose robustness is expressed as **landmark accuracy per pose**, not as a pose-angle error. (a) Table 3: inter-ocular NME per
  effective pose bin of pose-stress (6DRepNet-estimated pose plus the applied rotation); (b) Table 4: NME per in-plane rotation
  (roll 0–315°) = Roll360 equivariance; (c) Table 5: NME under projective camera perturbations of pitch ±15/±25° and yaw ±15°;
  (d) Table 6: head-NME (normalized by the crop side) of the three real sets stratified by 6DRepNet-estimated pose.
- pose-stress (024) uses real images with exact ground-truth geometric transforms, so the degradation under geometric transforms is
  interpretable even with the leak (042 §0). The image-quality perturbations (style-shift, Table 7) were added at the user's request
  on 2026-08-29. They are applied deterministically to the rendered crop (mblur9/21 = motion blur of 9/21 px along 30°, warm/cool =
  channel gains, gamma0.6/1.6, gray, jpeg30 = re-compression at JPEG quality 30). The absolute values are fit measures (see above);
  the degradation rates in parentheses are read as "robustness relative to the training distribution" (042 §0: the more accurate
  the model, the larger the relative degradation under strong blur). **The perturbations are applied in crop pixels**, so their
  strength changes with the input resolution: for the 96 models (vitt-096 / hg0-096), mblur9/21 is 9/21 px on a 96 px crop (24/56 px in 256 terms) and the
  JPEG blocks are relatively larger as well. Do not read the Table 7 rows of the 96 models side by side with the 256 models.

## 1. Models

| Model | Configuration | Input | Params | GMACs / GFLOPs | CPU ms |
|---|---|---|---|---|---|
| vitl-320 | DINOv3 ViT-L/16 teacher + point-query decoder (clean_v3, 047) | 320 | 308.2M | 131.3 / 262.6 | 520 |
| vitt-256 | ViT-T/16 student distilled from teacher clean_v3 (student_s256_96gb_r2, 048 §1) | 256 | 9.0M | 2.05 / 4.09 | 12.4 |
| hg0-256 | PP-HGNetV2-B0 stage0-2 + FPN + small-decoder student (student_hg0_wsd, 048 §2, 049) | 256 | 1.63M | 0.70 / 1.40 | 5.2 |
| vitt-096 | ViT-T/16 student at 96×96, fine-tuned from vitt-256 (r2 e449) (student_s096_96gb_r2, 048 §3) | 96 | 9.0M | 0.43 / 0.85 | 4.6 |
| hg0-096 | PP-HGNetV2-B0 stage0-2 + FPN + small-decoder student at 96×96, fine-tuned from hg0-256 (e386) (student_hg0_s096_wsd, 048 §4) | 96 | 1.63M | 0.13 / 0.26 | 1.5 |

GMACs / GFLOPs are measured with `torch.utils.flop_counter.FlopCounterMode` (batch 1, whole model = backbone + decoder + heads,
FLOPs = 2 × MACs). Because the CPU flash implementation of SDPA is not counted, SDPA is replaced by an explicit matmul during the
measurement only, so the QKᵀ / AV products of the attention are included (they account for 16 GFLOPs of vitl-320 and 0.7 GFLOPs of
vitt-256; the 0.68 GMACs of hg0 in 049 excludes the decoder attention, 0.70 with it).
hg0-096 (e134, added 2026-08-29): 0.128 / 0.257 measured the same way (hg0-256 re-measured alongside as 0.700 / 1.400 to check the method).
Its 1.5 ms was measured in a session in which vitt-096 / hg0-256 gave 3.9 / 4.4 ms (their table values 4.6 / 5.2 come from their own
acceptance sessions), i.e. the session-to-session spread of the latency instrument is about 15 %.

All instruments are run with `--use-ema --official / --stratify-real / --pose-stress --stress-n 300 / --style-shift --style-n 300`.
Logs: `runs/<run>/eval_best_{official,stratreal,stress,style}.log` (vitl-320 was re-run on the 8 GB machine on 2026-08-29; the
values match 047).

## 2. Tables

### Table 1. NME (%, inter-ocular, lower is better)
| Model | WFLW Full | Pose | Expr. | Illum. | Makeup | Occl. | Blur | COFW Full | 300W Full | Comm. | Chal. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D-ViT (paper) | 3.75 | 6.43 | 3.85 | 4.06 | 3.57 | 4.47 | 4.37 | 4.13 | 2.85 | 2.43 | 4.56 |
| vitl-320 | 1.51 | 2.54 | 1.55 | 1.47 | 1.42 | 1.56 | 1.58 | 1.21 | 1.03 | 0.96 | 1.34 |
| vitt-256 | 3.36 | 5.64 | 3.43 | 3.25 | 3.19 | 3.56 | 3.60 | 2.76 | 2.66 | 2.36 | 3.90 |
| hg0-256 | 5.32 | 9.53 | 5.54 | 5.16 | 5.09 | 6.29 | 6.04 | 3.77 | 3.87 | 3.37 | 5.96 |
| vitt-096 | 5.14 | 8.65 | 5.32 | 4.96 | 5.06 | 5.52 | 5.40 | 3.90 | 3.85 | 3.57 | 5.02 |
| hg0-096 | 7.46 | 14.71 | 7.77 | 7.19 | 8.10 | 8.91 | 8.16 | 4.99 | 4.77 | 4.24 | 6.91 |

### Table 2. FR10 (%, lower is better) and AUC10 (%, higher is better) on WFLW
| Model | FR10 Full | Pose | Exp. | Ill. | Mu. | Occ. | Blur | AUC10 Full | Pose | Exp. | Ill. | Mu. | Occ. | Blur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D-ViT (paper) | 1.76 | 8.28 | 1.27 | 1.29 | 1.94 | 3.80 | 2.07 | 63.7 | 40.1 | 62.6 | 64.7 | 64.7 | 57.1 | 58.6 |
| vitl-320 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 84.9 | 74.5 | 84.5 | 85.4 | 85.8 | 84.4 | 84.2 |
| vitt-256 | 0.28 | 1.53 | 0.32 | 0.00 | 0.00 | 0.54 | 0.52 | 66.5 | 43.9 | 65.7 | 67.5 | 68.2 | 64.5 | 64.1 |
| hg0-256 | 7.00 | 32.21 | 5.73 | 5.44 | 6.31 | 13.04 | 9.44 | 49.2 | 17.5 | 46.7 | 50.3 | 50.2 | 42.4 | 42.8 |
| vitt-096 | 4.20 | 23.31 | 4.78 | 2.72 | 2.91 | 5.57 | 3.75 | 49.7 | 20.2 | 47.7 | 51.2 | 49.9 | 46.2 | 47.1 |
| hg0-096 | 17.88 | 67.48 | 18.79 | 14.18 | 20.87 | 26.09 | 22.38 | 36.1 | 5.8 | 31.9 | 37.6 | 34.4 | 28.7 | 30.0 |

### Table 3. NME (%) by yaw / pitch bin (effective pose bins of pose-stress)
| Model | Yaw 0–30 | Yaw 30–60 | Yaw 60–95 | Pitch −95..−45 | Pitch −45..−15 | Pitch −15..15 | Pitch 15..45 | Pitch 45..95 |
|---|---|---|---|---|---|---|---|---|
| vitl-320 | 1.26 | 1.71 | 2.04 | 1.76 | 1.37 | 1.30 | 1.29 | 1.85 |
| vitt-256 | 2.94 | 4.10 | 4.68 | 4.07 | 3.20 | 3.06 | 2.99 | 4.43 |
| hg0-256 | 4.21 | 6.35 | 8.14 | 6.26 | 4.63 | 4.46 | 4.35 | 7.51 |
| vitt-096 | 4.49 | 6.21 | 7.37 | 5.93 | 4.82 | 4.77 | 4.54 | 7.06 |
| hg0-096 | 5.69 | 8.70 | 10.85 | 8.02 | 6.17 | 6.21 | 5.79 | 10.07 |
| n | 5933 | 1124 | 143 | 223 | 2503 | 2944 | 1398 | 48 |

### Table 4. NME (%) by roll (pose-stress, n=300 per set)
| Model | Set | Roll 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 | worst−base |
|---|---|---|---|---|---|---|---|---|---|---|
| vitl-320 | wflw | 1.61 | 1.61 | 1.64 | 1.60 | 1.63 | 1.61 | 1.63 | 1.60 | +0.02 |
| vitl-320 | 300w | 1.12 | 1.11 | 1.14 | 1.11 | 1.13 | 1.11 | 1.13 | 1.11 | +0.01 |
| vitl-320 | cofw | 1.29 | 1.30 | 1.31 | 1.30 | 1.30 | 1.30 | 1.30 | 1.29 | +0.02 |
| vitt-256 | wflw | 3.67 | 3.61 | 3.67 | 3.60 | 3.70 | 3.61 | 3.68 | 3.61 | +0.03 |
| vitt-256 | 300w | 2.93 | 2.88 | 2.89 | 2.89 | 2.95 | 2.92 | 2.96 | 2.90 | +0.03 |
| vitt-256 | cofw | 3.01 | 2.96 | 3.02 | 2.97 | 2.98 | 2.96 | 2.98 | 2.97 | +0.01 |
| hg0-256 | wflw | 5.96 | 5.95 | 5.90 | 6.06 | 5.92 | 6.03 | 5.98 | 5.96 | +0.10 |
| hg0-256 | 300w | 4.14 | 4.11 | 4.26 | 4.20 | 4.26 | 4.29 | 4.21 | 4.16 | +0.15 |
| hg0-256 | cofw | 4.12 | 4.08 | 4.06 | 4.06 | 4.03 | 4.11 | 4.12 | 4.06 | +0.00 |
| vitt-096 | wflw | 6.30 | 6.05 | 6.17 | 5.98 | 6.30 | 6.26 | 6.40 | 6.19 | +0.10 |
| vitt-096 | 300w | 4.58 | 4.41 | 4.55 | 4.50 | 5.03 | 5.11 | 5.14 | 4.71 | +0.57 |
| vitt-096 | cofw | 4.74 | 4.61 | 4.72 | 4.49 | 4.73 | 4.49 | 4.56 | 4.55 | +0.00 |
| hg0-096 | wflw | 8.89 | 9.03 | 9.09 | 9.21 | 8.98 | 9.33 | 9.90 | 8.72 | +1.01 |
| hg0-096 | 300w | 5.57 | 5.61 | 5.82 | 6.07 | 6.40 | 6.59 | 6.05 | 5.95 | +1.03 |
| hg0-096 | cofw | 5.84 | 5.79 | 6.05 | 5.71 | 5.76 | 5.51 | 5.68 | 5.73 | +0.21 |

### Table 5. NME (%) under camera pitch / yaw perturbation (pose-stress, n=300 per set)
| Model | Set | base | cam pitch −25 | −15 | +15 | +25 | cam yaw −15 | +15 | p+25 y+15 | p−25 y−15 |
|---|---|---|---|---|---|---|---|---|---|---|
| vitl-320 | wflw | 1.61 | 1.78 | 1.63 | 1.58 | 1.60 | 1.55 | 1.55 | 1.56 | 1.78 |
| vitl-320 | 300w | 1.12 | 1.25 | 1.15 | 1.09 | 1.10 | 1.08 | 1.08 | 1.08 | 1.22 |
| vitl-320 | cofw | 1.29 | 1.36 | 1.29 | 1.26 | 1.29 | 1.23 | 1.24 | 1.25 | 1.35 |
| vitt-256 | wflw | 3.67 | 3.84 | 3.67 | 3.52 | 3.48 | 3.52 | 3.50 | 3.44 | 3.94 |
| vitt-256 | 300w | 2.93 | 3.07 | 2.93 | 2.81 | 2.75 | 2.78 | 2.81 | 2.73 | 3.03 |
| vitt-256 | cofw | 3.01 | 3.12 | 3.01 | 2.95 | 2.92 | 2.90 | 2.90 | 2.92 | 3.12 |
| hg0-256 | wflw | 5.96 | 5.99 | 5.71 | 5.71 | 5.38 | 5.55 | 5.60 | 5.31 | 6.08 |
| hg0-256 | 300w | 4.14 | 4.46 | 4.24 | 4.05 | 3.98 | 3.97 | 4.04 | 3.90 | 4.39 |
| hg0-256 | cofw | 4.12 | 4.27 | 4.11 | 4.02 | 3.96 | 3.98 | 3.97 | 3.99 | 4.23 |
| vitt-096 | wflw | 6.30 | 5.94 | 5.83 | 5.97 | 5.51 | 5.78 | 5.76 | 5.30 | 6.05 |
| vitt-096 | 300w | 4.58 | 4.49 | 4.39 | 4.41 | 4.10 | 4.24 | 4.34 | 4.06 | 4.41 |
| vitt-096 | cofw | 4.74 | 4.53 | 4.49 | 4.45 | 4.18 | 4.33 | 4.42 | 4.20 | 4.47 |
| hg0-096 | wflw | 8.89 | 8.20 | 8.27 | 8.44 | 7.72 | 8.37 | 8.19 | 7.35 | 8.34 |
| hg0-096 | 300w | 5.57 | 5.38 | 5.38 | 5.45 | 4.94 | 5.16 | 5.25 | 4.81 | 5.34 |
| hg0-096 | cofw | 5.84 | 5.62 | 5.57 | 5.67 | 5.28 | 5.35 | 5.53 | 5.13 | 5.48 |

### Table 6. head-NME (×100) stratified by 6DRepNet-estimated pose (3,696 real images from the three sets)
| Model | mean | Yaw 0–30 | Yaw 30–60 | Yaw 60–95 | Pitch −90..−30 | Pitch −30..−10 | Pitch −10..10 | Pitch 10..30 | Pitch 30..90 |
|---|---|---|---|---|---|---|---|---|---|
| vitl-320 | 0.37 | 0.36 | 0.42 | 0.45 | 0.42 | 0.37 | 0.36 | 0.40 | 0.50 |
| vitt-256 | 0.85 | 0.81 | 0.98 | 1.01 | 0.98 | 0.85 | 0.82 | 0.93 | 1.13 |
| hg0-256 | 1.29 | 1.21 | 1.60 | 1.85 | 1.71 | 1.28 | 1.21 | 1.50 | 2.09 |
| vitt-096 | 1.27 | 1.22 | 1.46 | 1.58 | 1.48 | 1.27 | 1.22 | 1.38 | 1.71 |
| hg0-096 | 1.75 | 1.60 | 2.22 | 2.85 | 2.53 | 1.76 | 1.59 | 2.04 | 2.94 |
| n | | 2939 | 672 | 85 | 143 | 1076 | 2136 | 232 | 32 |

### Table 7. style-shift: NME (%) per image-quality perturbation (degradation vs clean in parentheses, n=300 per set)
| Model | Set | clean | mblur9 | mblur21 | warm | cool | gamma0.6 | gamma1.6 | gray | jpeg30 |
|---|---|---|---|---|---|---|---|---|---|---|
| vitl-320 | wflw_test | 1.49 | 1.57 (+5.5%) | 2.09 (+40.4%) | 1.51 (+1.4%) | 1.51 (+1.5%) | 1.50 (+1.1%) | 1.51 (+1.3%) | 1.60 (+7.8%) | 1.55 (+4.5%) |
| vitl-320 | 300w_valid | 1.04 | 1.13 (+8.6%) | 1.46 (+40.6%) | 1.06 (+2.0%) | 1.05 (+1.2%) | 1.06 (+1.8%) | 1.05 (+1.4%) | 1.09 (+4.9%) | 1.08 (+4.3%) |
| vitl-320 | cofw_test | 1.21 | 1.27 (+5.3%) | 1.60 (+32.3%) | 1.22 (+1.3%) | 1.23 (+1.5%) | 1.22 (+1.1%) | 1.22 (+1.2%) | 1.27 (+5.0%) | 1.25 (+3.6%) |
| vitt-256 | wflw_test | 3.26 | 3.32 (+1.8%) | 4.60 (+41.1%) | 3.32 (+1.9%) | 3.33 (+2.1%) | 3.35 (+2.7%) | 3.36 (+3.0%) | 3.48 (+6.6%) | 3.36 (+3.1%) |
| vitt-256 | 300w_valid | 2.65 | 2.73 (+2.6%) | 3.66 (+37.7%) | 2.68 (+1.1%) | 2.69 (+1.3%) | 2.71 (+2.0%) | 2.72 (+2.3%) | 2.75 (+3.7%) | 2.73 (+2.9%) |
| vitt-256 | cofw_test | 2.76 | 2.82 (+2.1%) | 3.51 (+27.2%) | 2.79 (+1.0%) | 2.79 (+1.1%) | 2.83 (+2.4%) | 2.81 (+1.7%) | 2.85 (+3.3%) | 2.81 (+1.8%) |
| hg0-256 | wflw_test | 5.08 | 5.28 (+3.9%) | 8.16 (+60.5%) | 5.14 (+1.2%) | 5.23 (+2.9%) | 5.13 (+0.9%) | 5.29 (+4.2%) | 5.41 (+6.4%) | 5.35 (+5.2%) |
| hg0-256 | 300w_valid | 3.85 | 3.88 (+0.8%) | 5.20 (+35.0%) | 3.90 (+1.2%) | 3.90 (+1.3%) | 3.88 (+0.7%) | 3.92 (+1.7%) | 3.95 (+2.5%) | 3.90 (+1.0%) |
| hg0-256 | cofw_test | 3.79 | 3.90 (+2.9%) | 4.93 (+30.0%) | 3.83 (+1.1%) | 3.83 (+0.9%) | 3.82 (+0.8%) | 3.82 (+0.8%) | 3.91 (+3.2%) | 3.85 (+1.5%) |
| vitt-096 | wflw_test | 5.01 | 6.26 (+25.0%) | 24.80 (+395.2%) | 5.10 (+1.9%) | 5.09 (+1.6%) | 5.13 (+2.4%) | 5.21 (+4.1%) | 5.37 (+7.3%) | 5.36 (+7.0%) |
| vitt-096 | 300w_valid | 3.82 | 4.42 (+15.8%) | 18.31 (+379.4%) | 3.86 (+1.1%) | 3.86 (+1.1%) | 3.90 (+2.1%) | 3.94 (+3.1%) | 3.96 (+3.8%) | 4.04 (+5.8%) |
| vitt-096 | cofw_test | 3.89 | 4.52 (+16.3%) | 15.95 (+310.4%) | 3.95 (+1.8%) | 3.95 (+1.6%) | 3.99 (+2.8%) | 4.00 (+2.8%) | 4.05 (+4.2%) | 4.08 (+5.0%) |
| hg0-096 | wflw_test | 7.01 | 9.18 (+31.0%) | 32.00 (+356.7%) | 7.15 (+2.0%) | 7.30 (+4.2%) | 7.11 (+1.5%) | 7.36 (+5.0%) | 7.77 (+10.9%) | 9.10 (+29.9%) |
| hg0-096 | 300w_valid | 4.64 | 5.68 (+22.5%) | 19.79 (+326.5%) | 4.68 (+1.0%) | 4.74 (+2.1%) | 4.70 (+1.2%) | 4.72 (+1.7%) | 4.80 (+3.5%) | 5.60 (+20.8%) |
| hg0-096 | cofw_test | 5.03 | 5.71 (+13.7%) | 18.03 (+258.9%) | 5.07 (+0.8%) | 5.15 (+2.4%) | 5.04 (+0.3%) | 5.16 (+2.6%) | 5.19 (+3.2%) | 5.74 (+14.2%) |


## 3. Additional notes

- Tables 1 / 2 have the same columns as D-ViT's Table 1 / 2 (the 7 WFLW subsets, COFW, the 3 300W subsets / FR10 and AUC10 on WFLW).
- The n of Tables 3–5 comes from 3 sets (wflw / 300w / cofw, 300 images each) × 16 configurations = 14,400 evaluations re-binned by
  effective pose. Roll (Table 4) is a geometrically exact equivariance check: the smaller worst−base, the more Roll360-equivariant
  the model (hg0-096 is the least equivariant: +1.01 / +1.03 / +0.21; every other model stays within +0.6). Under the camera perturbations (Table 5), tilting the pitch upward (+) tends to lower the NME for every model (the
  view from below the chin is favorable); for the 256 models, −25° and the −25/−15 combination are the worst.
- In Table 7, mblur21 is the worst perturbation for every model (+30 to +60 % for the 256 models, +260 to +395 % for the 96 models, see
  §0), while color temperature, gray and jpeg30 stay at +1 to +7 % — except hg0-096, whose gray (+3 to +11 %) and jpeg30
  (+14 to +30 %) degradations are clearly larger than those of the other models. The photometric augmentations on the training side (JPEG
  quality 35–85 with probability 0.3, noise, blur, grayscale, brightness / gamma) keep them in-distribution. JPEG is a single
  point at quality 30 (no sweep).
- The head-NME of Table 6 is normalized by the crop side (pad 0.05, side = head bbox × 1.1) and is therefore in different units from
  inter-ocular NME % (roughly inter-ocular ≈ head-NME × 3–4).
- hg0-096 (CNN @96) vs vitt-096 (ViT-T @96) at the same input size: WFLW 7.46 vs 5.14, WFLW pose 14.71 vs 8.65, 300W 4.77 vs 3.86,
  COFW 4.99 vs 3.90, 300wlp_val 0.0185 vs 0.0100 and stratify-real |yaw| 60–95 2.85 vs 1.58. The CNN buys a 2.6× lower latency
  (1.5 vs 3.9 ms in the same session) with a large accuracy loss; it fails all three deployment criteria of 043 §4 (048 §4).

Generator: `scripts/make_results_tables.py` (it only reads eval_best_*.log; `uv run python scripts/make_results_tables.py > tables.md`).
