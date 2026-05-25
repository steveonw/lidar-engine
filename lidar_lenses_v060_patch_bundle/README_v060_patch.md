# lidar_lenses_wave v0.6.0 — tested preset/material patch

This patch folds in the sweep results from the v0.5.x testing cycle.

## What changed

- Promoted tested indoor settings:
  - `edge_anti_min = 0.38`
  - `adaptive_edge_percentile = 94`
  - `carrier_mode = "ensemble"`
  - `stack = 4`
- Kept anti-light / anti-sound as the main indoor structure signal.
- Kept overhead as a non-wave layout/map preset.
- Added explicit `MATERIAL_PROFILES` so material targets are not only shape-based.
- Updated default acoustic / ultrasonic / polarization functions to use `piece_type` profiles first.
- Updated material classification:
  - `hard_smooth_ult_min` default raised to `0.85`
  - `soft_material` can now trigger for cloth/leather-style high texture values
  - `soft_material_ultrasonic_max` added to avoid sweeping wood/stone/metal into soft labels
- Preset diagnostics now record material classifier thresholds.

## Recommended use

Use this as File A in the upload-A+B Colab sweep notebook.

File A:
`lidar_lenses_wave_v060.py`

File B:
any compatible sweep spreadsheet, such as `lidar_wave_material_label_sweep_plan_v059.xlsx`.

## Stable mental model

- `overhead_layout` = layout/map truth
- `indoor_structure` = anti-light + anti-sound indoor structure
- `outdoor_occlusion` = partial occluder / mixed-depth stress testing
- `material_scan` = acoustic/ultrasonic/polarization material inspection
- `full_diagnostic` = all-channel debugging sheet

The classifier is still a hint layer. The raw channels are the main signal.

- `material_scan` uses `material_priority=True`, so material labels win over edge/occluder labels on target boards.
