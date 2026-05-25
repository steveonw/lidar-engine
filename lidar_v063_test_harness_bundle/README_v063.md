# LiDAR/Wave v0.6.3 Test Harness

This bundle patches v0.6.1 in two ways:

1. **Material-label classifier rules**
   - Adds `metal_or_glass`, `stone_hard`, and `wood_material` labels.
   - Keeps `soft_material`, `foliage`, `hard_smooth`, `solid_surface`, etc.
   - Exposes new classifier threshold columns for spreadsheet tests:
     - `metal_glass_ult_min`
     - `metal_glass_pol_min`
     - `stone_hard_ult_min`
     - `stone_hard_pol_max`
     - `wood_acoustic_min`
     - `wood_acoustic_max`
     - `wood_ultrasonic_min`
     - `wood_ultrasonic_max`
     - `wood_texture_min`
     - `wood_texture_max`
     - `wood_polarization_max`
     - `soft_material_ult_max`
     - `soft_material_pol_max`

2. **Zero-ray edge-case patch**
   - `_stratified_pixels(..., n_samples=0, ...)` now returns empty arrays.
   - `fire_burst(..., n_samples=0, ...)` produces a valid empty burst.
   - `compute_wave_channels()` and `classify_pixels()` can handle the empty burst.

## Colab flow

Upload these two files into the notebook:

- A: `lidar_lenses_wave_v063.py`
- B: `lidar_wave_test_plan_v063.xlsx`

Then run the notebook:

- `lidar_v063_test_harness.ipynb`

## What to inspect

The material-label work should be judged by both raw diagnostics and labels:

- `material_channel_report.csv`
- `material_discrimination_summary.csv`
- `sweep_metrics.csv`
- `edge_case_report.csv`
- `top_contact_sheets.png`

The goal is not just to maximize a score. It is to verify that raw material channels separate known materials, classifier thresholds are wired, deterministic tests are stable, and edge cases fail gracefully or pass cleanly.


## v0.6.3 notes

This patch scopes fine-grained material labels so they only fire when
`material_labels=True`.

Defaults:
- `material_scan`: `material_labels=True`
- `indoor_structure`: `material_labels=False`
- `outdoor_occlusion`: `material_labels=False`
- `full_diagnostic`: `material_labels=False` by default, but the spreadsheet can enable it

Reason: v0.6.2 proved the material classes were wired, but `stone_hard`
was too broad and stole structural pixels in indoor/outdoor presets.
v0.6.3 keeps structural presets structural while allowing material-scan
runs to use `metal_or_glass`, `stone_hard`, and `wood_material`.
