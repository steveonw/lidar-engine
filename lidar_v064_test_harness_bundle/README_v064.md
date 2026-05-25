# LiDAR/Wave v0.6.4 Test Harness

This version focuses on the material target-board issue found in v0.6.3.

## Main change

v0.6.4 adds a dedicated material target board scene designed to make every
material panel visibly measurable:

- wood
- metal
- glass
- plastic
- cloth
- carpet
- stone
- foliage
- ground

The notebook now reports whether expected materials were actually visible in
`material_channel_report`.

## Colab upload flow

Upload:

A. `lidar_lenses_wave_v064.py`  
B. `lidar_wave_test_plan_v064.xlsx`

Then run the notebook:

`lidar_v064_test_harness.ipynb`

## New/updated outputs

The notebook can produce:

- `sweep_metrics.csv`
- `material_channel_report.csv`
- `material_discrimination_summary.csv`
- `material_presence_report.csv`
- `determinism_report.csv`
- `edge_case_report.csv`
- `top_contact_sheets.png`
- `lidar_wave_test_outputs_v064.zip`

## What to check first

1. Open `material_presence_report.csv`.
2. Confirm all expected materials are present.
3. Then inspect `material_channel_report.csv`.
4. Only tune classifier thresholds after confirming the raw channels are visible and separated.

## Expected result

The material board should no longer report only cloth/foliage/ground/stone.
It should expose wood, metal, glass, plastic, cloth, carpet, stone, foliage,
and ground with enough pixels for diagnostics.

Structural presets still keep `material_labels=False`, so fine material labels
should not steal pixels from indoor/outdoor scene-structure outputs.
