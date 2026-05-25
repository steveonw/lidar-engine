# LiDAR/Wave v0.6.5 Test Harness

This version focuses on **presentation + material-board evaluation**, not raw sensor math.

## Files

- `lidar_lenses_wave_v065.py` — engine module
- `lidar_v065_test_harness.ipynb` — upload-A+B Colab notebook
- `lidar_wave_test_plan_v065.xlsx` — test plan workbook
- `v065_smoke/` — local smoke-test outputs

## Colab upload flow

Open the notebook, run the first cell, then upload:

- A = `lidar_lenses_wave_v065.py`
- B = `lidar_wave_test_plan_v065.xlsx`

The notebook writes:

- `sweep_metrics.csv`
- `material_channel_report.csv`
- `material_discrimination_summary.csv`
- `material_label_agreement_report.csv`
- `material_presence_report.csv`
- `determinism_report.csv`
- `edge_case_report.csv`
- `top_contact_sheets.png`
- a zipped output archive

## v0.6.5 changes

### 1. Classification display styles

The classifier labels are unchanged, but the contact-sheet display can now use styles:

- `default`
- `indoor_sparse`
- `material_focus`

`indoor_sparse` darkens `solid_surface` so indoor classification panels do not wash out as beige/white.
`material_focus` de-emphasizes generic `solid_surface` / `hard_smooth` and highlights fine material labels.

### 2. Material label agreement report

The engine now reports label agreement by dominant material/piece type. This avoids scoring the whole material board by floor/backdrop pixels.

Example expectations:

- `metal`, `glass` → `metal_or_glass`
- `wood` → `wood_material`
- `stone` → `stone_hard`
- `cloth`, `carpet` → `soft_material`
- `foliage` → `foliage`

### 3. Workbook additions

The test plan includes a `classification_style` column and material diagnostics rows that exercise the material agreement report.

## Notes

Raw channels are still the primary signal. The classifier is a compressed overlay for tests and visualization.
