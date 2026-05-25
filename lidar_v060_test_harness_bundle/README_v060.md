# LiDAR/Wave v0.6.0 Test Harness

This bundle changes the workflow from “find a better score” to “document engine behavior and catch regressions.”

## Files

- `lidar_lenses_wave_v060.py` — engine with classifier threshold plumbing and material-channel reporting helpers.
- `lidar_v060_test_harness.ipynb` — Colab notebook with upload-A+B flow.
- `lidar_wave_test_plan_v060.xlsx` — editable test workbook.
- `README_v060.md` — this file.

## Colab workflow

1. Open `lidar_v060_test_harness.ipynb` in Colab.
2. Run the first cell.
3. Upload:
   - A: `lidar_lenses_wave_v060.py`
   - B: `lidar_wave_test_plan_v060.xlsx`
4. Run the notebook.

The notebook writes:

- `sweep_metrics.csv`
- `sweep_metrics.xlsx`
- `material_channel_report.csv`
- `material_channel_report.xlsx`
- `material_discrimination_summary.csv`
- `determinism_report.csv`
- `edge_case_report.csv`
- `top_contact_sheets.png`
- `lidar_wave_test_outputs_v060.zip`

## What changed in v0.6.0

### 1. Classifier threshold columns are now sweepable

The notebook forwards optional spreadsheet columns into `classify_pixels()`:

- `hard_smooth_ult_min`
- `soft_material_texture_min`
- `soft_material_intensity_min`
- `solid_intensity_min`
- `solid_coh_min`
- `foliage_intensity_max`
- `foliage_texture_min`
- `partial_occluder_intensity_max`
- `partial_occluder_var_min`
- plus the existing `edge_anti_min` and `adaptive_edge_percentile`

Rows with intentionally extreme thresholds are included as wiring tests.

### 2. Material channel diagnostics

For each material test row, the notebook reports per-material means and percentiles:

- acoustic intensity
- ultrasonic intensity
- acoustic texture
- acoustic softness
- polarization
- depth variance
- coherence / anti-coherence channels

Use this report before tuning thresholds.

If raw material values overlap, no classifier threshold can reliably separate them. Fix the priors/impedance model first.

### 3. Determinism and variance tests

The notebook can run the same config multiple times with:

- same seed: expected stable output
- different seeds: expected sampling/transparency variance

### 4. Edge cases

The workbook includes tests for:

- zero rays
- zero coverage
- empty scene
- one primitive
- high transparency
- simple partial occluder
- flat top-down scene

## Testing philosophy

The `Targets` sheet is the behavior contract.
The sweep plan is the test case list.
The metrics and reports are the evidence.

Do not treat every run as an optimization contest. Some rows are controls that should produce obvious changes. If they do not, something is not wired correctly.
