# LiDAR Lenses Wave v0.6.9 Test Harness

v0.6.9 is a cleanup/stability candidate before v0.7.0. It keeps the v0.6.8 sensor math and focuses on making the engine easier to trust and test.

## Upload flow in Colab

Upload:

- A = `lidar_lenses_wave_v069.py`
- B = `lidar_wave_test_plan_v069.xlsx`

Then run `lidar_v069_test_harness.ipynb`.

## Main v0.6.9 changes

- Runtime diagnostics split:
  - `autoframe_runtime_seconds`
  - `render_runtime_seconds`
  - `report_runtime_seconds`
  - `save_runtime_seconds`
  - `total_runtime_seconds`
- Preset-specific contact sheets:
  - `compact_diagnostic`
  - `edge_debug`
  - existing `indoor_structure`, `outdoor_occlusion`, `material_scan`, `full_diagnostic`, `overhead_layout`
- Edge-case expectations:
  - `pass`
  - `pass_empty`
  - `expected_pathological`
  - `fail`
- Edge threshold diagnostics are present in every wave run.
- CLI/preset path remains first-class:
  - `python lidar_lenses_wave_v069.py --preset=material_scan --material-board`
  - `python lidar_lenses_wave_v069.py --preset=indoor_structure`
  - `python lidar_lenses_wave_v069.py --pack`
- Documentation language is kept conservative: acoustic/material/polarization channels are heuristic synthetic sensing channels, not calibrated real-world physics.

## Important interpretation

`geom_edge` is not a material. It is a boundary/structure layer.

Use:

- `geom_edge_mask` and `geom_edge_overlay` for structure/boundary analysis
- `material_core` for material scoring
- `material_filled` for visualization/secondary attribution

## Spreadsheet sheets

- `SweepPlan` — main preset regressions
- `MaterialDiagnostics` — material board and material-core reports
- `DeterminismPlan` — same-seed and different-seed tests
- `EdgeCases` — pathological/empty/zero-ray cases
- `Targets` — informal expected ranges
- `Notes` — test-plan notes

## Current status

v0.6.9 is intended as the last cleanup release before v0.7.0. Bigger ideas such as sensor morphology, animal-eye sampling, timed rays, and custom beam shapes should wait for v0.8.
