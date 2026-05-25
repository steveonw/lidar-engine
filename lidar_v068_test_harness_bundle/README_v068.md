# LiDAR/Wave v0.6.8 Test Harness

This release is a cleanup/stability pass after the v0.6.6 geom-edge milestone.

## Upload flow

In Colab:

- **A** = `lidar_lenses_wave_v068.py`
- **B** = `lidar_wave_test_plan_v068.xlsx`

Open `lidar_v068_test_harness.ipynb`, run the upload cell, upload A and B, then run the notebook.

## What v0.6.8 changes

- `run_sensor_preset()` now uses `material_prior_acoustic`, `material_prior_ultrasonic`, and `material_prior_polarization` by default.
- `fire_burst_wave()` accepts and passes through material-prior functions.
- Material-label presets force ultrasonic + polarization evidence when `material_labels=True`.
- Diagnostics report the actual `L_ref` used, rather than recomputing from the last burst.
- Diagnostics include missing label fractions: `partial_occluder`, `optical_only`, and `acoustic_only`.
- `edge_score` and `edge_confidence` are added as derived channel outputs.
- Preset rendering caches classification once per run, avoiding redundant `classify_pixels()` calls.
- The command-line path now uses the preset system, e.g.:

```bash
python lidar_lenses_wave_v068.py --preset=material_scan --material-board --out=preset_out
python lidar_lenses_wave_v068.py --pack --out=preset_pack
```

## Expected reports

The notebook writes:

- `sweep_metrics.csv`
- `material_channel_report.csv`
- `material_label_agreement.csv`
- `material_core_agreement.csv`
- `material_filled_agreement.csv`
- `boundary_adjacency_report.csv`
- `structure_density_report.csv`
- `determinism_report.csv`
- `edge_case_report.csv`
- `top_contact_sheets.png`
- a zipped output bundle

## Interpretation

v0.6.8 is not a new physics-feature release. It is an internal-consistency release:
the material-board preset now uses material priors by default, geom-edge evidence is exposed as a continuous score/confidence layer, and the reports are more complete and faster.


## v0.6.8 edge-threshold guardrail

This release fixes the all-views contact-sheet issue where `edge_score` contained visible structure but
`edge_confidence`, `geom_edge_mask`, and `geom_edge_overlay` could go black.

The fix is a conservative upper clamp:

```python
edge_threshold = min(edge_threshold, edge_anti_max)
```

Default:

```python
edge_anti_max = 0.95
```

New/updated diagnostics:

- `edge_threshold_used`
- `edge_anti_max`
- `edge_score_p90`
- `edge_score_p95`
- `edge_score_p99`
- `edge_confidence_max`

Expected behavior: if `edge_score` has visible structure, the binary/overlay edge panels should no longer be empty solely because the adaptive threshold reached 1.0.
