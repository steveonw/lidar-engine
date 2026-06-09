# LiDAR Engine v0.8-alpha5.1 — BeamEvidenceLite Reporting Patch

Experimental v0.8 alpha bundle for `steveonw/lidar-engine`.

This does **not** replace the v0.7.0 stable public baseline. v0.7.0 remains the shipped stable experimental engine. Alpha5.1 is a **harness/reporting-only cleanup** on top of the alpha5 BeamEvidenceLite behavior.

## What changed from alpha5

The engine behavior is intentionally unchanged from alpha5.

Alpha5.1 improves the Colab/test-harness output so review is easier:

- `quick_verdict.csv` now includes the ratio gates that made alpha5 work:
  - `beam_mode_gap_ratio_p95`
  - `beam_back_ratio_p95`
  - `gap_ratio_candidate_frac_of_hit_px`
  - `back_ratio_candidate_frac_of_hit_px`
  - `coherent_guard_candidate_frac_of_hit_px`
- `beam_return_summary.csv` keeps the full wide per-run metrics.
- `edge_threshold_diagnostics_summary.csv` is now built from the channel summary rows so derived ratio statistics are not silently omitted.
- New `gate_breakdown.csv` gives one row per run/gate for easier plotting.
- New `class_balance_summary.csv` gives one row per run/class.
- Each run folder now gets a `*_candidate_gates.png` visual sheet showing the candidate masks that lead to final partial/edge/solid decisions.

## Colab workflow

Open:

```text
lidar_v080_alpha5_1_upload_harness.ipynb
```

Upload:

```text
lidar_lenses_wave_v080_alpha5_1.py
lidar_wave_test_plan_v080_alpha5_1.xlsx
```

Run all cells.

The notebook downloads:

```text
lidar_wave_test_outputs_v080_alpha5_1.zip
```

Send that zip back for review.

## Main files

```text
lidar_lenses_wave_v080_alpha5_1.py
v080_alpha5_1_upload_harness.py
lidar_v080_alpha5_1_upload_harness.ipynb
lidar_wave_test_plan_v080_alpha5_1.xlsx
lidar_wave_test_plan_v080_alpha5_1_smoke.xlsx
v080_alpha5_1_test_outputs_smoke.zip
```

## New/repaired harness outputs

```text
gate_breakdown.csv
class_balance_summary.csv
edge_threshold_diagnostics_summary.csv  # now includes ratio-derived stats
quick_verdict.csv                       # now includes ratio gate fractions/p95 values
run_*/..._candidate_gates.png
```

Candidate gates visualized per run:

```text
hit
valid_return
split_candidate
front_back_candidate
gap_candidate
gap_ratio_candidate
back_ratio_candidate
excess_candidate
mix_candidate
coherence_candidate
coherent_guard_candidate
partial_candidate
geom_edge_candidate
coherence_solid_candidate
split_or_back_candidate
partial_geom_overlap_candidate
```

## Local run

```bash
python v080_alpha5_1_upload_harness.py \
  --engine lidar_lenses_wave_v080_alpha5_1.py \
  --plan lidar_wave_test_plan_v080_alpha5_1.xlsx \
  --out v080_alpha5_1_test_outputs \
  --zip lidar_wave_test_outputs_v080_alpha5_1.zip
```

## Smoke verification

A tiny smoke plan was run locally:

```text
runs_requested: 2
runs_completed: 2
runs_failed: 0
```

The smoke run is intentionally small; use the main spreadsheet in Colab for the real visual/metric review.
