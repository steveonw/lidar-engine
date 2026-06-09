# Release Notes — v0.8-alpha5.1 BeamEvidenceLite

## Status

Experimental alpha harness/reporting patch. Keep v0.7.0 as the stable public baseline.

## Purpose

Alpha5 produced the best-balanced BeamEvidenceLite behavior so far. Alpha5.1 freezes that engine behavior and improves the harness reports so the new ratio channels and gate decisions are visible in the output zip.

## Engine behavior

No intentional sensing/classification behavior changes from alpha5.

The included engine file is version-renamed for convenience:

```text
lidar_lenses_wave_v080_alpha5_1.py
```

but the goal of alpha5.1 is reporting, not algorithm change.

## Harness changes

Added/repaired output reporting:

```text
quick_verdict.csv
  now includes beam_mode_gap_ratio_p95, beam_back_ratio_p95,
  gap_ratio_candidate_frac_of_hit_px, back_ratio_candidate_frac_of_hit_px,
  coherent_guard_candidate_frac_of_hit_px

edge_threshold_diagnostics_summary.csv
  now built from channel summary rows so derived ratio percentiles/candidate counts appear

gate_breakdown.csv
  long-form one-row-per-run/per-gate counts and fractions

class_balance_summary.csv
  long-form one-row-per-run/per-class counts and fractions

*_candidate_gates.png
  per-run mask sheet showing hit/valid/split/front-back/gap-ratio/back-ratio/
  coherent-guard/partial/edge/solid candidate behavior
```

## Why this patch exists

Alpha5 fixed the alpha4 over-permissive partial-occluder behavior using:

```text
beam_mode_gap_ratio
beam_back_ratio
coherent_surface_guard
```

Alpha5.1 makes those gates easy to review from the zip without opening the `.npz` arrays manually.

## Verification

Tiny smoke run completed:

```text
runs_requested: 2
runs_completed: 2
runs_failed: 0
```

No full 480 release-quality check was run in this environment.
