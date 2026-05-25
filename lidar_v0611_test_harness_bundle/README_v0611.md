# LiDAR Lenses Wave v0.6.11

v0.6.11 is a small tuning release after v0.6.10.

v0.6.10 successfully replaced raw anti-wave edge classification with a geometry-supported edge score, but the old raw-anti edge threshold was too high for the quieter fused score. The result was clean but under-active `geom_edge`.

v0.6.11 keeps the v0.6.10 edge architecture and only retunes the threshold floor for `geom_fused`.

## Main change

```text
edge_score_raw  = min(light_anti, sound_anti)
edge_score_geom = raw anti-wave × geometric support
edge_score      = edge_score_geom by default
```

v0.6.11 changes the default fused-edge floor:

```text
geom_fused edge_anti_min: ~0.08
raw_anti control:         ~0.30–0.35
```

The adaptive percentile still decides the final scene-aware threshold when it is above the floor.

## Why

`edge_score_geom` is intentionally much quieter than `edge_score_raw`, because it suppresses broad carrier/interference residue. It should not use the same threshold floor as raw anti-wave.

## Colab usage

Upload:

```text
A = lidar_lenses_wave_v0611.py
B = lidar_wave_test_plan_v0611.xlsx
```

Run `lidar_v0611_test_harness.ipynb`.

## Expected result

Compared with v0.6.10:

```text
more geom_edge than the too-quiet fused run
far less stripe/noise than raw_anti
edge_confidence and geom_edge_overlay stay active
```

## Key outputs

```text
sweep_metrics.csv
edge_threshold_diagnostics_summary.csv
structure_density_report.csv
material_core_agreement.csv
material_filled_agreement.csv
boundary_adjacency_report.csv
top_contact_sheets.png
```

## Recommended status

If v0.6.11 contact sheets look clean and `geom_edge_frac` returns to the useful band, this can become the v0.7 release candidate.
