# LiDAR Lenses Wave v0.6.10

v0.6.10 is the wave/edge cleanup after the wave-pattern lab v003.

## What changed

The wave lab showed that raw anti-wave evidence is useful, but too stripe-prone to drive `geom_edge` directly. v0.6.10 separates raw evidence from geometry-supported edge evidence.

### New channels

- `edge_score_raw`: raw `min(light_anti, sound_anti)` anti-wave evidence.
- `edge_score_geom`: anti-wave evidence gated by depth-gradient/depth-variance support.
- `edge_score`: active edge score used by the classifier.
- `depth_gradient`: normalized depth-gradient support map.
- `depth_var_support`: normalized depth-variance support map.

### New defaults

- `edge_score_mode = "geom_fused"`
- `edge_fusion_mode = "depth_grad_mul"`
- `carrier_mode = "ensemble"` now defaults to a golden-angle four-basis schedule when no explicit `carrier_angles` are supplied.

### Why

The old path could show visible striping in `light_anti`, `sound_anti`, `edge_score`, and then leak that into `geom_edge`. v0.6.10 keeps raw anti-wave as a debug signal but makes `geom_edge` depend on geometric support.

## Key comparison

Use the included test plan:

- Run 1: `geom_fused + depth_grad_mul`
- Run 3: `raw_anti + raw`

Run 3 should usually show more `geom_edge` on carrier/stripe residue. Run 1 should be cleaner.

## Colab use

Upload:

1. `lidar_lenses_wave_v0610.py`
2. `lidar_wave_test_plan_v0610.xlsx`

Then run:

`lidar_v0610_test_harness.ipynb`

## Recommended status

v0.6.10 is a candidate for the final v0.7 path if the contact sheets look cleaner than v0.6.9 and the regression metrics stay stable.

## Important note

This is still a lightweight heuristic synthetic multimodal sensing engine, not a calibrated physical LiDAR/sonar simulator.
