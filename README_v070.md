# LiDAR Lenses Wave v0.7.0

**Status:** stable public experimental engine  
**Promoted from:** v0.6.11 after 480×320 / 8 rpp / stack 4 visual check  
**Default edge path:** `geom_fused`

LiDAR Lenses Wave is a compact NumPy/PIL synthetic multimodal sensing sandbox. It raycasts analytic 3D primitive scenes and produces depth, optical wave, acoustic/ultrasonic, material-prior, polarization-proxy, and boundary-aware diagnostic views.

## What is stable in v0.7.0

- Single-file engine: `lidar_lenses_wave_v070.py`
- Preset-based runner: `run_sensor_preset(...)`
- Boundary-aware edge system:
  - `edge_score_raw`
  - `edge_score_geom`
  - `edge_score`
  - `edge_confidence`
  - `geom_edge_mask`
  - `geom_edge_overlay`
- Material evaluation views:
  - `classification`
  - `material_core`
  - `material_filled`
- Reports:
  - structure density
  - material core/fill agreement
  - boundary adjacency
  - edge threshold diagnostics
  - runtime split diagnostics

## Recommended Colab flow

Upload:

```text
lidar_lenses_wave_v070.py
lidar_v070_test_harness.ipynb
lidar_wave_test_plan_v070.xlsx
```

For final visual checks, use:

```text
v070_release_quality_480_harness.ipynb
v070_release_quality_480.py
release_quality_480_plan_v070.xlsx
```

## Interpretation rule

`geom_edge` is not a material. It is a boundary/structure signal.

Use:

```text
edge_score_raw  = debug/control wave evidence
edge_score_geom = geometry-supported edge evidence
material_core   = best material truth metric
material_filled = visualization/region repair
```

## Known limitations

- First-hit raycasting; no full multipath simulation
- Transparency is statistical, not full volumetric transport
- Acoustic/ultrasonic channels are heuristic material-response channels
- Polarization is a scalar proxy, not full Stokes/Mueller physics
- This is a lightweight synthetic sensing lab, not calibrated real LiDAR/sonar hardware simulation

## Next major direction

v0.8 should be **sensor morphology / sensor zoo**:

```text
weighted rays
ray timing
beam profiles
foveated eyes
compound eyes
slit pupils
sonar cones
rolling scan sensors
custom release schedules
```
