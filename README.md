# LiDAR Lenses Wave Engine

A compact, single-file synthetic LiDAR / wave-sensing probe engine written in Python with NumPy and PIL.

This repo contains the **v0.7.0 stable public experimental build** of the engine, plus Colab notebooks, spreadsheet test plans, and release-quality visual harnesses.

## What it is

LiDAR Lenses Wave is a lightweight synthetic scene-inspection engine. It takes a simple 3D scene made from primitives and produces diagnostic sensor-style views:

- shaded render
- depth and depth variance
- optical coherence / anti-coherence
- acoustic and ultrasonic intensity
- polarization-like material proxy
- material classification
- boundary-aware edge maps
- material core / material filled views
- contact sheets and CSV diagnostics

The engine is useful for testing how generated 3D scenes behave under different synthetic sensing modes.

## Current release

**Version:** `v0.7.0`

Main files:

```text
lidar_lenses_wave_v070.py
lidar_v070_test_harness.ipynb
lidar_wave_test_plan_v070.xlsx
v070_release_quality_480.py
v070_release_quality_480_harness.ipynb
release_quality_480_plan_v070.xlsx
README_v070.md
RELEASE_NOTES_v070.md
```

## Quick start

### Option 1 — Colab standard test harness

Upload these files to Google Colab:

```text
lidar_lenses_wave_v070.py
lidar_v070_test_harness.ipynb
lidar_wave_test_plan_v070.xlsx
```

Open `lidar_v070_test_harness.ipynb` and run all cells.

The harness produces contact sheets, CSV reports, diagnostic JSON files, material reports, edge reports, and a downloadable output ZIP.

### Option 2 — release-quality 480×320 visual check

Upload:

```text
lidar_lenses_wave_v070.py
v070_release_quality_480.py
v070_release_quality_480_harness.ipynb
release_quality_480_plan_v070.xlsx
```

Open `v070_release_quality_480_harness.ipynb` and run all cells.

Default visual-check settings:

```text
width: 480
height: 320
rays_per_pixel: 8
stack: 4
edge_score_mode: geom_fused
edge_fusion_mode: depth_grad_mul
```

## Engine concept

The engine separates scene inspection into layers:

```text
depth / depth_variance
  geometry and range

light_anti / sound_anti
  raw wave-inspired structure signals

edge_score_raw
  raw anti-wave evidence

edge_score_geom
  anti-wave evidence filtered by geometry support

geom_edge
  boundary / skeleton / discontinuity layer

material_core
  material labels with boundary pixels ignored

material_filled
  material labels with conservative edge filling
```

The important rule:

```text
geom_edge is not a material.
```

It marks boundaries, silhouettes, mixed-depth transitions, and strong optical/acoustic discontinuities.

## Default edge behavior

v0.7.0 uses the cleaner fused edge path by default:

```text
edge_score_mode = geom_fused
edge_fusion_mode = depth_grad_mul
edge_anti_min = 0.08
```

`raw_anti` is still available as a debug/control mode.

## Outputs

Typical contact-sheet panels include:

```text
shaded
depth
light_anti
sound_anti
edge_score_raw
edge_score_geom
edge_confidence
geom_edge_overlay
classification
depth_variance
acoustic_intensity
ultrasonic_intensity
material_core
material_filled
```

Typical reports include:

```text
sweep_metrics.csv
material_channel_report.csv
material_discrimination_summary.csv
material_presence_report.csv
material_core_agreement.csv
material_filled_agreement.csv
boundary_adjacency_report.csv
structure_density_report.csv
edge_threshold_diagnostics_summary.csv
determinism_report.csv
edge_case_report.csv
```

## Included scenes and presets

Common scenes:

```text
demo
material_targets
occluder_gate
```

Common presets:

```text
compact_diagnostic
indoor_structure
outdoor_occlusion
material_scan
edge_debug
full_diagnostic
```

## Requirements

The engine is intentionally lightweight.

Core dependencies:

```text
numpy
Pillow
matplotlib
pandas
openpyxl
```

For Colab, the notebooks install or use the required packages.

## Run locally

Example:

```bash
python lidar_lenses_wave_v070.py --preset=indoor_structure
```

Release-quality visual harness:

```bash
python v070_release_quality_480.py \
  --engine lidar_lenses_wave_v070.py \
  --plan release_quality_480_plan_v070.xlsx \
  --outdir v070_release_quality_480_outputs
```

## Known limitations

This is an experimental synthetic sensing engine, not a calibrated physics simulator.

Known limits:

- first-hit raycasting
- no full multipath simulation
- transparency is statistical
- acoustic / ultrasonic channels are heuristic material-response channels
- polarization is a scalar proxy, not full Stokes/Mueller polarization
- material classification works best through `material_core`, not raw classification alone

## Roadmap

### v0.7.x

Cleanup, docs, examples, and test harness stability.

### v0.8 idea: sensor morphology / sensor zoo

Possible future work:

```text
weighted rays
ray timing
beam profiles
foveated eyes
compound eyes
slit pupils
sonar cones
rolling scans
custom ray-release schedules
```

## License

Add a license before treating this as reusable open-source software.

Suggested simple choices:

- MIT License
- Apache-2.0 License




Quick GitHub cleanup:
