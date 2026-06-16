# LiDAR Lenses Wave Engine

A compact, single-file synthetic LiDAR / wave-sensing probe engine written in
Python with NumPy and PIL — plus a browser studio that drives the real engine for
inspecting and verifying 3D scenes.

**Version: `v0.8`** (working build). v0.8 keeps the v0.7.0 engine core and adds a
within-frame adaptive sampler and a full browser front-end for scene inspection. The
v0.7.0 Colab harnesses and test plans are retained.

## What it is

LiDAR Lenses Wave takes a 3D scene — built-in primitives or an uploaded mesh — and
produces diagnostic, sensor-style views: shaded render, depth and depth variance,
optical / acoustic coherence and anti-coherence, acoustic / ultrasonic intensity, a
polarization-like material proxy, material classification, and boundary-aware edge
maps, assembled into contact sheets with CSV / JSON diagnostics.

Its practical job is **scene inspection and verification**: render a generated or
hand-built 3D scene back through the engine to check whether the geometry came out
the way you intended — scale, framing, holes, silhouette, solidity, placement — and
iterate. The studio's *Check placement* and *Build point cloud* actions (below) are
built for exactly this "did I place it right?" loop.

### Honest scope

This is an experimental synthetic-sensing engine, not a calibrated physics
simulator, and two things are worth being clear about up front:

- **The "wave" channels are not independent sensors.** Depth, the coherence /
  anti-coherence channels, acoustic / ultrasonic intensity, and the polarization
  proxy are all transforms of one first-hit depth buffer plus per-material priors,
  so they correlate heavily with one another. They are useful as visual cues and
  inspection aids — not as separate physical measurements.
- **It is static first-hit raycasting.** One viewpoint at a time, first surface hit,
  no multipath, no motion, no true wave propagation. Transparency is statistical;
  polarization is a scalar proxy, not full Stokes / Mueller.

Said plainly: it is a strong multi-channel *visualizer and scene checker*, not a
sensor-physics simulator.

## What's new in v0.8

- **Patched engine** (`lidar_lenses_wave_v080_alpha5_1_patched.py`) — the v0.7 fixes
  plus a material-channel de-saturation fix.
- **Scout-then-fill sampler** ("smart sampling") — an opt-in within-frame adaptive
  sampler. It spends ~10% of the ray budget scouting to find where rays actually
  return, then concentrates the rest around those confirmed hits, so more of the
  subject is resolved for the same ray count. Exposed as
  `run_sensor_preset(..., sampler="scout_fill")`. The win is largest when the
  subject is small in frame (a thin model fills only a sliver, so a uniform scan
  wastes most rays on empty space). It is **quality-per-budget, not a speed-up** — it
  casts the same number of rays, and rays that hit geometry cost more to trace than
  rays into empty space.
- **Camera Studio** (`lidar_studio_camera.py`) — a single-file, standard-library web
  front-end that drives the *real* engine in the browser via the same
  `run_sensor_preset()` call the engine's own harnesses use. Three actions:
  - **Run scan** — the full multimodal contact sheet, with the smart-sampling toggle.
  - **Check placement** — three canonical views (front / side / top) plus, for
    primitive scenes, a geometry sanity report: inventory by type, cross-type
    interpenetrations, bounds, and an advisory floating count.
  - **Build point cloud** — scans five canonical views, fuses the hits (auto-
    registered: world coordinates + known camera poses, so fusion is just
    concatenation, no alignment), and exports a self-contained interactive three.js
    point-cloud HTML. The smart-sampling toggle controls its density.

Full studio documentation, command-line flags, security flags, and HTTP endpoints
are in **`README_lidar_studio_camera.md`**.

## Files

v0.8 working build:

```
lidar_lenses_wave_v080_alpha5_1_patched.py   engine (v0.7 core + de-sat fix + scout-fill sampler)
lidar_studio_camera.py                        browser studio (scan / placement / point cloud)
README_lidar_studio_camera.md                 studio documentation
```

v0.7.0 (retained — engine lineage, Colab harnesses, test plans, sample outputs):

```
lidar_lenses_wave_v070.py
lidar_v070_test_harness.ipynb
lidar_wave_test_plan_v070.xlsx
v070_release_quality_480.py
v070_release_quality_480_harness.ipynb
release_quality_480_plan_v070.xlsx
RELEASE_NOTES_v070.md
v070_smoke_*                                  sample contact sheet, masks, diagnostics
```

## Engine concept

The engine separates scene inspection into layers:

```
depth / depth_variance     geometry and range
light_anti / sound_anti    raw wave-inspired structure signals
edge_score_raw             raw anti-wave evidence
edge_score_geom            anti-wave evidence filtered by geometry support
geom_edge                  boundary / silhouette / discontinuity layer
material_core              material labels with boundary pixels ignored
material_filled            material labels with conservative edge filling
```

The important rule: **`geom_edge` is not a material.** It marks boundaries,
silhouettes, mixed-depth transitions, and strong optical / acoustic discontinuities.
v0.8 keeps v0.7.0's default fused edge path (`edge_score_mode = geom_fused`,
`edge_fusion_mode = depth_grad_mul`, `edge_anti_min = 0.08`); `raw_anti` remains
available as a debug / control mode.

### Outputs

Typical contact-sheet panels: `shaded`, `depth`, `light_anti`, `sound_anti`,
`edge_score_raw`, `edge_score_geom`, `edge_confidence`, `geom_edge_overlay`,
`classification`, `depth_variance`, `acoustic_intensity`, `ultrasonic_intensity`,
`material_core`, `material_filled`.

Typical Colab-harness reports: `sweep_metrics.csv`, `material_channel_report.csv`,
`material_discrimination_summary.csv`, `material_presence_report.csv`,
`material_core_agreement.csv`, `material_filled_agreement.csv`,
`boundary_adjacency_report.csv`, `structure_density_report.csv`,
`edge_threshold_diagnostics_summary.csv`, `determinism_report.csv`,
`edge_case_report.csv`.

Built-in scenes: `demo`, `material_targets`, `occluder_gate`. Common presets:
`compact_diagnostic`, `indoor_structure`, `outdoor_occlusion`, `material_scan`,
`edge_debug`, `full_diagnostic`.

## Quick start

### The studio (recommended for v0.8)

```bash
pip install numpy Pillow          # core; add matplotlib pandas openpyxl for the Colab harnesses
python lidar_studio_camera.py --engine lidar_lenses_wave_v080_alpha5_1_patched.py
# then open http://localhost:8080
```

Pick a scene (cabin demo, material board, or upload a `.stl` / `.obj`), choose a
preset, and **Run scan** — or **Check placement** / **Build point cloud**. The
**Smart sampling** checkbox concentrates rays on the subject and drives both the scan
and the point cloud. The server binds to `127.0.0.1` (local only) by default; see the
studio README for remote-bind and engine-loading security flags.

### The engine directly / Colab

```bash
python lidar_lenses_wave_v080_alpha5_1_patched.py --preset=indoor_structure
```

Or use the retained v0.7.0 Colab harnesses — upload the engine, the matching
`*_test_harness.ipynb`, and the `*_plan_*.xlsx`, then run all cells for full contact
sheets, CSV / JSON diagnostics, and a downloadable output ZIP.

## Requirements

Core: **Python 3.8+**, `numpy`, `Pillow`. The Colab harnesses also use `matplotlib`,
`pandas`, `openpyxl`. The studio adds nothing — server and page are pure standard
library; the point-cloud viewer pulls three.js from a CDN in the browser.

## Known limitations

- First-hit raycasting; no full multipath simulation; one static viewpoint per scan.
- The wave channels correlate strongly (shared depth buffer + material priors) — they
  are inspection aids, not independent sensors.
- Transparency is statistical; polarization is a scalar proxy, not full
  Stokes / Mueller.
- Material classification works best through `material_core`, not raw classification
  alone.
- **Smart sampling** is quality-per-budget, not a speed-up, and needs a sampler-aware
  engine (the studio detects this and falls back to uniform otherwise).
- **Placement floating-detection is advisory** — it over-reports on plane / billboard
  geometry, so trust the canonical views and the inventory / interpenetration report.
  Uploaded meshes get the views only, not the per-object report.
- The point cloud embeds its points inline in the HTML, so denser clouds mean bigger
  files (~190 KB at ~5k points, roughly 0.5–1 MB at ~14k).

## Roadmap

**v0.8 (done):** material-channel de-saturation fix; within-frame scout-then-fill
sampler; the Camera Studio with scan, placement check, and multi-view point-cloud
export.

**Explored, not merged:** a rolling / cross-frame ray-memory fork for steerable or
moving sensors (world-hit-voxel keying plus a frontier / information-gain gate). It
only pays off on dynamic scenes and stays a separate research direction rather than
part of the static-inspection engine.

**v0.8+ ideas — sensor morphology / "sensor zoo" (aspirational, not built):** weighted
rays, ray timing, beam profiles, foveated eyes, compound eyes, slit pupils, sonar
cones, rolling scans, custom ray-release schedules.

## License

Copyright (c) <2026> Steveon Walker

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

