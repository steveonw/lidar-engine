# Release Notes — v0.7.0

v0.7.0 promotes the v0.6.11 engine to the stable public experimental line.

## Why v0.7.0

The v0.6.x series established the core system:

- multimodal wave channels
- material priors
- material target board
- `geom_edge` as a boundary layer
- `material_core` / `material_filled`
- edge score and confidence diagnostics
- runtime and threshold reporting
- Colab sweep harnesses

The final release-quality 480×320 visual run showed that v0.6.11 had the right balance:

```text
480×320
8 rays/pixel
stack 4
edge_score_mode = geom_fused
edge_fusion_mode = depth_grad_mul
```

## Default edge behavior

v0.7.0 keeps the v0.6.11 edge architecture:

```text
edge_score_raw  = raw min(light_anti, sound_anti)
edge_score_geom = anti-wave × geometry support
edge_score      = geom_fused by default
```

`raw_anti` remains available as a debug/control mode.

## Recommended default

```text
edge_score_mode = geom_fused
edge_fusion_mode = depth_grad_mul
edge_anti_min = 0.08
```

## What changed from v0.6.11

Mostly naming, packaging, and release framing.

The math is intentionally not changed. v0.7.0 is a release promotion, not another tuning experiment.

## Included artifacts

- engine
- standard test harness
- standard test-plan spreadsheet
- 480×320 release-quality harness
- 480×320 release-quality plan
- README
- release notes
