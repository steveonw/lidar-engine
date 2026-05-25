# LiDAR/Wave v0.6.6 Test Harness

This version promotes `geom_edge` from a nuisance classifier label into a first-class structure/boundary signal.

## Files

- `lidar_lenses_wave_v066.py` — engine
- `lidar_v066_test_harness.ipynb` — Colab upload-A+B notebook
- `lidar_wave_test_plan_v066.xlsx` — editable test plan workbook

## Colab flow

1. Open `lidar_v066_test_harness.ipynb`.
2. Run the first upload/import cell.
3. Upload:
   - A: `lidar_lenses_wave_v066.py`
   - B: `lidar_wave_test_plan_v066.xlsx`
4. Run all cells.

## New v0.6.6 outputs

The engine still saves the normal contact sheet and diagnostics, and now also writes derived boundary/material images:

- `*_geom_edge_mask.png`
- `*_geom_edge_overlay.png`
- `*_material_core.png`
- `*_material_filled.png`

The notebook also writes new CSV/XLSX reports:

- `structure_density_report.csv`
- `boundary_adjacency_report.csv`
- `material_core_agreement.csv`
- `material_filled_agreement.csv`

## Interpretation

`geom_edge` now has two jobs:

1. As structure: preserve it as an edge/skeleton layer.
2. As a material filter: ignore or conservatively fill it when scoring material interiors.

For material tests, compare:

- full agreement: whole target, including edges
- core agreement: ignores `geom_edge`
- filled agreement: fills confident edge pixels from neighboring labels

If core agreement is much better than full agreement, the material classifier is not necessarily wrong; the target is boundary-heavy.
