# LiDAR/Wave v0.6.1 Test Harness

This patch fixes the v0.6.0 workbook-reading issue where title rows caused `KeyError('scene')`.

Use in Colab:
1. Open `lidar_v061_test_harness.ipynb`
2. Upload A: `lidar_lenses_wave_v061.py`
3. Upload B: `lidar_wave_test_plan_v061.xlsx`
4. Run the notebook

v0.6.1 changes:
- Robust Excel reader scans for the real header row (`enabled`, `scene`, `preset`)
- Clean workbook with headers on row 1
- Explicit classifier threshold columns in SweepPlan
- Wiring-test rows for soft/hard material thresholds

If you use the old v0.6.0 workbook, the v0.6.1 notebook should still read it correctly.
