
#!/usr/bin/env python3
"""
v070_release_quality_480.py

Release-quality 480x320 visual check harness for lidar_lenses_wave_v070.py.

Runs a small preset pack at:
  width=480
  height=320
  rays_per_pixel=8
  stack=4
  edge_score_mode=geom_fused
  edge_fusion_mode=depth_grad_mul

Outputs:
  release_quality_summary.csv
  release_quality_overview.png
  all contact sheets / diagnostics / channel npz files
  v070_release_quality_480_outputs.zip
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import pandas as pd
except Exception:
    pd = None


def load_engine(engine_path: str):
    spec = importlib.util.spec_from_file_location("llw_release_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llw_release_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_scene(llw, scene_name: str):
    s = str(scene_name or "demo").strip().lower()
    if s in ("demo", "cabin", "cabin_demo", "default"):
        return llw._build_demo_scene(), "demo"
    if s in ("material", "material_targets", "material_board", "board"):
        prims = llw._build_material_target_board_scene()
        if hasattr(llw, "apply_material_prior_transparency"):
            prims = llw.apply_material_prior_transparency(prims)
        return prims, "material_targets"
    raise ValueError(f"Unknown scene {scene_name!r}. Use demo or material_targets.")


def read_plan(plan_path: str):
    if pd is None:
        raise RuntimeError("pandas/openpyxl are required in the Colab notebook. Run the install cell first.")
    df = pd.read_excel(plan_path, sheet_name="Release480")
    df = df.fillna("")
    rows = []
    for _, row in df.iterrows():
        enabled = str(row.get("enabled", "TRUE")).strip().lower()
        if enabled in ("0", "false", "no", "off"):
            continue
        rows.append(row.to_dict())
    return rows


def as_int(v, default):
    try:
        if v == "" or v is None:
            return default
        return int(v)
    except Exception:
        return default


def as_float(v, default):
    try:
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def collect_diag_summary(diag: dict, row: dict, paths: dict):
    counts = diag.get("classification_counts") or {}
    depth_stats = diag.get("depth_stats") or {}
    n_hit = max(1, int(counts.get("solid_surface", 0)) + int(counts.get("geom_edge", 0)) +
                int(counts.get("hard_smooth", 0)) + int(counts.get("partial_occluder", 0)) +
                int(counts.get("optical_only", 0)) + int(counts.get("acoustic_only", 0)) +
                int(counts.get("uncertain", 0)) + int(counts.get("foliage", 0)) +
                int(counts.get("metal_or_glass", 0)) + int(counts.get("stone_hard", 0)) +
                int(counts.get("wood_material", 0)) + int(counts.get("soft_material", 0)))
    return {
        "run_name": row.get("run_name", ""),
        "scene": diag.get("scene", row.get("scene", "")),
        "preset": diag.get("preset", row.get("preset", "")),
        "width": diag.get("width"),
        "height": diag.get("height"),
        "rays_per_pixel": diag.get("rays_per_pixel"),
        "stacked_bursts": diag.get("stacked_bursts"),
        "edge_score_mode": diag.get("edge_score_mode"),
        "edge_fusion_mode": diag.get("edge_fusion_mode"),
        "coverage": depth_stats.get("coverage", diag.get("coverage")),
        "depth_span_p05_p95": depth_stats.get("depth_span_p05_p95"),
        "edge_threshold_used": diag.get("edge_threshold_used"),
        "edge_score_raw_p95": diag.get("edge_score_raw_p95"),
        "edge_score_geom_p95": diag.get("edge_score_geom_p95"),
        "edge_confidence_max": diag.get("edge_confidence_max"),
        "geom_edge_count": counts.get("geom_edge", 0),
        "geom_edge_frac_est": float(counts.get("geom_edge", 0)) / n_hit,
        "partial_occluder_count": counts.get("partial_occluder", 0),
        "optical_only_count": counts.get("optical_only", 0),
        "uncertain_count": counts.get("uncertain", 0),
        "solid_surface_count": counts.get("solid_surface", 0),
        "hard_smooth_count": counts.get("hard_smooth", 0),
        "total_runtime_seconds": diag.get("total_runtime_seconds"),
        "autoframe_runtime_seconds": diag.get("autoframe_runtime_seconds"),
        "render_runtime_seconds": diag.get("render_runtime_seconds"),
        "report_runtime_seconds": diag.get("report_runtime_seconds"),
        "save_runtime_seconds": diag.get("save_runtime_seconds"),
        "contact_sheet": paths.get("contact_sheet", ""),
        "diagnostics": paths.get("diagnostics", ""),
        "notes": row.get("notes", ""),
    }


def make_overview(summary_rows, out_path: Path, thumb_w=420):
    images = []
    for r in summary_rows:
        p = r.get("contact_sheet", "")
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        scale = thumb_w / img.width
        thumb = img.resize((thumb_w, max(1, int(img.height * scale))))
        title = f"{r.get('scene')} / {r.get('preset')} | edge={r.get('geom_edge_frac_est', 0):.3f}"
        pad = 34
        canvas = Image.new("RGB", (thumb.width, thumb.height + pad), (20, 20, 20))
        canvas.paste(thumb, (0, pad))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 8), title, fill=(240, 240, 240), font=font)
        images.append(canvas)
    if not images:
        return None
    cols = 1 if len(images) <= 2 else 2
    rows = (len(images) + cols - 1) // cols
    cell_w = max(i.width for i in images)
    cell_h = max(i.height for i in images)
    overview = Image.new("RGB", (cols * cell_w, rows * cell_h), (10, 10, 10))
    for idx, img in enumerate(images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        overview.paste(img, (x, y))
    overview.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--outdir", default="v070_release_quality_480_outputs")
    args = ap.parse_args()

    llw = load_engine(args.engine)
    rows = read_plan(args.plan)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, row in enumerate(rows):
        scene_name = row.get("scene", "demo")
        preset = str(row.get("preset", "compact_diagnostic"))
        run_name = str(row.get("run_name", f"run_{i+1:02d}_{scene_name}_{preset}"))
        prims, scene_key = build_scene(llw, scene_name)
        run_dir = outdir / run_name
        run_dir.mkdir(exist_ok=True)

        overrides = {
            "width": as_int(row.get("width"), 480),
            "height": as_int(row.get("height"), 320),
            "rays_per_pixel": as_int(row.get("rays_per_pixel"), 8),
            "stack": as_int(row.get("stack"), 4),
            "edge_score_mode": str(row.get("edge_score_mode", "geom_fused")),
            "edge_fusion_mode": str(row.get("edge_fusion_mode", "depth_grad_mul")),
            "edge_anti_min": as_float(row.get("edge_anti_min"), 0.08),
            "adaptive_edge_percentile": as_float(row.get("adaptive_edge_percentile"), 92.0),
            "edge_anti_max": as_float(row.get("edge_anti_max"), 0.95),
            "pilot_rays": as_int(row.get("pilot_rays"), 4000),
        }

        result = llw.run_sensor_preset(
            prims,
            preset_name=preset,
            scene_name=scene_key,
            out_dir=str(run_dir),
            seed=as_int(row.get("seed"), 42),
            **overrides,
        )
        summary_rows.append(collect_diag_summary(result["diagnostics"], row, result.get("paths", {})))

    summary_csv = outdir / "release_quality_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)

    overview_path = outdir / "release_quality_overview.png"
    make_overview(summary_rows, overview_path)

    zip_path = outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outdir.rglob("*"):
            z.write(p, arcname=p.relative_to(outdir.parent))

    print(json.dumps({
        "outdir": str(outdir),
        "summary_csv": str(summary_csv),
        "overview": str(overview_path),
        "zip": str(zip_path),
        "runs": len(summary_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
