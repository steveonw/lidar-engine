
"""
v080_alpha5_1_upload_harness.py

Spreadsheet-driven Colab/local harness for LiDAR Engine v0.8.0-alpha5.1 BeamEvidenceLite.

Flow modeled after the older v0.6.10 harness:
  1. Upload/select engine .py
  2. Upload/select test plan .xlsx
  3. Run every enabled row in sheet "sweep"
  4. Save per-run contact sheets, channels.npz, masks/material views, diagnostics.json
  5. Save summary CSV/JSON reports
  6. Zip the whole output folder for download

Expected uploads in Colab:
  - lidar_lenses_wave_v080_alpha5_1.py
  - lidar_wave_test_plan_v080_alpha5_1.xlsx
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_OUT = "v080_alpha5_1_test_outputs"


def find_first(patterns: Iterable[str], label: str) -> str:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"Could not find {label}. Tried: {list(patterns)}")
    return matches[-1]


def load_engine(engine_path: str):
    engine_path = str(engine_path)
    spec = importlib.util.spec_from_file_location("llw_v080_alpha5_1", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import engine from {engine_path!r}")
    llw = importlib.util.module_from_spec(spec)
    sys.modules["llw_v080_alpha5_1"] = llw
    spec.loader.exec_module(llw)
    return llw


def _safe_name(value: Any) -> str:
    s = str(value).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "scene"


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_str(value: Any, default: Optional[str] = None) -> Optional[str]:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    s = str(value).strip()
    return s if s else default


def _rot(llw, rx=0, ry=0, rz=0):
    return llw.make_rotation_matrix(rx, ry, rz)


def _prim(llw, shape, center, half, color, piece_id, piece_type, rot=None, transparency=0.0):
    if rot is None:
        rot = _rot(llw)
    return llw.Primitive(
        shape=shape,
        center=np.array(center, float),
        half_extents=np.array(half, float),
        rotation_matrix=rot,
        inv_rotation_matrix=rot.T,
        color=tuple(float(c) for c in color),
        piece_id=int(piece_id),
        piece_type=str(piece_type),
        transparency=float(transparency),
    )


def build_occluder_gate(llw):
    """Small v0.6.10-style gate + foliage + background wall stress scene."""
    prims = []
    pid = 1
    prims.append(_prim(llw, "box", [0, -0.5, 0], [7, 0.5, 7], [0.55, 0.50, 0.38], pid, "ground")); pid += 1
    for x in [-2.2, -1.4, -0.6, 0.6, 1.4, 2.2]:
        prims.append(_prim(llw, "cylinder", [x, 0.7, 1.2], [0.06, 0.7, 0.06], [0.82, 0.76, 0.60], pid, "wood")); pid += 1
    prims.append(_prim(llw, "box", [0, 1.1, 1.2], [2.7, 0.08, 0.06], [0.82, 0.76, 0.60], pid, "wood")); pid += 1
    prims.append(_prim(llw, "sphere", [0, 1.35, 1.25], [1.2, 0.75, 0.4], [0.25, 0.70, 0.25], pid, "foliage", transparency=0.55)); pid += 1
    prims.append(_prim(llw, "box", [0, 0.7, 2.8], [2.2, 0.7, 0.08], [0.35, 0.35, 0.35], pid, "stone")); pid += 1
    if hasattr(llw, "apply_material_prior_transparency"):
        prims = llw.apply_material_prior_transparency(prims)
    return prims


def scene_from_name(llw, name: str):
    n = str(name).strip().lower()
    if n in ("demo", "cabin", "cabin_demo", "mini_saloon", "indoor", "demo_compact"):
        prims = llw._build_demo_scene()
    elif n in ("material_targets", "material_board", "material", "material_scan"):
        prims = llw._build_material_target_board_scene()
    elif n in ("occluder_gate", "occlusion", "outdoor_occlusion"):
        return build_occluder_gate(llw)
    else:
        raise ValueError(f"Unknown scene {name!r}. Supported: demo, material_board, occluder_gate.")
    if hasattr(llw, "apply_material_prior_transparency"):
        prims = llw.apply_material_prior_transparency(prims)
    return prims


def load_plan(plan_path: str) -> pd.DataFrame:
    plan = pd.read_excel(plan_path, sheet_name="sweep")
    if "enabled" not in plan.columns:
        plan["enabled"] = True
    plan = plan[plan["enabled"].map(lambda v: _to_bool(v, True))].copy()
    if plan.empty:
        raise ValueError("No enabled rows found in sheet 'sweep'.")
    return plan


def row_overrides(row: pd.Series) -> Dict[str, Any]:
    int_cols = [
        "width", "height", "rays_per_pixel", "stack", "pilot_rays",
        "min_return_hit_count"
    ]
    float_cols = [
        "edge_anti_min", "adaptive_edge_percentile", "edge_anti_max",
        "beam_width", "min_ray_weight", "return_mix_gain",
        "return_q_near", "return_q_far",
        "return_expected_spread_rel", "return_expected_spread_grad",
        "min_return_coverage",
        "beam_split_min_weight_frac", "beam_split_min_abs_gap",
        "beam_split_confidence_gain", "beam_coherence_mix_gain",
        "beam_coherence_split_gain",
        "partial_occluder_mix_min", "foliage_mix_min", "hard_partial_mix_min",
        "beam_split_min", "beam_split_front_min", "beam_split_back_min",
        "beam_coherence_solid_min",
        "partial_occluder_gap_min", "partial_occluder_gap_ratio_min",
        "partial_occluder_back_ratio_min", "partial_occluder_excess_min",
        "partial_occluder_coherence_max", "coherent_surface_guard",
        "fov_deg", "depth_wavelength", "attenuation_per_meter",
        "min_coverage", "min_depth_span"
    ]
    str_cols = [
        "camera", "lens", "sampling_mode", "beam_profile", "carrier_mode",
        "edge_score_mode", "edge_fusion_mode", "classification_style",
        "classification_priority"
    ]
    bool_cols = [
        "include_ultrasonic", "include_polarization", "material_labels",
        "auto_frame", "wave"
    ]

    overrides: Dict[str, Any] = {}

    for k in int_cols:
        if k in row:
            v = _to_int(row.get(k), None)
            if v is not None:
                overrides[k] = v

    for k in float_cols:
        if k in row:
            v = _to_float(row.get(k), None)
            if v is not None:
                overrides[k] = v

    for k in str_cols:
        if k in row:
            v = _to_str(row.get(k), None)
            if v is not None:
                overrides[k] = v

    for k in bool_cols:
        if k in row:
            try:
                missing = pd.isna(row.get(k))
            except Exception:
                missing = False
            if not missing:
                overrides[k] = _to_bool(row.get(k), False)

    # Optional panels column accepts a Python/JSON list or comma-separated string.
    if "panels" in row:
        v = _to_str(row.get("panels"), None)
        if v:
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)):
                    overrides["panels"] = [str(x).strip() for x in parsed]
                else:
                    overrides["panels"] = [x.strip() for x in v.split(",") if x.strip()]
            except Exception:
                overrides["panels"] = [x.strip() for x in v.split(",") if x.strip()]

    return overrides


def pct_from_channel(channels: Dict[str, np.ndarray], hit: np.ndarray, name: str, pct: float) -> Optional[float]:
    arr = channels.get(name)
    if arr is None or hit is None or not np.any(hit):
        return None
    vals = np.asarray(arr)[hit]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, pct))




def candidate_counts_from_channels(channels: Dict[str, np.ndarray], diag: Dict[str, Any]) -> Dict[str, Any]:
    """Alpha5 helper: count evidence candidates before exclusive class priority.

    These are diagnostic counts, not final labels. They help answer whether
    geom_edge or foliage priority is swallowing plausible partial-occluder
    evidence.
    """
    if not channels:
        return {}
    hit = np.asarray(channels.get("hit_count", np.zeros((1, 1)))) > 0
    hit_px = int(hit.sum()) or 1
    ck = dict(diag.get("classifier_kwargs") or {})

    def arr(name: str, default: float = 0.0):
        base = next(iter(channels.values()))
        return np.asarray(channels.get(name, np.zeros_like(base, dtype=float) + default), dtype=float)

    ret_valid = arr("return_valid_stats") > 0.5
    coverage = arr("return_coverage")
    split = arr("beam_split_score")
    front = arr("beam_front_strength")
    back = arr("beam_back_strength")
    gap = arr("beam_mode_gap")
    gap_ratio = arr("beam_mode_gap_ratio")
    if not np.any(gap_ratio):
        depth_for_gap = arr("return_depth_mean")
        gap_ratio = np.where(depth_for_gap > 1e-9, np.clip(gap / np.maximum(depth_for_gap, 1e-9), 0.0, 1.0), 0.0)
    back_ratio = arr("beam_back_ratio")
    if not np.any(back_ratio):
        back_ratio = np.where(front > 1e-9, np.clip(back / np.maximum(front, 1e-9), 0.0, 1.0), 0.0)
    excess = arr("return_depth_spread_excess")
    mix = arr("return_mix_score")
    coherence = arr("beam_coherence")
    edge_score = arr("edge_score_geom")

    min_cov = float(ck.get("min_return_coverage", diag.get("min_return_coverage", 0.25)) or 0.25)
    min_hits = int(ck.get("min_return_hit_count", diag.get("min_return_hit_count", 4)) or 4)
    split_min = float(ck.get("beam_split_min", diag.get("beam_split_min", 0.22)) or 0.22)
    front_min = float(ck.get("beam_split_front_min", diag.get("beam_split_front_min", 0.10)) or 0.10)
    back_min = float(ck.get("beam_split_back_min", diag.get("beam_split_back_min", 0.07)) or 0.07)
    gap_min = float(ck.get("partial_occluder_gap_min", diag.get("partial_occluder_gap_min", 0.20)) or 0.20)
    gap_ratio_min = float(ck.get("partial_occluder_gap_ratio_min", diag.get("partial_occluder_gap_ratio_min", 0.015)) or 0.015)
    back_ratio_min = float(ck.get("partial_occluder_back_ratio_min", diag.get("partial_occluder_back_ratio_min", 0.12)) or 0.12)
    excess_min = float(ck.get("partial_occluder_excess_min", diag.get("partial_occluder_excess_min", 0.05)) or 0.05)
    mix_min = float(ck.get("partial_occluder_mix_min", diag.get("partial_occluder_mix_min", 0.020)) or 0.020)
    coh_max = float(ck.get("partial_occluder_coherence_max", diag.get("partial_occluder_coherence_max", 0.82)) or 0.82)
    solid_min = float(ck.get("beam_coherence_solid_min", diag.get("beam_coherence_solid_min", 0.45)) or 0.45)
    coherent_guard = float(ck.get("coherent_surface_guard", diag.get("coherent_surface_guard", 0.92)) or 0.92)
    edge_thr = float(diag.get("edge_threshold_used") or ck.get("edge_anti_min", 0.08) or 0.08)

    valid = hit & ret_valid & (coverage >= min_cov) & (arr("hit_count") >= min_hits)
    split_candidate = valid & (split > split_min)
    front_back_candidate = valid & (front > front_min) & (back > back_min)
    gap_candidate = valid & (gap > gap_min)
    gap_ratio_candidate = valid & (gap_ratio > gap_ratio_min)
    back_ratio_candidate = valid & (back_ratio > back_ratio_min)
    excess_candidate = valid & (excess > excess_min)
    coherent_guard_candidate = valid & (coherence > coherent_guard) & (split < max(split_min, 0.30)) & (gap_ratio < max(gap_ratio_min, 0.02))
    partial_candidate = (
        split_candidate & front_back_candidate & gap_candidate & gap_ratio_candidate
        & back_ratio_candidate & excess_candidate
        & (mix > mix_min) & (coherence < coh_max) & ~coherent_guard_candidate
    )
    geom_edge_candidate = hit & (edge_score > edge_thr)
    coherence_solid_candidate = valid & (coherence > solid_min) & (split < split_min) & (mix < max(0.04, mix_min * 2.0))
    split_or_back_candidate = valid & ((split > split_min) | (back > back_min))

    out = {
        "valid_return_candidate_count": int(valid.sum()),
        "split_candidate_count": int(split_candidate.sum()),
        "front_back_candidate_count": int(front_back_candidate.sum()),
        "gap_candidate_count": int(gap_candidate.sum()),
        "gap_ratio_candidate_count": int(gap_ratio_candidate.sum()),
        "back_ratio_candidate_count": int(back_ratio_candidate.sum()),
        "excess_candidate_count": int(excess_candidate.sum()),
        "coherent_guard_candidate_count": int(coherent_guard_candidate.sum()),
        "partial_candidate_count": int(partial_candidate.sum()),
        "geom_edge_candidate_count": int(geom_edge_candidate.sum()),
        "coherence_solid_candidate_count": int(coherence_solid_candidate.sum()),
        "split_or_back_candidate_count": int(split_or_back_candidate.sum()),
        "partial_geom_overlap_candidate_count": int((partial_candidate & geom_edge_candidate).sum()),
    }
    for k, v in list(out.items()):
        out[k.replace("_count", "_frac_of_hit_px")] = float(v / hit_px)
    return out


def candidate_masks_from_channels(channels: Dict[str, np.ndarray], diag: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return named boolean masks for alpha5.1 gate-debug contact sheets.

    This mirrors candidate_counts_from_channels, but keeps every gate as a
    separately viewable mask so the output zip shows which condition is doing
    the work: split, front/back, gap ratio, back ratio, excess spread,
    coherence guard, and final partial-candidate agreement.
    """
    if not channels:
        return {}

    base = np.asarray(next(iter(channels.values())))
    hit = np.asarray(channels.get("hit_count", np.zeros_like(base))) > 0
    ck = dict(diag.get("classifier_kwargs") or {})

    def arr(name: str, default: float = 0.0):
        return np.asarray(channels.get(name, np.zeros_like(base, dtype=float) + default), dtype=float)

    ret_valid = arr("return_valid_stats") > 0.5
    coverage = arr("return_coverage")
    hit_count = arr("hit_count")
    split = arr("beam_split_score")
    front = arr("beam_front_strength")
    back = arr("beam_back_strength")
    gap = arr("beam_mode_gap")
    depth = arr("return_depth_mean")
    gap_ratio = arr("beam_mode_gap_ratio")
    if not np.any(gap_ratio):
        gap_ratio = np.where(depth > 1e-9, np.clip(gap / np.maximum(depth, 1e-9), 0.0, 1.0), 0.0)
    back_ratio = arr("beam_back_ratio")
    if not np.any(back_ratio):
        back_ratio = np.where(front > 1e-9, np.clip(back / np.maximum(front, 1e-9), 0.0, 1.0), 0.0)
    excess = arr("return_depth_spread_excess")
    mix = arr("return_mix_score")
    coherence = arr("beam_coherence")
    edge_score = arr("edge_score_geom")

    min_cov = float(ck.get("min_return_coverage", diag.get("min_return_coverage", 0.25)) or 0.25)
    min_hits = int(ck.get("min_return_hit_count", diag.get("min_return_hit_count", 4)) or 4)
    split_min = float(ck.get("beam_split_min", diag.get("beam_split_min", 0.22)) or 0.22)
    front_min = float(ck.get("beam_split_front_min", diag.get("beam_split_front_min", 0.10)) or 0.10)
    back_min = float(ck.get("beam_split_back_min", diag.get("beam_split_back_min", 0.07)) or 0.07)
    gap_min = float(ck.get("partial_occluder_gap_min", diag.get("partial_occluder_gap_min", 0.20)) or 0.20)
    gap_ratio_min = float(ck.get("partial_occluder_gap_ratio_min", diag.get("partial_occluder_gap_ratio_min", 0.015)) or 0.015)
    back_ratio_min = float(ck.get("partial_occluder_back_ratio_min", diag.get("partial_occluder_back_ratio_min", 0.12)) or 0.12)
    excess_min = float(ck.get("partial_occluder_excess_min", diag.get("partial_occluder_excess_min", 0.05)) or 0.05)
    mix_min = float(ck.get("partial_occluder_mix_min", diag.get("partial_occluder_mix_min", 0.020)) or 0.020)
    coh_max = float(ck.get("partial_occluder_coherence_max", diag.get("partial_occluder_coherence_max", 0.82)) or 0.82)
    solid_min = float(ck.get("beam_coherence_solid_min", diag.get("beam_coherence_solid_min", 0.45)) or 0.45)
    coherent_guard = float(ck.get("coherent_surface_guard", diag.get("coherent_surface_guard", 0.92)) or 0.92)
    edge_thr = float(diag.get("edge_threshold_used") or ck.get("edge_anti_min", 0.08) or 0.08)

    valid = hit & ret_valid & (coverage >= min_cov) & (hit_count >= min_hits)
    split_candidate = valid & (split > split_min)
    front_back_candidate = valid & (front > front_min) & (back > back_min)
    gap_candidate = valid & (gap > gap_min)
    gap_ratio_candidate = valid & (gap_ratio > gap_ratio_min)
    back_ratio_candidate = valid & (back_ratio > back_ratio_min)
    excess_candidate = valid & (excess > excess_min)
    mix_candidate = valid & (mix > mix_min)
    coherence_candidate = valid & (coherence < coh_max)
    coherent_guard_candidate = valid & (coherence > coherent_guard) & (split < max(split_min, 0.30)) & (gap_ratio < max(gap_ratio_min, 0.02))
    partial_candidate = (
        split_candidate & front_back_candidate & gap_candidate & gap_ratio_candidate
        & back_ratio_candidate & excess_candidate & mix_candidate
        & coherence_candidate & ~coherent_guard_candidate
    )
    geom_edge_candidate = hit & (edge_score > edge_thr)
    coherence_solid_candidate = valid & (coherence > solid_min) & (split < split_min) & (mix < max(0.04, mix_min * 2.0))
    split_or_back_candidate = valid & ((split > split_min) | (back > back_min))

    return {
        "hit": hit,
        "valid_return": valid,
        "split_candidate": split_candidate,
        "front_back_candidate": front_back_candidate,
        "gap_candidate": gap_candidate,
        "gap_ratio_candidate": gap_ratio_candidate,
        "back_ratio_candidate": back_ratio_candidate,
        "excess_candidate": excess_candidate,
        "mix_candidate": mix_candidate,
        "coherence_candidate": coherence_candidate,
        "coherent_guard_candidate": coherent_guard_candidate,
        "partial_candidate": partial_candidate,
        "geom_edge_candidate": geom_edge_candidate,
        "coherence_solid_candidate": coherence_solid_candidate,
        "split_or_back_candidate": split_or_back_candidate,
        "partial_geom_overlap_candidate": partial_candidate & geom_edge_candidate,
    }


def _mask_panel(mask: np.ndarray, title: str, size: Tuple[int, int] = (180, 124)) -> Image.Image:
    """Render one boolean mask panel for gate-debug sheets."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        m = np.zeros((size[1], size[0]), dtype=bool)
    img = Image.fromarray((m.astype(np.uint8) * 255), mode="L").convert("RGB")
    img = img.resize(size, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size[0], size[1] + 22), (245, 245, 245))
    panel.paste(img, (0, 22))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, size[0] - 1, 21], fill=(28, 32, 42))
    draw.text((5, 5), title[:32], fill=(255, 255, 255))
    return panel


def save_candidate_gate_sheet(channels: Dict[str, np.ndarray], diag: Dict[str, Any], out_path: Path) -> Optional[str]:
    """Save a compact visual sheet of alpha5.1 gate masks for one run."""
    masks = candidate_masks_from_channels(channels, diag)
    if not masks:
        return None
    ordered = [
        "hit", "valid_return", "split_candidate", "front_back_candidate",
        "gap_candidate", "gap_ratio_candidate", "back_ratio_candidate", "excess_candidate",
        "mix_candidate", "coherence_candidate", "coherent_guard_candidate", "partial_candidate",
        "geom_edge_candidate", "coherence_solid_candidate", "split_or_back_candidate", "partial_geom_overlap_candidate",
    ]
    panels = [_mask_panel(masks[k], k) for k in ordered if k in masks]
    if not panels:
        return None
    cols = 4
    w, h = panels[0].size
    rows = int(math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (230, 230, 230))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * w, (i // cols) * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def gate_breakdown_rows(beam_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Long-form gate counts/fracs for easier plotting and spreadsheet review."""
    gates = [
        "valid_return_candidate", "split_candidate", "front_back_candidate",
        "gap_candidate", "gap_ratio_candidate", "back_ratio_candidate",
        "excess_candidate", "coherent_guard_candidate", "partial_candidate",
        "geom_edge_candidate", "coherence_solid_candidate", "split_or_back_candidate",
        "partial_geom_overlap_candidate",
    ]
    rows: List[Dict[str, Any]] = []
    for r in beam_rows:
        for gate in gates:
            rows.append({
                "run_id": r.get("run_id"),
                "scene": r.get("scene"),
                "preset": r.get("preset"),
                "beam_width": r.get("beam_width"),
                "gate": gate,
                "count": r.get(f"{gate}_count"),
                "frac_of_hit_px": r.get(f"{gate}_frac_of_hit_px"),
                "hit_pixels": r.get("hit_pixels"),
                "partial_occluder_frac_of_hit_px": r.get("partial_occluder_frac_of_hit_px"),
                "solid_surface_frac_of_hit_px": r.get("solid_surface_frac_of_hit_px"),
                "uncertain_frac_of_hit_px": r.get("uncertain_frac_of_hit_px"),
            })
    return rows


def class_balance_rows(beam_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classes = ["geom_edge", "partial_occluder", "foliage", "solid_surface", "uncertain"]
    rows: List[Dict[str, Any]] = []
    for r in beam_rows:
        for cls in classes:
            rows.append({
                "run_id": r.get("run_id"),
                "scene": r.get("scene"),
                "preset": r.get("preset"),
                "class": cls,
                "count": r.get(f"{cls}_count"),
                "frac_of_hit_px": r.get(f"{cls}_frac_of_hit_px"),
                "hit_pixels": r.get("hit_pixels"),
            })
    return rows


def edge_summary_rows_from_beam_rows(beam_rows: List[Dict[str, Any]], all_diag: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Alpha5.1: build edge/gate summary from channel summaries, not only diagnostics.

    Earlier alpha5 diagnostics omitted some derived ratio percentiles from the
    compact edge summary even though the npz had those channels. This uses
    beam_rows so beam_back_ratio_p95 and beam_mode_gap_ratio_p95 are visible.
    """
    diag_by_run = {d.get("run_id"): d for d in all_diag}
    cols = [
        "run_id", "scene", "preset", "coverage", "hit_pixels",
        "edge_threshold_used", "edge_fusion_mode", "beam_profile", "beam_width",
        "geom_edge_frac_of_hit_px", "partial_occluder_frac_of_hit_px",
        "foliage_frac_of_hit_px", "solid_surface_frac_of_hit_px", "uncertain_frac_of_hit_px",
        "edge_score_geom_p95", "edge_score_raw_p95", "return_mix_score_p95",
        "return_depth_spread_excess_p95", "beam_split_score_p95",
        "beam_mode_gap_p95", "beam_mode_gap_ratio_p95", "beam_back_ratio_p95",
        "beam_coherence_p50", "beam_coherence_p95",
        "partial_candidate_count", "partial_candidate_frac_of_hit_px",
        "gap_ratio_candidate_count", "gap_ratio_candidate_frac_of_hit_px",
        "back_ratio_candidate_count", "back_ratio_candidate_frac_of_hit_px",
        "coherent_guard_candidate_count", "coherent_guard_candidate_frac_of_hit_px",
        "partial_geom_overlap_candidate_count", "partial_geom_overlap_candidate_frac_of_hit_px",
        "coherence_solid_candidate_count", "coherence_solid_candidate_frac_of_hit_px",
        "return_coverage_p50",
    ]
    rows = []
    for r in beam_rows:
        row = {k: r.get(k) for k in cols if k in r}
        d = diag_by_run.get(r.get("run_id"), {})
        for k in ["edge_score_mode", "edge_anti_min", "edge_anti_max", "adaptive_edge_percentile"]:
            if k in d and k not in row:
                row[k] = d.get(k)
        rows.append(row)
    return rows

def channel_summary(result: Dict[str, Any], run_id: int, scene: str, preset: str) -> Dict[str, Any]:
    ch = result.get("channels", {})
    diag = result.get("diagnostics", {})
    hit = np.asarray(ch.get("hit_count", np.zeros((1, 1)))) > 0
    total_px = int(hit.size)
    hit_px = int(hit.sum())
    labels = diag.get("classification_counts") or {}
    row = {
        "run_id": run_id,
        "scene": scene,
        "preset": preset,
        "coverage": float((diag.get("depth_stats") or {}).get("coverage", 0.0)),
        "hit_pixels": hit_px,
        "total_pixels": total_px,
        "geom_edge_count": int(labels.get("geom_edge", 0)),
        "partial_occluder_count": int(labels.get("partial_occluder", 0)),
        "foliage_count": int(labels.get("foliage", 0)),
        "solid_surface_count": int(labels.get("solid_surface", 0)),
        "uncertain_count": int(labels.get("uncertain", 0)),
        "geom_edge_frac_of_hit_px": float(int(labels.get("geom_edge", 0)) / max(hit_px, 1)),
        "partial_occluder_frac_of_hit_px": float(int(labels.get("partial_occluder", 0)) / max(hit_px, 1)),
        "foliage_frac_of_hit_px": float(int(labels.get("foliage", 0)) / max(hit_px, 1)),
        "solid_surface_frac_of_hit_px": float(int(labels.get("solid_surface", 0)) / max(hit_px, 1)),
        "uncertain_frac_of_hit_px": float(int(labels.get("uncertain", 0)) / max(hit_px, 1)),
        "beam_profile": diag.get("beam_profile"),
        "beam_width": diag.get("beam_width"),
        "edge_fusion_mode": diag.get("edge_fusion_mode"),
        "edge_threshold_used": diag.get("edge_threshold_used"),
    }
    row.update(candidate_counts_from_channels(ch, diag))
    for name in [
        "return_mix_score_raw", "return_mix_score",
        "return_depth_spread", "return_depth_spread_excess",
        "return_coverage", "return_valid_stats", "return_weight_sum",
        "return_expected_spread", "return_mix_support",
        "beam_split_score", "beam_split_support", "beam_front_depth",
        "beam_back_depth", "beam_front_strength", "beam_back_strength",
        "beam_back_ratio", "beam_mode_gap", "beam_mode_gap_ratio",
        "beam_mode_confidence", "beam_coherence",
        "edge_score_geom", "edge_score_raw"
    ]:
        for p in [50, 90, 95, 99]:
            row[f"{name}_p{p}"] = pct_from_channel(ch, hit, name, p)
    return row

def quick_verdict_rows(beam_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact one-row-per-run comparison sheet with alpha5.1 ratio diagnostics.

    This is the first file to open after a Colab run. It includes final class
    fractions plus the ratio gates that explain why partial_occluder was kept
    or suppressed.
    """
    rows: List[Dict[str, Any]] = []
    for r in beam_rows:
        uncertain = float(r.get("uncertain_frac_of_hit_px") or 0.0)
        solid = float(r.get("solid_surface_frac_of_hit_px") or 0.0)
        partial = float(r.get("partial_occluder_frac_of_hit_px") or 0.0)
        partial_cand = float(r.get("partial_candidate_frac_of_hit_px") or 0.0)
        geom = float(r.get("geom_edge_frac_of_hit_px") or 0.0)
        split95 = r.get("beam_split_score_p95")
        coherence50 = r.get("beam_coherence_p50")
        gap_ratio95 = r.get("beam_mode_gap_ratio_p95")
        back_ratio95 = r.get("beam_back_ratio_p95")
        coherent_guard = float(r.get("coherent_guard_candidate_frac_of_hit_px") or 0.0)

        status = "watch"
        notes = []
        if uncertain > 0.12:
            status = "too_uncertain"
            notes.append("uncertain above 12%")
        elif partial > 0.08:
            status = "too_partial"
            notes.append("partial above 8%; inspect candidate gates")
        elif solid > 0.75 and partial <= 0.04 and uncertain <= 0.06:
            status = "promising"
            notes.append("balanced solid/partial/uncertain")
        elif solid > 0.60:
            status = "usable"
            notes.append("solid recovery working")
        else:
            notes.append("mixed/needs visual review")

        if partial_cand > partial * 2.5 and partial_cand > 0.01:
            notes.append("partial candidates exceed final labels")
        if coherent_guard > 0.15:
            notes.append(f"coherent_guard_active={coherent_guard:.3f}")
        if split95 is not None:
            notes.append(f"split_p95={float(split95):.3f}")
        if gap_ratio95 is not None:
            notes.append(f"gap_ratio_p95={float(gap_ratio95):.3f}")
        if back_ratio95 is not None:
            notes.append(f"back_ratio_p95={float(back_ratio95):.3f}")
        if coherence50 is not None:
            notes.append(f"coherence_p50={float(coherence50):.3f}")

        rows.append({
            "run_id": r.get("run_id"),
            "scene": r.get("scene"),
            "preset": r.get("preset"),
            "beam_width": r.get("beam_width"),
            "coverage": r.get("coverage"),
            "geom_edge_frac_of_hit_px": geom,
            "partial_occluder_frac_of_hit_px": partial,
            "foliage_frac_of_hit_px": r.get("foliage_frac_of_hit_px"),
            "solid_surface_frac_of_hit_px": solid,
            "uncertain_frac_of_hit_px": uncertain,
            "partial_candidate_frac_of_hit_px": partial_cand,
            "partial_geom_overlap_candidate_frac_of_hit_px": r.get("partial_geom_overlap_candidate_frac_of_hit_px"),
            "split_candidate_frac_of_hit_px": r.get("split_candidate_frac_of_hit_px"),
            "front_back_candidate_frac_of_hit_px": r.get("front_back_candidate_frac_of_hit_px"),
            "gap_ratio_candidate_frac_of_hit_px": r.get("gap_ratio_candidate_frac_of_hit_px"),
            "back_ratio_candidate_frac_of_hit_px": r.get("back_ratio_candidate_frac_of_hit_px"),
            "coherent_guard_candidate_frac_of_hit_px": r.get("coherent_guard_candidate_frac_of_hit_px"),
            "coherence_solid_candidate_frac_of_hit_px": r.get("coherence_solid_candidate_frac_of_hit_px"),
            "return_mix_score_p95": r.get("return_mix_score_p95"),
            "return_depth_spread_excess_p95": r.get("return_depth_spread_excess_p95"),
            "beam_split_score_p95": split95,
            "beam_mode_gap_ratio_p95": gap_ratio95,
            "beam_back_ratio_p95": back_ratio95,
            "beam_coherence_p50": coherence50,
            "beam_coherence_p95": r.get("beam_coherence_p95"),
            "recommended_status": status,
            "notes": "; ".join(notes),
        })
    return rows


def flatten_counts(diag: Dict[str, Any], run_id: int, scene: str, preset: str) -> List[Dict[str, Any]]:
    counts = diag.get("classification_counts") or {}
    total = sum(int(v) for v in counts.values()) or 1
    return [
        {
            "run_id": run_id,
            "scene": scene,
            "preset": preset,
            "label": k,
            "count": int(v),
            "frac_of_all_pixels": float(int(v) / total),
        }
        for k, v in sorted(counts.items())
    ]


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_top_contact_sheets(contact_paths: List[str], out_path: Path, max_images: int = 8) -> Optional[str]:
    imgs = []
    for p in contact_paths[:max_images]:
        if p and os.path.exists(p):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            scale = min(560 / max(w, 1), 1.0)
            im = im.resize((int(w * scale), int(h * scale)))
            label = Path(p).parent.name
            canvas = Image.new("RGB", (im.width, im.height + 26), "white")
            canvas.paste(im, (0, 26))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 6), label, fill=(0, 0, 0))
            imgs.append(canvas)
    if not imgs:
        return None
    cols = 2 if len(imgs) > 1 else 1
    rows = int(math.ceil(len(imgs) / cols))
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, im in enumerate(imgs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def zip_outputs(out_dir: str | Path, zip_path: Optional[str | Path] = None) -> str:
    out_dir = Path(out_dir)
    if zip_path is None:
        zip_path = out_dir.with_suffix(".zip")
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir.parent))
    return str(zip_path)


def run_plan(engine_path: str, plan_path: str, out_dir: str = DEFAULT_OUT) -> Dict[str, Any]:
    llw = load_engine(engine_path)
    plan = load_plan(plan_path)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    all_diag: List[Dict[str, Any]] = []
    beam_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    material_report_rows: List[Dict[str, Any]] = []
    material_core_rows: List[Dict[str, Any]] = []
    material_filled_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    contact_paths: List[str] = []
    candidate_gate_paths: List[str] = []
    errors: List[Dict[str, Any]] = []

    for idx, row in plan.iterrows():
        run_id = _to_int(row.get("run_id"), len(all_diag) + 1) or (len(all_diag) + 1)
        scene = _to_str(row.get("scene"), "demo") or "demo"
        preset = _to_str(row.get("preset"), "beam_return_debug") or "beam_return_debug"
        seed = _to_int(row.get("seed"), 42) or 42
        run_label = f"run_{run_id:02d}_{_safe_name(scene)}_{_safe_name(preset)}"
        run_dir = out / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        overrides = row_overrides(row)

        print(f"\n=== run {run_id}: scene={scene} preset={preset} seed={seed} ===")
        print("overrides:", overrides)
        try:
            prims = scene_from_name(llw, scene)
            result = llw.run_sensor_preset(
                prims,
                preset_name=preset,
                scene_name=f"{_safe_name(scene)}_{run_id:02d}_{_safe_name(preset)}",
                out_dir=str(run_dir),
                seed=seed,
                **overrides,
            )

            diag = dict(result.get("diagnostics") or {})
            diag["run_id"] = run_id
            diag["input_scene"] = scene
            diag["input_preset"] = preset
            diag["contact_sheet"] = result.get("paths", {}).get("contact_sheet")
            diag["channels_npz"] = result.get("paths", {}).get("channels")
            diag["diagnostics_json"] = result.get("paths", {}).get("diagnostics")
            all_diag.append(diag)

            contact = result.get("paths", {}).get("contact_sheet")
            if contact:
                contact_paths.append(contact)

            gate_sheet = save_candidate_gate_sheet(
                result.get("channels", {}),
                diag,
                run_dir / f"{_safe_name(scene)}_{run_id:02d}_{_safe_name(preset)}_candidate_gates.png",
            )
            if gate_sheet:
                candidate_gate_paths.append(gate_sheet)
                diag["candidate_gate_sheet"] = gate_sheet

            beam_rows.append(channel_summary(result, run_id, scene, preset))
            class_rows.extend(flatten_counts(diag, run_id, scene, preset))

            sd = dict(result.get("structure_density") or {})
            sd.update({"run_id": run_id, "scene": scene, "preset": preset})
            structure_rows.append(sd)

            for rec in result.get("material_report") or []:
                rec = dict(rec); rec.update({"run_id": run_id, "scene": scene, "preset": preset})
                material_report_rows.append(rec)
            for rec in result.get("material_core_agreement") or []:
                rec = dict(rec); rec.update({"run_id": run_id, "scene": scene, "preset": preset})
                material_core_rows.append(rec)
            for rec in result.get("material_filled_agreement") or []:
                rec = dict(rec); rec.update({"run_id": run_id, "scene": scene, "preset": preset})
                material_filled_rows.append(rec)
            for rec in result.get("boundary_adjacency") or []:
                rec = dict(rec); rec.update({"run_id": run_id, "scene": scene, "preset": preset})
                boundary_rows.append(rec)

        except Exception as exc:
            err = {
                "run_id": run_id,
                "scene": scene,
                "preset": preset,
                "error": repr(exc),
            }
            errors.append(err)
            print("ERROR:", repr(exc))

    write_rows_csv(out / "sweep_metrics.csv", all_diag)
    write_rows_csv(out / "beam_return_summary.csv", beam_rows)
    write_rows_csv(out / "quick_verdict.csv", quick_verdict_rows(beam_rows))
    write_rows_csv(out / "gate_breakdown.csv", gate_breakdown_rows(beam_rows))
    write_rows_csv(out / "class_balance_summary.csv", class_balance_rows(beam_rows))
    write_rows_csv(out / "classification_counts.csv", class_rows)
    write_rows_csv(out / "structure_density_report.csv", structure_rows)
    write_rows_csv(out / "material_channel_report.csv", material_report_rows)
    write_rows_csv(out / "material_core_agreement.csv", material_core_rows)
    write_rows_csv(out / "material_filled_agreement.csv", material_filled_rows)
    write_rows_csv(out / "boundary_adjacency_report.csv", boundary_rows)

    edge_cols = [
        "preset", "input_scene", "run_id", "edge_threshold_used",
        "edge_score_p90", "edge_score_p95", "edge_score_p99",
        "edge_score_raw_p95", "edge_score_geom_p95",
        "return_mix_score_raw_p95", "return_mix_score_p95",
        "return_depth_spread_p95", "return_depth_spread_excess_p95",
        "beam_split_score_p95", "beam_mode_gap_p95",
        "beam_mode_gap_ratio_p95", "beam_back_ratio_p95",
        "beam_coherence_p50", "beam_coherence_p95",
        "partial_candidate_count", "partial_candidate_frac_of_hit_px",
        "geom_edge_candidate_count", "partial_geom_overlap_candidate_count",
        "return_coverage_p50", "edge_confidence_max",
        "beam_profile", "beam_width", "edge_score_mode", "edge_fusion_mode",
        "edge_anti_min", "edge_anti_max", "adaptive_edge_percentile",
    ]
    # Alpha5.1 reporting fix: use beam_rows so derived ratio channel
    # percentiles/candidate counts are present in the compact edge summary.
    edge_rows = edge_summary_rows_from_beam_rows(beam_rows, all_diag)
    write_rows_csv(out / "edge_threshold_diagnostics_summary.csv", edge_rows)

    if errors:
        write_rows_csv(out / "errors.csv", errors)

    top_path = make_top_contact_sheets(contact_paths, out / "top_contact_sheets.png")

    summary = {
        "engine": str(engine_path),
        "plan": str(plan_path),
        "out_dir": str(out),
        "runs_requested": int(len(plan)),
        "runs_completed": int(len(all_diag)),
        "runs_failed": int(len(errors)),
        "top_contact_sheets": top_path,
        "candidate_gate_sheets": candidate_gate_paths,
        "csv_outputs": [
            "sweep_metrics.csv",
            "beam_return_summary.csv",
            "quick_verdict.csv",
            "gate_breakdown.csv",
            "class_balance_summary.csv",
            "classification_counts.csv",
            "edge_threshold_diagnostics_summary.csv",
            "structure_density_report.csv",
            "material_channel_report.csv",
            "material_core_agreement.csv",
            "material_filled_agreement.csv",
            "boundary_adjacency_report.csv",
        ],
    }
    with (out / "harness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nHarness summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None, help="Engine .py file. If omitted, auto-detects lidar_lenses_wave_v080*.py.")
    parser.add_argument("--plan", default=None, help="Spreadsheet plan .xlsx. If omitted, auto-detects *test_plan*.xlsx.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--zip", dest="zip_path", default=None)
    args = parser.parse_args(argv)

    engine = args.engine or find_first(["lidar_lenses_wave_v080_alpha5_1.py", "lidar_lenses_wave_v080*.py", "lidar_lenses_wave_v*.py"], "engine .py")
    plan = args.plan or find_first(["lidar_wave_test_plan_v080_alpha5_1.xlsx", "*test_plan*.xlsx", "*.xlsx"], "test plan .xlsx")
    print("ENGINE:", engine)
    print("PLAN:", plan)
    summary = run_plan(engine, plan, args.out)
    zip_path = zip_outputs(args.out, args.zip_path)
    print("\nZIP:", zip_path)
    return {"summary": summary, "zip_path": zip_path}


if __name__ == "__main__":
    main()
