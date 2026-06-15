#!/usr/bin/env python3
"""
lidar_studio_camera.py — camera/lens-control copy of the LiDAR Studio front-end.

Unlike the Tier-2 HTML demos (which re-implement a thin JS approximation), this
front-end drives the REAL Python engine. You point a browser at it, pick a
scene — a built-in demo OR your own uploaded 3D model — choose a sensor preset,
and it runs the full multimodal pipeline (depth / coherence / acoustic /
ultrasonic / polarization / material classification / boundary-aware edges) and
shows you the engine's actual diagnostic contact sheet and reports.

This is the "true power" view: the same `run_sensor_preset()` call the test
harnesses use, wired to a browser with model upload.

Usage:
    python lidar_studio_camera.py
    python lidar_studio_camera.py --engine lidar_lenses_wave_v070.py --port 8080

Then open http://localhost:8080 in a browser.

Supported uploads:
    .stl  (binary or ASCII)        — via the engine's native loader
    .obj  (Wavefront, triangulated) — via a tiny built-in parser

No third-party web dependencies: only Python's standard library plus whatever
the engine itself needs (numpy, Pillow). Everything is one file.

SECURITY NOTE
-------------
The /load_engine and /upload_engine endpoints execute arbitrary Python by
design (they import an engine module). That is fine for a single local user —
you already have a shell on the box — but it is a remote-code-execution hole
the moment the server is reachable from the network. Therefore:

  * The server REFUSES to bind to a non-loopback host unless you pass
    --allow-remote (you have to opt in to network exposure explicitly).
  * The engine-loading endpoints are enabled automatically on loopback, but on
    a remote bind they stay DISABLED unless you also pass --allow-engine-load.
  * Request bodies are capped (--max-upload-mb, default 200) so a client can't
    exhaust memory just by claiming a huge Content-Length.

Even with both flags set, only expose this to networks/users you trust.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import importlib.util
import ipaddress
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# Default to the latest v0.8 alpha engine if present, else the v0.7.0 stable one.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ENGINES = [
    os.path.join(_HERE, "lidar_v080_alpha5_1_beam_evidence_lite",
                 "lidar_lenses_wave_v080_alpha5_1.py"),
    os.path.join(_HERE, "lidar_lenses_wave_v070.py"),
]

ENG = None          # the loaded engine module
ENGINE_PATH = None  # path it was loaded from

# Set in main(): whether the load/upload-engine (RCE) endpoints are reachable,
# and the hard cap on request body size in bytes.
ALLOW_ENGINE_LOAD = True
MAX_BODY_BYTES = 200 * 1024 * 1024

# Serializes engine use: prevents concurrent scans and engine swaps mid-scan
# from racing on the shared global ENG (the server is multi-threaded).
_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
# Host / safety helpers
# ──────────────────────────────────────────────────────────────────────────
def is_loopback_host(host: str) -> bool:
    """True if `host` resolves only to loopback addresses (so the server is not
    reachable off the machine). Unresolvable / odd hosts are treated as NOT
    loopback, i.e. the safe-by-default answer."""
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
        addrs = {info[4][0] for info in infos}
        return bool(addrs) and all(
            ipaddress.ip_address(a).is_loopback for a in addrs)
    except socket.gaierror:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Engine loading
# ──────────────────────────────────────────────────────────────────────────
def load_engine(path: str):
    spec = importlib.util.spec_from_file_location("llw_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import engine from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llw_engine"] = mod
    spec.loader.exec_module(mod)
    patch_engine(mod)
    return mod


def patch_engine(eng):
    """Make the engine's diagnostics Scene/mesh-aware so an uploaded mesh can
    flow through the SAME pipeline as the built-in primitive demos.

    This is a small, additive compatibility shim applied at runtime to whatever
    engine is loaded — the engine source files are left untouched. The engine
    already raycasts meshes (BVH + Möller-Trumbore) and ``Scene`` already holds
    a ``meshes`` list; the only gaps are that the primitive-oriented diagnostics
    iterate a plain ``prims`` list and that camera auto-framing
    (``scene_bounds``) only knew about primitives.

    Module-level functions resolve ``scene_bounds`` from the module globals at
    call time, so replacing ``eng.scene_bounds`` is picked up by
    ``make_camera_for_preset`` and friends. Dunder methods are looked up on the
    type, so assigning to the class works for ``Scene``/``Mesh`` too.
    """
    if not (hasattr(eng, "Scene") and hasattr(eng, "Mesh")):
        return  # not a LiDAR engine; set_engine() will report this clearly.
    Scene, Mesh = eng.Scene, eng.Mesh

    # 1) Scene is iterable + sized: yields primitives then meshes as "pieces",
    #    so `for p in prims` / `len(prims)` work when handed a whole Scene.
    if "__iter__" not in vars(Scene):
        def _scene_iter(self):
            yield from self.primitives
            yield from self.meshes
        Scene.__iter__ = _scene_iter
    if "__len__" not in vars(Scene):
        def _scene_len(self):
            return len(self.primitives) + len(self.meshes)
        Scene.__len__ = _scene_len

    # 2) Mesh reports a shape ("mesh") and a center, so the shape-based acoustic
    #    / polarization heuristics fall through to their neutral default while an
    #    explicit piece_type still drives MATERIAL_PRIORS.
    if not hasattr(Mesh, "shape"):
        Mesh.shape = property(lambda self: "mesh")
    if not hasattr(Mesh, "center"):
        Mesh.center = property(lambda self: (self.aabb_min + self.aabb_max) / 2.0)

    # 3) scene_bounds understands mesh AABBs (and accepts a Scene, now iterable).
    #    Guard against double-wrapping: only wrap an engine's own scene_bounds,
    #    never a shim we've already installed (matters if patch_engine is ever
    #    re-run on the same module object).
    if not getattr(eng.scene_bounds, "_llw_mesh_aware", False):
        _orig_scene_bounds = eng.scene_bounds

        def _scene_bounds(prims):
            meshes = [p for p in prims if getattr(p, "shape", None) == "mesh"]
            if not meshes:
                return _orig_scene_bounds(prims)
            prim_only = [p for p in prims if getattr(p, "shape", None) != "mesh"]
            lows = [np.asarray(m.aabb_min, dtype=np.float64) for m in meshes]
            highs = [np.asarray(m.aabb_max, dtype=np.float64) for m in meshes]
            if prim_only:
                pb = _orig_scene_bounds(prim_only)
                lows.append(pb["min"]); highs.append(pb["max"])
            mn = np.vstack(lows).min(axis=0)
            mx = np.vstack(highs).max(axis=0)
            center = (mn + mx) / 2.0
            span = np.maximum(mx - mn, 1e-6)
            return {"min": mn, "max": mx, "center": center, "span": span}

        _scene_bounds._llw_mesh_aware = True
        eng.scene_bounds = _scene_bounds


# ──────────────────────────────────────────────────────────────────────────
# Model parsing → engine Mesh
# ──────────────────────────────────────────────────────────────────────────
def parse_obj(text: str, color=(0.72, 0.72, 0.75), piece_id: int = 1000):
    """Minimal Wavefront OBJ → engine Mesh. Triangulates polygon faces (fan),
    ignores texture/normal indices and everything that isn't v/f."""
    verts = []
    faces = []
    for line in text.splitlines():
        parts = line.split()  # splits on any whitespace incl. tabs
        if not parts:
            continue
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "f" and len(parts) >= 4:
            idx = []
            for t in parts[1:]:
                raw = t.split("/")[0]
                if raw == "":
                    continue
                i = int(raw)
                # OBJ indices are 1-based; negatives are relative to verts so far.
                idx.append((i - 1) if i > 0 else (len(verts) + i))
            for k in range(1, len(idx) - 1):  # fan triangulation
                faces.append([idx[0], idx[k], idx[k + 1]])
    if len(verts) < 3 or len(faces) < 1:
        raise ValueError("OBJ had no usable triangles (need 'v' and 'f' lines)")
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    # Bounds-check faces before handing them to the engine: a face referencing a
    # vertex that doesn't exist would otherwise surface as a confusing
    # out-of-range error deep inside raycasting.
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError(
            f"OBJ has face indices out of range (verts={len(V)}, "
            f"face index range {int(F.min())}..{int(F.max())})")
    return ENG.Mesh(vertices=V, faces=F, color=color, piece_id=piece_id)


def load_mesh_from_upload(filename: str, raw: bytes, piece_id: int = 1000):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".obj":
        return parse_obj(raw.decode("utf-8", errors="replace"), piece_id=piece_id)
    if ext == ".stl":
        # The engine's STL loader takes a path; hand it a temp file.
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
            tf.write(raw)
            tmp = tf.name
        try:
            return ENG.load_stl(tmp, color=(0.72, 0.72, 0.75), piece_id=piece_id)
        finally:
            os.unlink(tmp)
    raise ValueError(f"unsupported file type {ext!r} (use .stl or .obj)")


def normalize_mesh(mesh, target_size: float = 6.0):
    """Recenter on XZ, drop the base to y=0, and scale so the largest dimension
    is ~target_size metres. Keeps arbitrary-unit models inside the range/eye
    assumptions the presets are tuned for."""
    mn = mesh.aabb_min.astype(np.float64)
    mx = mesh.aabb_max.astype(np.float64)
    center = (mn + mx) / 2.0
    size = float(np.max(mx - mn))
    scale = target_size / size if size > 1e-9 else 1.0
    V = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale
    V[:, 1] -= V[:, 1].min()  # base sits on the ground plane
    return ENG.Mesh(vertices=V, faces=mesh.faces, color=mesh.color,
                    piece_id=mesh.piece_id, piece_type=mesh.piece_type)


def _cluster_decimate(V, F, grid_n):
    """One vertex-clustering pass: snap vertices to a grid_n³ grid, collapse each
    occupied cell to its centroid, remap faces, drop degenerate triangles.
    Returns (new_vertices, new_faces)."""
    mn = V.min(axis=0)
    ext = float(np.max(V.max(axis=0) - mn)) or 1.0
    cell = ext / grid_n
    ijk = np.floor((V - mn) / cell).astype(np.int64)
    # Pack the 3 cell indices into one key for a fast 1-D unique.
    span = int(ijk.max()) + 2
    key = (ijk[:, 0] * span + ijk[:, 1]) * span + ijk[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    nC = int(inv.max()) + 1
    newV = np.zeros((nC, 3), dtype=np.float64)
    counts = np.zeros(nC, dtype=np.float64)
    np.add.at(newV, inv, V)
    np.add.at(counts, inv, 1.0)
    newV /= counts[:, None]
    newF = inv[F]  # remap each face's vertex ids to its cluster ids
    good = ((newF[:, 0] != newF[:, 1]) &
            (newF[:, 1] != newF[:, 2]) &
            (newF[:, 0] != newF[:, 2]))
    return newV, newF[good]


def decimate_mesh(mesh, target_tris: int):
    """Reduce a mesh toward `target_tris` via vertex clustering (no external
    deps). Binary-searches the grid resolution for the largest triangle count
    that stays under the budget. Returns mesh unchanged if already small."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if F.shape[0] <= target_tris:
        return mesh
    lo, hi, best = 4, 1024, None
    for _ in range(10):
        mid = (lo + hi) // 2
        nv, nf = _cluster_decimate(V, F, mid)
        if nf.shape[0] > target_tris:
            hi = mid                       # too fine → fewer cells
        else:
            best = (nv, nf); lo = mid      # fits budget → try finer
        if hi - lo <= 1:
            break
    if best is None:                       # even the coarsest pass overshot
        best = _cluster_decimate(V, F, lo)
    nv, nf = best
    if nf.shape[0] < 4:                    # pathological — keep the original
        return mesh
    return ENG.Mesh(vertices=nv, faces=nf, color=mesh.color,
                    piece_id=mesh.piece_id, piece_type=mesh.piece_type)


# ──────────────────────────────────────────────────────────────────────────
# Scene building + scan
# ──────────────────────────────────────────────────────────────────────────
def build_scene(req: dict):
    """Return (scene_or_prims, scene_name, info_dict)."""
    kind = req.get("scene", "demo")
    if kind == "demo":
        return ENG._build_demo_scene(), "cabin_demo", {"source": "built-in cabin demo"}
    if kind == "material_board":
        return (ENG._build_material_target_board_scene(), "material_targets",
                {"source": "built-in material target board"})
    if kind == "upload":
        filename = req.get("filename", "model.stl")
        data = base64.b64decode(req["data_b64"])
        mesh = load_mesh_from_upload(filename, data)
        orig_tris = int(mesh.faces.shape[0])
        # Big meshes are slow to raycast; simplify toward a triangle budget.
        simplified_to = None
        if req.get("simplify", True):
            try:
                target = int(req.get("max_tris") or 40000)
            except (TypeError, ValueError):
                target = 40000
            target = max(1000, min(500000, target))
            if orig_tris > target:
                mesh = decimate_mesh(mesh, target)
                simplified_to = int(mesh.faces.shape[0])
        material = req.get("material", "auto")
        if material and material != "auto":
            mesh.piece_type = material
        if req.get("normalize", True):
            mesh = normalize_mesh(mesh)
        info = {
            "source": f"uploaded {filename}",
            "triangles": int(mesh.faces.shape[0]),
            "triangles_original": orig_tris,
            "simplified": simplified_to is not None,
            "vertices": int(mesh.vertices.shape[0]),
            "material": material,
            "bounds_min": [round(float(x), 3) for x in mesh.aabb_min],
            "bounds_max": [round(float(x), 3) for x in mesh.aabb_max],
        }
        return ENG.Scene(meshes=[mesh]), "uploaded_model", info
    raise ValueError(f"unknown scene kind {kind!r}")


def _fast_frame_camera(scene, preset, probe_w: int = 44, probe_h: int = 30):
    """Cheap stand-in for the engine's (slow) auto-framer.

    Fires a few low-res candidate bursts on a ring around the scene and picks
    the tightest framing that still keeps a margin (highest coverage that stays
    under a cap, so the object fills the frame without clipping). Returns
    (position, target) as lists, or (None, None) to fall back to defaults.
    """
    try:
        b = ENG.scene_bounds(scene)
        c = np.asarray(b["center"], dtype=np.float64)
        radius = float(np.linalg.norm(b["span"])) * 0.5
        if radius <= 0:
            return None, None
        best = None          # (coverage, position) among un-clipped candidates
        fallback = None      # farthest candidate, if everything looks clipped
        for dk in (1.0, 1.3, 1.7, 2.2):
            for el in (0.45, 0.9):
                for az in (0.0, 1.5708, 3.1416, 4.7124):
                    d = np.array([np.cos(az), el, np.sin(az)], dtype=np.float64)
                    d /= np.linalg.norm(d)
                    pos = c + d * radius * dk
                    cam = ENG.make_camera_for_preset(
                        scene, preset, width=probe_w, height=probe_h,
                        position=pos.tolist(), target=c.tolist())
                    cov = ENG.fire_burst(cam, scene, probe_w * probe_h, 0).coverage
                    if fallback is None or dk > fallback[0]:
                        fallback = (dk, pos)
                    if cov <= 0.92 and (best is None or cov > best[0]):
                        best = (cov, pos)
        pos = (best[1] if best is not None else fallback[1])
        return pos.tolist(), c.tolist()
    except Exception:
        return None, None  # any trouble → let the preset's default camera stand


def _png_b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def run_scan(req: dict) -> dict:
    if ENG is None:
        raise RuntimeError("no engine loaded — set one in the Engine box (or start with --engine)")
    scene, scene_name, info = build_scene(req)

    preset = req.get("preset", "full_diagnostic")
    if preset not in ENG.SENSOR_PRESETS:
        raise ValueError(f"unknown preset {preset!r}")

    overrides = {}
    for key, cast in (("width", int), ("height", int), ("rays_per_pixel", int)):
        if req.get(key) not in (None, ""):
            overrides[key] = cast(req[key])
    manual_camera = bool(req.get("manual_camera", False))
    if manual_camera:
        def vec3(prefix):
            vals = []
            for axis in "xyz":
                raw = req.get(f"{prefix}_{axis}")
                if raw in (None, ""):
                    raise ValueError(f"manual camera requires {prefix}_{axis}")
                vals.append(float(raw))
            return vals

        overrides["position"] = vec3("cam_pos")
        overrides["target"] = vec3("cam_tgt")
        up_vals = [req.get(f"cam_up_{axis}") for axis in "xyz"]
        if any(v not in (None, "") for v in up_vals):
            overrides["up"] = vec3("cam_up")

        lens = (req.get("lens") or "").strip()
        if lens:
            overrides["lens"] = lens
        for key in ("fov_deg", "fisheye_fov_deg", "ortho_size"):
            if req.get(key) not in (None, ""):
                overrides[key] = float(req[key])
        # Manual composition should mean "capture exactly this view"; do not let
        # preset auto-framing replace the requested camera later in the engine.
        overrides["auto_frame"] = False
        if bool(req.get("fast", True)):
            overrides["stack"] = 1
    # Keep browser scans responsive: clamp resolution.
    if "width" in overrides:
        overrides["width"] = max(64, min(800, overrides["width"]))
    if "height" in overrides:
        overrides["height"] = max(64, min(600, overrides["height"]))
    if "rays_per_pixel" in overrides:
        overrides["rays_per_pixel"] = max(1, min(16, overrides["rays_per_pixel"]))
    # Fast preview: auto-framing (many candidate-camera casts) and burst stacking
    # dominate runtime — on a decimated heavy mesh they turn a 0.6s scan into 95s.
    # Skipping them keeps every channel, just unframed + single-burst (noisier).
    fast = bool(req.get("fast", True))
    if fast and not manual_camera:
        overrides["auto_frame"] = False
        overrides["stack"] = 1
        # The engine's auto-framing produces the good views but is the dominant
        # cost (~90s on a heavy mesh). We replace it with a cheap framing search
        # of our own — a handful of low-res candidate bursts — and hand the
        # winning camera to the engine with auto_frame off. This matters for the
        # built-in scenes too: compact_diagnostic frames them via auto_frame, so
        # without a substitute camera the unframed default barely sees them.
        pos, tgt = _fast_frame_camera(scene, preset)
        if pos is not None:
            overrides["position"] = pos
            overrides["target"] = tgt
    try:
        seed = int(req.get("seed") or 42)  # empty field / missing → default
    except (TypeError, ValueError):
        seed = 42

    with tempfile.TemporaryDirectory() as out_dir:
        result = ENG.run_sensor_preset(
            scene, preset_name=preset, scene_name=scene_name,
            out_dir=out_dir, seed=seed, **overrides,
        )
        paths = result["paths"]
        images = {}
        for label, key in (("Contact sheet", "contact_sheet"),
                           ("Geom edge overlay", "geom_edge_overlay"),
                           ("Material core", "material_core"),
                           ("Material filled", "material_filled")):
            p = paths.get(key)
            if p and os.path.exists(p):
                images[label] = _png_b64(p)

    diag = result["diagnostics"]
    return {
        "ok": True,
        "scene_info": info,
        "preset": preset,
        "preset_description": diag.get("description", ""),
        "images": images,
        "stats": {
            "width": diag.get("width"),
            "height": diag.get("height"),
            "lens": diag.get("lens"),
            "rays_per_pixel": diag.get("rays_per_pixel"),
            "camera_position": [round(x, 2) for x in diag.get("camera_position", [])],
            "depth_stats": {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in (diag.get("depth_stats") or {}).items()},
            "total_runtime_seconds": diag.get("total_runtime_seconds"),
            "fast": fast,
            "manual_camera": manual_camera,
            "camera_target": [round(x, 2) for x in diag.get("camera_target", [])],
        },
        "classification_counts": diag.get("classification_counts") or {},
        "material_report": result.get("material_report") or [],
        "warnings": diag.get("warnings") or [],
    }


# ──────────────────────────────────────────────────────────────────────────
# HTML page
# ──────────────────────────────────────────────────────────────────────────
def set_engine(path: str):
    """Load `path` as the active engine, validating it looks like one. Used both
    at startup and by the /load_engine endpoint when no engine was auto-found."""
    global ENG, ENGINE_PATH
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"no such file: {p}")
    mod = load_engine(p)
    if not (hasattr(mod, "SENSOR_PRESETS") and hasattr(mod, "run_sensor_preset")
            and hasattr(mod, "Scene") and hasattr(mod, "Mesh")):
        raise ValueError(f"{os.path.basename(p)} doesn't look like a LiDAR Lenses "
                         "Wave engine (missing SENSOR_PRESETS / run_sensor_preset / Scene / Mesh)")
    ENG, ENGINE_PATH = mod, p
    return mod


_ENGINE_UPLOAD_DIR = None


def _cleanup_engine_upload_dir():
    """Remove the temp dir that holds UI-uploaded engines, if any. Registered
    with atexit so uploaded engines don't accumulate for the process lifetime."""
    global _ENGINE_UPLOAD_DIR
    if _ENGINE_UPLOAD_DIR and os.path.isdir(_ENGINE_UPLOAD_DIR):
        shutil.rmtree(_ENGINE_UPLOAD_DIR, ignore_errors=True)
    _ENGINE_UPLOAD_DIR = None


def save_and_set_engine(filename: str, raw: bytes):
    """Persist an uploaded engine .py to a temp dir and load it — the upload
    equivalent of set_engine(), mirroring how models are uploaded."""
    global _ENGINE_UPLOAD_DIR
    if _ENGINE_UPLOAD_DIR is None:
        _ENGINE_UPLOAD_DIR = tempfile.mkdtemp(prefix="lidar_studio_engine_")
    name = os.path.basename(filename or "engine.py") or "engine.py"
    if not name.endswith(".py"):
        name += ".py"
    dest = os.path.join(_ENGINE_UPLOAD_DIR, name)
    with open(dest, "wb") as f:
        f.write(raw)
    return set_engine(dest)


def engine_meta() -> dict:
    """Current engine status + the option lists the page needs. Works whether or
    not an engine is loaded, so the UI can offer a loader when none was found."""
    loaded = ENG is not None
    presets = sorted(getattr(ENG, "SENSOR_PRESETS", {}).keys()) if loaded else []
    materials = ["auto"] + sorted(getattr(ENG, "MATERIAL_PRIORS", {}).keys()) if loaded else ["auto"]
    return {
        "loaded": loaded,
        "engine": os.path.basename(ENGINE_PATH) if ENGINE_PATH else None,
        "engine_path": ENGINE_PATH,
        "presets": presets,
        "materials": materials,
        "candidates": [c for c in _DEFAULT_ENGINES],
        # Lets the UI hide the engine loader when the endpoints are disabled.
        "engine_load_allowed": ALLOW_ENGINE_LOAD,
    }


def index_html() -> str:
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR Lenses Wave — Camera Studio</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 system-ui, sans-serif; background:#0c1018; color:#dfe6f0; }
  header { padding:16px 22px; border-bottom:1px solid #1d2636; background:#0f1420; }
  header h1 { margin:0; font-size:18px; letter-spacing:.3px; }
  header .sub { color:#7f8da3; font-size:12px; margin-top:3px; }
  header .sub code { color:#9fd0ff; }
  .wrap { display:flex; gap:0; align-items:flex-start; }
  .panel { width:340px; min-width:340px; padding:18px 20px; border-right:1px solid #1d2636;
           position:sticky; top:0; height:100vh; overflow:auto; }
  .out { flex:1; padding:18px 22px; min-width:0; }
  label { display:block; margin:12px 0 4px; font-size:12px; color:#9aa7bd; text-transform:uppercase; letter-spacing:.4px; }
  select, input[type=number], input[type=file], input[type=text] { width:100%; padding:8px 9px; background:#121a28;
           border:1px solid #26324a; border-radius:7px; color:#e8eef7; font-size:14px; }
  .row { display:flex; gap:10px; } .row > div { flex:1; }
  .seg { display:flex; gap:6px; margin-top:4px; flex-wrap:wrap; }
  .seg button { flex:1; padding:8px 6px; background:#121a28; border:1px solid #26324a; color:#cdd7e8;
           border-radius:7px; cursor:pointer; font-size:13px; }
  .seg button.active { background:#1b4fa0; border-color:#3f7ad6; color:#fff; }
  #uploadBox { display:none; }
  .chk { display:flex; align-items:center; gap:8px; margin-top:10px; color:#cdd7e8; text-transform:none; letter-spacing:0; }
  .chk input { width:auto; }
  #go { width:100%; margin-top:18px; padding:12px; font-size:15px; font-weight:600; cursor:pointer;
        background:#2563c9; border:0; border-radius:8px; color:#fff; }
  #go:disabled { opacity:.5; cursor:wait; }
  .hint { color:#6f7d93; font-size:11px; margin-top:6px; }
  .card { background:#0f1623; border:1px solid #1d2636; border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .card h3 { margin:0 0 10px; font-size:13px; color:#9fd0ff; text-transform:uppercase; letter-spacing:.5px; }
  .imgs { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
  .imgcard { background:#0f1623; border:1px solid #1d2636; border-radius:10px; overflow:hidden; }
  .imgcard .cap { padding:7px 10px; font-size:12px; color:#9aa7bd; border-bottom:1px solid #1d2636; }
  .imgcard img { width:100%; display:block; background:#05080d; cursor:zoom-in; }
  .imgcard.sheet { grid-column:1/-1; }
  table { border-collapse:collapse; width:100%; font-size:12px; }
  th, td { text-align:left; padding:4px 8px; border-bottom:1px solid #18202e; }
  th { color:#7f8da3; font-weight:600; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:13px; }
  .kv .k { color:#7f8da3; } .kv .v { color:#e8eef7; }
  .pill { display:inline-block; padding:2px 9px; border-radius:20px; font-size:11px; margin:2px 4px 2px 0; }
  .err { background:#3a1216; border:1px solid #7a2630; color:#ffb4bc; padding:12px 14px; border-radius:8px; white-space:pre-wrap; }
  .empty { color:#5b6679; padding:40px 10px; text-align:center; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #fff6; border-top-color:#fff;
          border-radius:50%; animation:spin .7s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  dialog { border:0; background:transparent; padding:0; max-width:96vw; max-height:96vh; }
  dialog img { max-width:96vw; max-height:96vh; }
  dialog::backdrop { background:#000c; }
</style>
</head>
<body>
<header>
  <h1>LiDAR Lenses Wave — Camera Studio</h1>
  <div class="sub">Live front-end driving the real engine: <code id="engineName">…</code>. Pick a scene or upload your own 3D model, then run the full sensor pipeline.</div>
</header>
<div class="wrap">
  <div class="panel">
    <div id="engineStatus" class="hint" style="margin-bottom:6px">
      Engine: <b id="engCur" style="color:#9fd0ff">…</b>
      <a id="engToggle" style="color:#7fa8e0;cursor:pointer;margin-left:6px">change</a>
    </div>
    <div id="engineBox" style="display:none">
      <label>Upload engine file (.py)</label>
      <input type="file" id="engineFile" accept=".py">
      <button id="uploadEngine" style="width:100%;margin-top:8px;padding:9px;background:#2a3550;border:1px solid #3f4f74;border-radius:7px;color:#dfe6f0;cursor:pointer">Upload &amp; load engine</button>
      <label style="margin-top:14px">…or load by path on the server</label>
      <input type="text" id="enginePath" placeholder="/path/to/lidar_lenses_wave_v0xx.py"
             style="width:100%;padding:8px 9px;background:#121a28;border:1px solid #26324a;border-radius:7px;color:#e8eef7;font-size:13px">
      <button id="loadEngine" style="width:100%;margin-top:8px;padding:9px;background:#2a3550;border:1px solid #3f4f74;border-radius:7px;color:#dfe6f0;cursor:pointer">Load by path</button>
      <div class="hint" id="engineHint" style="margin-top:8px"></div>
      <div class="hint" id="engineErr" style="color:#ffb4bc"></div>
      <hr style="border:0;border-top:1px solid #1d2636;margin:16px 0">
    </div>
    <label>Scene</label>
    <div class="seg" id="sceneSeg">
      <button data-scene="demo" class="active">Cabin demo</button>
      <button data-scene="material_board">Material board</button>
      <button data-scene="upload">Upload model</button>
    </div>
    <div id="uploadBox">
      <label>3D model (.stl / .obj)</label>
      <input type="file" id="file" accept=".stl,.obj">
      <div class="hint">Binary or ASCII STL, or triangulated Wavefront OBJ.</div>
      <label>Material assumption</label>
      <select id="material"></select>
      <div class="hint">Drives acoustic / ultrasonic / polarization priors. "auto" = neutral mesh defaults.</div>
      <label class="chk"><input type="checkbox" id="normalize" checked> Normalize size &amp; drop to ground</label>
      <label class="chk"><input type="checkbox" id="simplify" checked> Simplify large meshes</label>
      <label>Max triangles</label>
      <input type="number" id="maxTris" value="40000" min="1000" max="500000" step="1000">
      <div class="hint">Heavy meshes are slow to raycast. Above this budget the model is decimated (vertex clustering) before scanning.</div>
    </div>

    <label>Sensor preset</label>
    <select id="preset"></select>

    <div class="row">
      <div><label>Width</label><input type="number" id="width" value="480" min="64" max="800"></div>
      <div><label>Height</label><input type="number" id="height" value="320" min="64" max="600"></div>
    </div>
    <div class="row">
      <div><label>Rays / px</label><input type="number" id="rays" value="6" min="1" max="16"></div>
      <div><label>Seed</label><input type="number" id="seed" value="42"></div>
    </div>

    <label class="chk"><input type="checkbox" id="fast" checked> Fast preview (no auto-frame, single burst)</label>

    <label class="chk"><input type="checkbox" id="manualCamera"> Manual camera / lens capture</label>
    <div id="cameraBox" style="display:none">
      <div class="hint">When enabled, Studio sends this exact camera to the engine and disables preset auto-framing.</div>
      <label>Lens</label>
      <select id="lens">
        <option value="pinhole">pinhole</option>
        <option value="telephoto">telephoto</option>
        <option value="fisheye">fisheye</option>
        <option value="orthographic">orthographic</option>
        <option value="equirectangular">equirectangular</option>
      </select>
      <div class="row">
        <div><label>FOV</label><input type="number" id="fov" value="60" min="1" max="180" step="1"></div>
        <div><label>Fisheye FOV</label><input type="number" id="fisheyeFov" value="180" min="1" max="360" step="1"></div>
      </div>
      <label>Ortho size</label>
      <input type="number" id="orthoSize" value="8" min="0.1" max="1000" step="0.1">
      <label>Position</label>
      <div class="row">
        <div><input type="number" id="camPosX" value="0" step="0.1" title="position x"></div>
        <div><input type="number" id="camPosY" value="1.65" step="0.1" title="position y"></div>
        <div><input type="number" id="camPosZ" value="6" step="0.1" title="position z"></div>
      </div>
      <label>Target</label>
      <div class="row">
        <div><input type="number" id="camTgtX" value="0" step="0.1" title="target x"></div>
        <div><input type="number" id="camTgtY" value="1" step="0.1" title="target y"></div>
        <div><input type="number" id="camTgtZ" value="0" step="0.1" title="target z"></div>
      </div>
      <label>Up vector</label>
      <div class="row">
        <div><input type="number" id="camUpX" value="0" step="0.1" title="up x"></div>
        <div><input type="number" id="camUpY" value="1" step="0.1" title="up y"></div>
        <div><input type="number" id="camUpZ" value="0" step="0.1" title="up z"></div>
      </div>
    </div>

    <button id="go">Run scan</button>
    <div class="hint">Fast preview is far quicker on heavy meshes. Uncheck for an auto-framed, multi-burst high-quality render (much slower).</div>
  </div>

  <div class="out" id="out">
    <div class="empty">Pick a scene and press <b>Run scan</b> to see the engine's diagnostic output.</div>
  </div>
</div>
<dialog id="zoom"><img id="zoomImg" src=""></dialog>

<script>
const $ = id => document.getElementById(id);
let scene = "demo";
let engineLoaded = false;
let engineLoadAllowed = true;

function applyMeta(m) {
  engineLoaded = !!m.loaded;
  engineLoadAllowed = m.engine_load_allowed !== false;
  $("engineName").textContent = m.engine || "none loaded";
  $("engCur").textContent = m.engine || "none loaded";
  // preset + material option lists
  $("preset").innerHTML = m.presets.map(p =>
    `<option value="${p}"${p==="full_diagnostic"?" selected":""}>${p}</option>`).join("");
  $("material").innerHTML = m.materials.map(x => `<option value="${x}">${x}</option>`).join("");
  // The loader is auto-open when nothing is loaded (and loading is allowed),
  // collapsible once one is. When loading is disabled (remote bind without
  // --allow-engine-load) the loader is hidden and the change link suppressed.
  const showLoader = engineLoadAllowed && !engineLoaded;
  $("engineBox").style.display = showLoader ? "block" : "none";
  $("engToggle").style.display = engineLoadAllowed ? "" : "none";
  $("engToggle").textContent = engineLoaded ? "change" : "";
  const cands = (m.candidates||[]).map(c => `<div>• <code>${c}</code></div>`).join("");
  if (!engineLoadAllowed && !engineLoaded) {
    $("engineHint").innerHTML = "No engine loaded and engine loading is disabled " +
      "on this (remote) server. Restart with <code>--engine PATH</code>, or with " +
      "<code>--allow-engine-load</code> to enable the loader.";
  } else {
    $("engineHint").innerHTML = (engineLoadAllowed && !engineLoaded) ?
      ("No engine was found automatically. Upload an engine <code>.py</code> or give a path." +
       (cands ? "<br>Tried:<br>" + cands : "")) : "";
  }
  $("engineErr").textContent = "";
  $("go").disabled = !engineLoaded;
  $("go").textContent = engineLoaded ? "Run scan" : "Load an engine first";
}

async function loadMeta() {
  try { applyMeta(await (await fetch("/meta")).json()); }
  catch (e) { $("engineName").textContent = "(failed to reach server)"; }
}

async function postEngine(url, payload, btn, busyLabel, idleLabel) {
  $("engineErr").textContent = "";
  btn.disabled = true; const prev = btn.textContent; btn.textContent = busyLabel;
  try {
    const m = await (await fetch(url, {method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)})).json();
    if (m.ok === false) { $("engineErr").textContent = m.error || "failed to load"; }
    else { applyMeta(m); }
  } catch (e) { $("engineErr").textContent = String(e); }
  btn.disabled = false; btn.textContent = idleLabel || prev;
}

async function loadEngine() {
  const path = $("enginePath").value.trim();
  if (!path) { $("engineErr").textContent = "Enter a path first."; return; }
  await postEngine("/load_engine", {path}, $("loadEngine"), "Loading…", "Load by path");
}

async function uploadEngine() {
  const f = $("engineFile").files[0];
  if (!f) { $("engineErr").textContent = "Choose a .py file first."; return; }
  const data_b64 = await fileToB64(f);
  await postEngine("/upload_engine", {filename:f.name, data_b64},
                   $("uploadEngine"), "Uploading…", "Upload & load engine");
}

$("sceneSeg").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  scene = b.dataset.scene;
  [...$("sceneSeg").children].forEach(x => x.classList.toggle("active", x === b));
  $("uploadBox").style.display = scene === "upload" ? "block" : "none";
});
$("manualCamera").addEventListener("change", () => {
  $("cameraBox").style.display = $("manualCamera").checked ? "block" : "none";
});

function fileToB64(f) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(btoa(String.fromCharCode(...new Uint8Array(r.result))));
    r.onerror = rej;
    r.readAsArrayBuffer(f);
  });
}

$("out").addEventListener("click", e => {
  if (e.target.tagName === "IMG" && e.target.dataset.zoom) {
    $("zoomImg").src = e.target.src; $("zoom").showModal();
  }
});
$("zoom").addEventListener("click", () => $("zoom").close());

async function run() {
  if (!engineLoaded) { alert("Load an engine first."); return; }
  const req = {
    scene, preset: $("preset").value,
    width: $("width").value, height: $("height").value,
    rays_per_pixel: $("rays").value, seed: $("seed").value,
    material: $("material").value, normalize: $("normalize").checked,
    simplify: $("simplify").checked, max_tris: $("maxTris").value,
    fast: $("fast").checked,
    manual_camera: $("manualCamera").checked,
  };
  if ($("manualCamera").checked) {
    Object.assign(req, {
      lens: $("lens").value,
      fov_deg: $("fov").value,
      fisheye_fov_deg: $("fisheyeFov").value,
      ortho_size: $("orthoSize").value,
      cam_pos_x: $("camPosX").value, cam_pos_y: $("camPosY").value, cam_pos_z: $("camPosZ").value,
      cam_tgt_x: $("camTgtX").value, cam_tgt_y: $("camTgtY").value, cam_tgt_z: $("camTgtZ").value,
      cam_up_x: $("camUpX").value, cam_up_y: $("camUpY").value, cam_up_z: $("camUpZ").value,
    });
  }
  if (scene === "upload") {
    const f = $("file").files[0];
    if (!f) { alert("Choose a .stl or .obj file first."); return; }
    req.filename = f.name;
    req.data_b64 = await fileToB64(f);
  }
  $("go").disabled = true;
  $("out").innerHTML = '<div class="empty"><span class="spin"></span>Running the engine…</div>';
  try {
    const r = await fetch("/scan", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(req)});
    const data = await r.json();
    if (!data.ok) { render_error(data.error || "unknown error"); }
    else { render(data); }
  } catch (err) { render_error(String(err)); }
  $("go").disabled = false;
}
$("go").addEventListener("click", run);
$("loadEngine").addEventListener("click", loadEngine);
$("uploadEngine").addEventListener("click", uploadEngine);
$("enginePath").addEventListener("keydown", e => { if (e.key === "Enter") loadEngine(); });
$("engToggle").addEventListener("click", () => {
  if (!engineLoaded || !engineLoadAllowed) return;  // already open and required, or disabled
  const box = $("engineBox");
  const open = box.style.display !== "none";
  box.style.display = open ? "none" : "block";
  $("engToggle").textContent = open ? "change" : "hide";
});
loadMeta();

function render_error(msg) {
  $("out").innerHTML = '<div class="card"><h3>Error</h3><div class="err"></div></div>';
  $("out").querySelector(".err").textContent = msg;
}

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(d) {
  let h = "";
  // images
  const order = ["Contact sheet","Geom edge overlay","Material core","Material filled"];
  h += '<div class="imgs">';
  for (const k of order) {
    if (!d.images[k]) continue;
    const cls = k === "Contact sheet" ? "imgcard sheet" : "imgcard";
    h += `<div class="${cls}"><div class="cap">${k}</div><img data-zoom="1" src="${d.images[k]}"></div>`;
  }
  h += '</div>';

  // run summary
  const s = d.stats, ds = s.depth_stats || {};
  h += '<div class="card"><h3>Run</h3><div class="kv">';
  h += `<div class="k">scene</div><div class="v">${esc(d.scene_info.source)}</div>`;
  if (d.scene_info.triangles != null) {
    let g = `${d.scene_info.triangles} tris / ${d.scene_info.vertices} verts`;
    if (d.scene_info.simplified)
      g += ` <span style="color:#caa53b">(simplified from ${d.scene_info.triangles_original})</span>`;
    h += `<div class="k">geometry</div><div class="v">${g}</div>`;
  }
  h += `<div class="k">preset</div><div class="v">${esc(d.preset)} — ${esc(d.preset_description||"")}</div>`;
  h += `<div class="k">render</div><div class="v">${s.width}×${s.height}, lens ${esc(s.lens)}, ${s.rays_per_pixel} rays/px</div>`;
  h += `<div class="k">camera</div><div class="v">pos [${(s.camera_position||[]).join(", ")}] → target [${(s.camera_target||[]).join(", ")}]</div>`;
  h += `<div class="k">coverage</div><div class="v">${ds.coverage!=null?(ds.coverage*100).toFixed(1)+"%":"—"}</div>`;
  if (ds.p05_depth!=null||ds.depth_p05!=null) {
    const p05=ds.depth_p05??ds.p05_depth, p50=ds.depth_p50??ds.p50_depth, p95=ds.depth_p95??ds.p95_depth;
    h += `<div class="k">depth p05/p50/p95</div><div class="v">${p05} / ${p50} / ${p95}</div>`;
  }
  h += `<div class="k">mode</div><div class="v">${s.manual_camera ? "manual camera capture" : (s.fast ? "fast preview (single burst, unframed)" : "high quality (auto-framed, stacked)")}</div>`;
  h += `<div class="k">runtime</div><div class="v">${s.total_runtime_seconds}s</div>`;
  h += '</div></div>';

  // classification
  const cc = d.classification_counts || {};
  const items = Object.entries(cc).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
  if (items.length) {
    const tot = items.reduce((a,[,v])=>a+v,0);
    const COL = {geom_edge:"#caa53b",hard_smooth:"#8fb8e0",metal_or_glass:"#c8d2e8",
      stone_hard:"#9b8f7e",wood_material:"#b07a44",soft_material:"#7aa05a",foliage:"#4f9d52",
      solid_surface:"#6f86b0",sky:"#1b2740",optical_only:"#caa0d0",acoustic_only:"#d08a8a",
      partial_occluder:"#d0a060",uncertain:"#5b6679"};
    h += '<div class="card"><h3>Classification</h3>';
    for (const [k,v] of items)
      h += `<span class="pill" style="background:${COL[k]||"#33405a"};color:#0b0f17">${esc(k)} ${(v/tot*100).toFixed(1)}%</span>`;
    h += '</div>';
  }

  // material channel report
  const mr = d.material_report || [];
  if (mr.length) {
    const cols = ["piece_type","pixel_count","acoustic_intensity_mean","ultrasonic_intensity_mean","polarization_mean","light_anti_mean"];
    h += '<div class="card"><h3>Material channel report</h3><table><tr>';
    for (const c of cols) h += `<th>${esc(c.replace(/_mean$/,""))}</th>`;
    h += '</tr>';
    for (const row of mr) {
      h += '<tr>';
      for (const c of cols) { let val=row[c]; if(typeof val==="number"&&!Number.isInteger(val)) val=val.toFixed(3); h += `<td>${val!=null?esc(val):"—"}</td>`; }
      h += '</tr>';
    }
    h += '</table></div>';
  }

  if ((d.warnings||[]).length)
    h += '<div class="card"><h3>Warnings</h3>' + d.warnings.map(w=>`<div class="hint">• ${esc(w)}</div>`).join("") + '</div>';

  $("out").innerHTML = h;
}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────
# HTTP server
# ──────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _content_length(self) -> int:
        try:
            return int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _read_json(self):
        n = self._content_length()
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _drain(self, n: int):
        """Discard up to n bytes of an oversized body so the connection can be
        reused / closed cleanly after we reject it."""
        remaining = n
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                break
            remaining -= len(chunk)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, index_html(), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(200, json.dumps(engine_meta()))
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": ENG is not None,
                                        "engine": os.path.basename(ENGINE_PATH) if ENGINE_PATH else None}))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        # Read the body before taking the lock; only the engine work is serialized.
        try:
            if self.path not in ("/scan", "/load_engine", "/upload_engine"):
                self._send(404, json.dumps({"ok": False, "error": "not found"}))
                return

            # Engine-loading endpoints are RCE by design — refuse them unless
            # explicitly enabled (always on for a loopback bind).
            if self.path in ("/load_engine", "/upload_engine") and not ALLOW_ENGINE_LOAD:
                self._drain(self._content_length())
                self._send(403, json.dumps({
                    "ok": False,
                    "error": "engine loading is disabled on this server "
                             "(start with --allow-engine-load to enable it)"}))
                return

            # Cap the body before reading it, so a client can't make us allocate
            # gigabytes just by claiming a huge Content-Length.
            n = self._content_length()
            if n > MAX_BODY_BYTES:
                self._drain(n)
                self._send(413, json.dumps({
                    "ok": False,
                    "error": f"request body too large ({n} bytes > "
                             f"{MAX_BODY_BYTES} limit; raise with --max-upload-mb)"}))
                return

            req = self._read_json()
            with _LOCK:
                if self.path == "/scan":
                    self._send(200, json.dumps(run_scan(req)))
                elif self.path == "/load_engine":
                    set_engine(req.get("path", ""))
                    print(f"  engine loaded via UI (path): {ENGINE_PATH}")
                    self._send(200, json.dumps(engine_meta()))
                else:  # /upload_engine
                    if not req.get("data_b64"):
                        raise ValueError("no file data received")
                    save_and_set_engine(req.get("filename", "engine.py"),
                                        base64.b64decode(req["data_b64"]))
                    print(f"  engine uploaded via UI: {ENGINE_PATH}")
                    self._send(200, json.dumps(engine_meta()))
        except (ValueError, KeyError, FileNotFoundError) as e:
            # Bad input / not found → 400, with the same JSON shape the UI reads.
            traceback.print_exc()
            self._send(400, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        except Exception as e:
            # Anything unexpected → 500.
            traceback.print_exc()
            self._send(500, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + (fmt % args) + "\n")


def main():
    global ENG, ENGINE_PATH, ALLOW_ENGINE_LOAD, MAX_BODY_BYTES
    ap = argparse.ArgumentParser(description="Web front-end for the LiDAR Lenses Wave engine.")
    ap.add_argument("--engine", default=None, help="path to the engine .py (default: latest found)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--allow-remote", action="store_true",
                    help="permit binding to a non-loopback host (network-exposed). "
                         "Off by default because the engine-loading endpoints run "
                         "arbitrary Python.")
    ap.add_argument("--allow-engine-load", action="store_true",
                    help="enable the /load_engine and /upload_engine endpoints even on "
                         "a remote bind. These execute arbitrary Python — only use this "
                         "on networks/users you trust. (Always enabled on loopback.)")
    ap.add_argument("--max-upload-mb", type=int, default=200,
                    help="maximum request body size in MB (default 200).")
    args = ap.parse_args()

    MAX_BODY_BYTES = max(1, args.max_upload_mb) * 1024 * 1024

    loopback = is_loopback_host(args.host)
    if not loopback and not args.allow_remote:
        print(f"Refusing to bind to non-loopback host {args.host!r}.")
        print("This server's engine-loading endpoints execute arbitrary Python, so")
        print("network exposure is opt-in. Re-run with --allow-remote if you really")
        print("mean to expose it (and consider --allow-engine-load separately).")
        sys.exit(2)

    # On loopback the local user is already trusted (they have a shell), so the
    # engine loader stays on. On a remote bind it's off unless explicitly asked for.
    ALLOW_ENGINE_LOAD = loopback or args.allow_engine_load

    atexit.register(_cleanup_engine_upload_dir)

    engine_path = args.engine
    if engine_path is None:
        for cand in _DEFAULT_ENGINES:
            if os.path.exists(cand):
                engine_path = cand
                break

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LiDAR Camera Studio → http://{args.host}:{args.port}")
    if not loopback:
        print(f"  ⚠ network-exposed bind ({args.host}); "
              f"engine loading {'ENABLED' if ALLOW_ENGINE_LOAD else 'disabled'}.")
    if engine_path and os.path.exists(engine_path):
        try:
            set_engine(engine_path)
            print(f"  engine : {ENGINE_PATH}")
            print(f"  presets: {', '.join(sorted(ENG.SENSOR_PRESETS.keys()))}")
        except Exception as e:
            print(f"  engine : failed to load {engine_path!r}: {e}")
            if ALLOW_ENGINE_LOAD:
                print("  → open the page and load an engine manually in the Engine box.")
            else:
                print("  → no engine loaded and engine loading is disabled; "
                      "restart with --engine PATH or --allow-engine-load.")
    else:
        print("  engine : none found automatically.")
        if ALLOW_ENGINE_LOAD:
            print("  → open the page and enter an engine .py path in the Engine box,")
            print("    or restart with --engine PATH.")
        else:
            print("  → engine loading is disabled; restart with --engine PATH "
                  "(or --allow-engine-load).")
    print("  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
