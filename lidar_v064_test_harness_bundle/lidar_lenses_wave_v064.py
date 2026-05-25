"""
lidar_lenses_wave.py — standalone multimodal ray/wave sensor sandbox (v0.6.4)

Single-file NumPy + PIL simulator built from a clean analytic raycaster,
extended into a synthetic sensing laboratory.  No package install needed.

Capabilities:
  - v0.6.2 classifier tuning: adaptive anti-edge threshold by scene percentile
    so dense interiors do not over-label carrier residue as geom_edge
  - Analytic primitive raycasting (boxes / spheres / cylinders)
  - Triangle meshes via Möller-Trumbore; two-level BVH acceleration; STL loader
  - Five lens types (pinhole / telephoto / fisheye / orthographic /
    equirectangular)
  - Rotated-carrier anti-channel de-striping; scrambled Halton sub-pixel sampling (~45% lower noise than jittered,
    no carrier-phase moiré thanks to per-burst Cranley-Patterson rotation)
  - Wave physics: optical light_coh / light_anti + acoustic sound_coh /
    sound_anti via a depth-phase + sub-pixel carrier phase
  - Dual-band acoustic: audible (~1 kHz) + ultrasonic (~40 kHz) impedance
  - Derived channels: acoustic_softness, acoustic_texture (attenuation-
    invariant), depth_per_pixel, depth_variance
  - Polarization preservation channel
  - Realism: sensor noise injection, atmospheric attenuation, burst
    stacking with proper depth weighting, range-aware classifier
    compensation
  - Statistical primitive transparency (per-ray probabilistic penetration)
    for foliage / fences / smoke; partial_occluder classification
    derived from within-pixel depth variance + porous-material signature
  - 9-category opinionated classifier (toggleable)
  - Procedural cabin demo and contact-sheet rendering

The cast_rays API accepts either a list of Primitives (legacy) or a Scene
(new). When given a list, it wraps in a Scene and builds a BVH transparently.

Three independent pipeline layers — stop at any:
  1. fire_burst()           → per-ray geometry
  2. compute_wave_channels()→ per-pixel coherence + intensity channels
  3. classify_pixels()      → opinionated per-pixel labels
"""
import math
import struct
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw


# ──────────────────────────────────────────────────────────
# PRIMITIVE STORAGE (unchanged from v2)
# ──────────────────────────────────────────────────────────
@dataclass
class Primitive:
    shape: str
    center: np.ndarray
    half_extents: np.ndarray
    rotation_matrix: np.ndarray
    inv_rotation_matrix: np.ndarray
    color: Tuple[float, float, float]
    piece_id: int
    piece_type: str
    # Probability that a given ray PASSES THROUGH this primitive instead
    # of reflecting off it.  0.0 = fully opaque (default, same as before).
    # 0.7 = ~70% of rays penetrate (e.g. canopy of leaves with gaps).
    # 0.95 = sparse mesh / wire fence.  When > 0, the ray caster needs
    # an RNG; without one, transparency is silently ignored.
    transparency: float = 0.0


def hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def make_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    cx, sx = math.cos(math.radians(rx_deg)), math.sin(math.radians(rx_deg))
    cy, sy = math.cos(math.radians(ry_deg)), math.sin(math.radians(ry_deg))
    cz, sz = math.cos(math.radians(rz_deg)), math.sin(math.radians(rz_deg))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def primitives_from_scene(result, packets) -> List[Primitive]:
    pieces = result.pieces
    if pieces:
        cx_c = sum(p.gx for p in pieces) / len(pieces)
        cz_c = sum(p.gz for p in pieces) / len(pieces)
    else:
        cx_c = cz_c = 0.0
    piece_by_id = {p.id: p for p in pieces}
    piece_type_by_id = {p.id: p.type for p in pieces}

    prims: List[Primitive] = []
    for pkt in packets:
        piece = piece_by_id.get(pkt["piece_id"])
        if piece is None:
            continue
        px = piece.gx - cx_c
        pz = piece.gz - cz_c
        py = 0.0
        for prim in pkt["primitives"]:
            shape = prim["shape"]
            local_pos = np.array(prim["position"], dtype=np.float64)
            rot_deg = prim.get("rotation", [0, 0, 0])
            R = make_rotation_matrix(*rot_deg)
            world_center = np.array([px, py, pz]) + local_pos
            dims = prim["dimensions"]
            if shape == "box":
                half = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
            elif shape == "cylinder":
                r = dims[1] if len(dims) > 1 else dims[0]
                h = dims[2] if len(dims) > 2 else dims[1]
                half = np.array([r, h/2, 0.0])
            elif shape == "sphere":
                r = dims[0]
                half = np.array([r, r, r])
            else:
                continue
            mat = prim.get("material", {})
            color = hex_to_rgb(mat.get("color", "#888888"))
            prims.append(Primitive(
                shape=shape, center=world_center, half_extents=half,
                rotation_matrix=R.T, inv_rotation_matrix=R,
                color=color, piece_id=pkt["piece_id"],
                piece_type=piece_type_by_id.get(pkt["piece_id"], "?"),
            ))
    return prims


# ──────────────────────────────────────────────────────────
# RAY-PRIMITIVE INTERSECTION
# ──────────────────────────────────────────────────────────
INF = 1e9
EPS = 1e-9


def _safe_inverse(d: np.ndarray) -> np.ndarray:
    sign = np.where(d >= 0, 1.0, -1.0)
    safe = np.where(np.abs(d) < EPS, sign * EPS, d)
    return 1.0 / safe


def ray_box_local(origins: np.ndarray, dirs: np.ndarray,
                  half: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Slab method, with argmin/argmax for the chosen-axis normal (v3 polish)."""
    inv = _safe_inverse(dirs)
    t1 = (-half - origins) * inv
    t2 = ( half - origins) * inv
    tmin_per_axis = np.minimum(t1, t2)
    tmax_per_axis = np.maximum(t1, t2)
    t_near = tmin_per_axis.max(axis=1)
    t_far  = tmax_per_axis.min(axis=1)
    hit = (t_far > t_near) & (t_far > EPS)
    use_far = (t_near <= EPS) & hit
    t = np.where(use_far, t_far, t_near)

    # Axis index that gave the chosen t (argmax for near, argmin for far)
    axis_near = np.argmax(tmin_per_axis, axis=1)
    axis_far  = np.argmin(tmax_per_axis, axis=1)
    axis = np.where(use_far, axis_far, axis_near)

    hit_pt = origins + dirs * t[:, None]
    normal = np.zeros_like(origins)
    rows = np.arange(len(origins))
    sign = np.sign(hit_pt[rows, axis])
    sign = np.where(sign == 0, 1.0, sign)
    normal[rows, axis] = np.where(hit, sign, 0.0)
    return np.where(hit, t, INF), normal


def ray_cylinder_local(origins: np.ndarray, dirs: np.ndarray,
                       r: float, half_h: float) -> Tuple[np.ndarray, np.ndarray]:
    """v3: test both side roots independently — fixes the t_side2 fallback bug."""
    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # Side surface — both roots tested
    a = dx*dx + dz*dz
    b = 2 * (ox*dx + oz*dz)
    c = ox*ox + oz*oz - r*r
    disc = b*b - 4*a*c
    valid = (disc >= 0) & (a > EPS)
    sqrt_disc = np.sqrt(np.where(disc < 0, 0, disc))
    a_safe = np.where(a < EPS, 1.0, a)
    t_s1 = (-b - sqrt_disc) / (2 * a_safe)
    t_s2 = (-b + sqrt_disc) / (2 * a_safe)
    y_s1 = oy + dy * t_s1
    y_s2 = oy + dy * t_s2
    s1_ok = valid & (t_s1 > EPS) & (np.abs(y_s1) <= half_h)
    s2_ok = valid & (t_s2 > EPS) & (np.abs(y_s2) <= half_h)
    t_s1_f = np.where(s1_ok, t_s1, INF)
    t_s2_f = np.where(s2_ok, t_s2, INF)
    t_side_final = np.minimum(t_s1_f, t_s2_f)

    # Caps
    dy_safe = np.where(np.abs(dy) < EPS, np.where(dy >= 0, EPS, -EPS), dy)
    t_top = (half_h - oy) / dy_safe
    t_bot = (-half_h - oy) / dy_safe

    def cap_hit(t_cap):
        x_at = ox + dx * t_cap
        z_at = oz + dz * t_cap
        return (x_at*x_at + z_at*z_at <= r*r) & (t_cap > EPS) & (np.abs(dy) > EPS)

    top_ok = cap_hit(t_top)
    bot_ok = cap_hit(t_bot)
    t_top_f = np.where(top_ok, t_top, INF)
    t_bot_f = np.where(bot_ok, t_bot, INF)

    t_all = np.stack([t_side_final, t_top_f, t_bot_f], axis=1)
    winner = np.argmin(t_all, axis=1)
    t_min = t_all[np.arange(len(t_all)), winner]

    normal = np.zeros_like(origins)
    is_side = (winner == 0) & (t_min < INF)
    if is_side.any():
        xh = ox[is_side] + dx[is_side] * t_min[is_side]
        zh = oz[is_side] + dz[is_side] * t_min[is_side]
        n_xz = np.stack([xh, np.zeros_like(xh), zh], axis=1)
        n_xz = n_xz / np.linalg.norm(n_xz, axis=1, keepdims=True).clip(EPS)
        normal[is_side] = n_xz
    normal[(winner == 1) & (t_min < INF), 1] =  1.0
    normal[(winner == 2) & (t_min < INF), 1] = -1.0
    return t_min, normal


def ray_sphere(origins: np.ndarray, dirs: np.ndarray,
               center: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    oc = origins - center
    a = (dirs * dirs).sum(axis=1)
    b = 2 * (dirs * oc).sum(axis=1)
    c = (oc * oc).sum(axis=1) - radius*radius
    disc = b*b - 4*a*c
    valid = disc >= 0
    sqrt_disc = np.sqrt(np.where(disc < 0, 0, disc))
    a_safe = np.where(a < EPS, 1.0, a)
    t1 = (-b - sqrt_disc) / (2 * a_safe)
    t2 = (-b + sqrt_disc) / (2 * a_safe)
    t = np.where(t1 > EPS, t1, t2)
    hit = valid & (t > EPS)
    normal = np.zeros_like(origins)
    if hit.any():
        hit_pt = origins[hit] + dirs[hit] * t[hit, None]
        n = hit_pt - center
        n = n / np.linalg.norm(n, axis=1, keepdims=True).clip(EPS)
        normal[hit] = n
    return np.where(hit, t, INF), normal


def cast_rays(origins, dirs, prims, rng=None):
    N = len(origins)
    best_t = np.full(N, INF)
    best_color = np.zeros((N, 3))
    best_normal = np.zeros((N, 3))
    best_pid = np.full(N, -1, dtype=np.int64)
    for prim in prims:
        if prim.shape == "sphere":
            t, normal = ray_sphere(origins, dirs, prim.center, prim.half_extents[0])
        else:
            o_local = (origins - prim.center) @ prim.rotation_matrix.T
            d_local = dirs @ prim.rotation_matrix.T
            if prim.shape == "box":
                t, n_local = ray_box_local(o_local, d_local, prim.half_extents)
            elif prim.shape == "cylinder":
                t, n_local = ray_cylinder_local(
                    o_local, d_local,
                    r=float(prim.half_extents[0]),
                    half_h=float(prim.half_extents[1]),
                )
            else:
                continue
            normal = n_local @ prim.inv_rotation_matrix.T
        # Statistical transparency: probabilistically skip this primitive's
        # hit for some rays.  These rays continue to find the next surface.
        if prim.transparency > 0 and rng is not None:
            skip = rng.random(t.shape) < prim.transparency
            t = np.where(skip, INF, t)
        color = np.array(prim.color)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_color[closer] = color
        best_normal[closer] = normal[closer]
        best_pid[closer] = prim.piece_id
    return best_t, best_color, best_normal, best_pid


# ──────────────────────────────────────────────────────────
# CAMERA / BURST (unchanged from v2)
# ──────────────────────────────────────────────────────────
@dataclass
class Camera:
    """Camera with selectable lens. Field semantics vary by lens:
        - pinhole:        fov_deg = total VERTICAL FOV (horizontal scales by aspect)
        - fisheye:        fisheye_fov_deg = total angular coverage (180 = full hemi);
                          returns ≤ n_samples rays (corners outside the lens
                          circle are dropped; ~78% retention for square images)
        - orthographic:   ortho_size = half-width of view frame in world units;
                          content outside ±ortho_size is silently clipped
        - equirectangular: ignores fov/ortho; full 360°×180° sphere
        - telephoto:      pinhole with narrow fov_deg (≤15)
    """
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    fov_deg: float = 60.0
    width: int = 600
    height: int = 400
    lens: str = "pinhole"
    fisheye_fov_deg: float = 180.0
    ortho_size: float = 8.0
    # Sub-pixel sampling: 'halton' (default, cleanest signal), 'grid'
    # (fastest, but aliasing risk on regular surfaces), 'jittered' (original
    # behavior, ~2× higher noise but unbiased).
    sampling_mode: str = "halton"


def _camera_basis(cam: Camera):
    """Right-handed camera basis: (forward, right, up). With gimbal-lock
    fallback for cameras where target-position is (anti)parallel to cam.up."""
    forward = cam.target - cam.position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam.up)
    rn = np.linalg.norm(right)
    if rn < EPS:
        # Pick an alternate vector that isn't parallel to forward.
        # If up is along Z, use X; otherwise use Z.
        alt = np.array([1., 0., 0.]) if abs(cam.up[2]) > 0.99 else np.array([0., 0., 1.])
        right = np.cross(forward, alt)
        right = right / np.linalg.norm(right)
    else:
        right = right / rn
    up = np.cross(right, forward)
    return forward, right, up


@dataclass
class Burst:
    """Output of fire_burst: per-ray geometry + hits for one camera shot.

    All arrays have length n_samples (the number of rays fired). Rays that
    missed have depth=INF. The per-ray `origins` field is what enables
    lens-symmetric fusion — pinhole bursts share origin but orthographic
    ones don't.
    """
    cam: object   # forward ref to Camera (defined in lenses.py)
    pixels: np.ndarray
    origins: np.ndarray
    dirs: np.ndarray
    depths: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    piece_ids: np.ndarray
    coverage: float = 0.0
    unique_pieces: int = 0
    pilot_score: float = 1.0


# ──────────────────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────────────────
def depth_to_color(depths, near, far):
    d = np.clip((depths - near) / (far - near + EPS), 0, 1)
    r = np.clip(1 - 2*d, 0, 1)
    g = np.clip(1 - np.abs(2*d - 1), 0, 1)
    b = np.clip(2*d - 1, 0, 1)
    return np.stack([r, g, b], axis=1)


def render_burst(burst, mode="lidar", near=0.5, far=20.0,
                 bg="#08101a", dot_size=1, auto_calibrate=False):
    W, H = burst.cam.width, burst.cam.height
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    hit = burst.depths < INF
    if not hit.any():
        return img
    hit_depths = burst.depths[hit]
    if auto_calibrate and mode == "lidar":
        near = float(hit_depths.min()) - 0.1
        far = float(np.percentile(hit_depths, 95))
        if far - near < 0.5:
            far = near + 0.5

    px = burst.pixels[hit, 0].astype(int)
    py = burst.pixels[hit, 1].astype(int)
    if mode == "lidar":
        cols = depth_to_color(hit_depths, near, far)
    elif mode == "material":
        cols = burst.colors[hit]
    elif mode == "shaded":
        light = np.array([0.4, 0.7, 0.3]); light /= np.linalg.norm(light)
        ndotl = np.clip((burst.normals[hit] * light).sum(axis=1), 0.2, 1.0)
        cols = burst.colors[hit] * ndotl[:, None]
    else:
        cols = burst.colors[hit]

    cols_int = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    for i in range(len(px)):
        x, y = int(px[i]), int(py[i])
        if 0 <= x < W and 0 <= y < H:
            c = tuple(int(v) for v in cols_int[i])
            if dot_size == 1:
                img.putpixel((x, y), c)
            else:
                draw.ellipse([x - dot_size, y - dot_size,
                              x + dot_size, y + dot_size], fill=c)
    return img


# ──────────────────────────────────────────────────────────
# PILOT-AND-PRUNE (unchanged from v2)
# ──────────────────────────────────────────────────────────
def pilot_and_prune(cams, prims, pilot_samples=2000, keep_k=4, seed=42):
    pilots = [fire_burst(cam, prims, pilot_samples, seed + i)
              for i, cam in enumerate(cams)]
    scores = np.array([p.coverage + 0.15 * math.log(1 + p.unique_pieces) for p in pilots])
    for p, s in zip(pilots, scores):
        p.pilot_score = float(s)
    keep_idx = list(np.argsort(scores)[-keep_k:][::-1])
    return keep_idx, pilots


# ──────────────────────────────────────────────────────────
# FUSION
# ──────────────────────────────────────────────────────────
def _burst_world_points(b: Burst):
    hit = b.depths < INF
    if not hit.any():
        return (np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros((0, 3)), np.array([]), np.array([], dtype=int))
    hit_idx = np.where(hit)[0]
    # Per-ray origins (correct for all source lenses — pinhole shares them,
    # orthographic doesn't). v0.2.0 made `origins` a required Burst field.
    origins = b.origins[hit_idx]
    pts = origins + b.dirs[hit_idx] * b.depths[hit_idx, None]
    return pts, b.colors[hit_idx], b.normals[hit_idx], b.depths[hit_idx], hit_idx


# NOTE: fuse_bursts_pointcloud, fuse_bursts_attention, and project_to_camera
# live in scene_lens.lenses (lens-aware versions). Import from there.


# Viridis approximation: black at 0 (no data), then standard viridis from
# dark purple → blue → cyan → green → yellow. Hardcoded 12 stops, lerped.
_VIRIDIS_STOPS = np.array([
    [0.000, 0.000, 0.000],   # 0.00 — black (no data)
    [0.267, 0.005, 0.329],   # 0.10 — dark purple
    [0.283, 0.141, 0.458],   # 0.20
    [0.254, 0.265, 0.530],   # 0.30
    [0.207, 0.372, 0.553],   # 0.40
    [0.164, 0.471, 0.558],   # 0.50
    [0.128, 0.567, 0.551],   # 0.60
    [0.135, 0.659, 0.518],   # 0.70
    [0.267, 0.749, 0.441],   # 0.80
    [0.478, 0.821, 0.318],   # 0.90
    [0.741, 0.873, 0.150],   # 0.95
    [0.993, 0.906, 0.144],   # 1.00 — bright yellow
])


def _confidence_heatmap(conf_arr: np.ndarray,
                        bg_rgb: Tuple[int, int, int] = (8, 16, 26),
                        smooth: bool = False) -> Image.Image:
    """Render a confidence array as a viridis-like heatmap."""
    H_max = conf_arr.max()
    if H_max < EPS:
        return Image.new("RGB", conf_arr.shape[::-1], bg_rgb)
    cf = conf_arr / H_max
    n_stops = len(_VIRIDIS_STOPS)
    stop_positions = np.linspace(0, 1, n_stops)

    # Vectorized lerp through the LUT
    out = np.zeros((*cf.shape, 3))
    for ch in range(3):
        out[..., ch] = np.interp(cf, stop_positions, _VIRIDIS_STOPS[:, ch])

    # Background where there was zero confidence
    out[cf == 0] = np.array(bg_rgb) / 255.0

    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    if smooth:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def composite_grid(images, labels, cols=3, pad=8, label_h=24, bg="#0a0a0e"):
    """Tile labeled images into a grid.  Cells size to the LARGEST image so
    panels with extra height (e.g. classification + legend) aren't clipped.
    """
    if not images:
        return Image.new("RGB", (100, 100), bg)
    iw = max(im.size[0] for im in images)
    ih = max(im.size[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    W = cols * (iw + pad) + pad
    H = rows * (ih + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (iw + pad)
        y = pad + r * (ih + label_h + pad)
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill="#cce0e0")
    return sheet


# ══════════════════════════════════════════════════════════════════════════
# v0.2.0 ADDITIONS — Triangle meshes, BVH acceleration, Scene container
# ══════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# TRIANGLE MESH
# ──────────────────────────────────────────────────────────
@dataclass
class Mesh:
    """Triangle mesh. Holds many triangles sharing a transform and material.

    Fields:
        vertices:     (N_v, 3) world-space vertex positions
        faces:        (N_f, 3) int — indices into vertices
        face_normals: (N_f, 3) — precomputed unit normals (auto if None at init)
        color:        single RGB tuple, applied to all faces (per-face colors
                      can be added later)
        piece_id, piece_type: same role as Primitive's fields

    The mesh caches v0/edge1/edge2 arrays for fast Möller-Trumbore and stores
    its own internal BVH over triangles, built lazily on first ray-cast.
    """
    vertices: np.ndarray
    faces: np.ndarray
    color: Tuple[float, float, float]
    piece_id: int
    piece_type: str = "mesh"
    face_normals: np.ndarray = None
    aabb_min: np.ndarray = None
    aabb_max: np.ndarray = None
    # Cached precomputed edge arrays (used by _cast_mesh)
    _tri_v0: np.ndarray = None
    _tri_e1: np.ndarray = None
    _tri_e2: np.ndarray = None
    _bvh: object = None  # BVHNode, built lazily

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.face_normals is None:
            self.face_normals = _compute_face_normals(self.vertices, self.faces)
        if self.aabb_min is None:
            self.aabb_min = self.vertices.min(axis=0)
            self.aabb_max = self.vertices.max(axis=0)
        # Cache triangle edges for Möller-Trumbore
        self._tri_v0 = self.vertices[self.faces[:, 0]]
        self._tri_e1 = self.vertices[self.faces[:, 1]] - self._tri_v0
        self._tri_e2 = self.vertices[self.faces[:, 2]] - self._tri_v0


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face unit normals via cross product of two edges."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True).clip(EPS)
    return n / norms


def _triangle_aabbs(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """AABB for each triangle: (N_f, 2, 3) where [...,0]=min, [...,1]=max."""
    v = vertices[faces]  # (N_f, 3, 3)
    return np.stack([v.min(axis=1), v.max(axis=1)], axis=1)


def ray_triangle_batch(origins: np.ndarray, dirs: np.ndarray,
                       v0: np.ndarray, edge1: np.ndarray, edge2: np.ndarray,
                       _chunk_threshold: int = 10_000_000
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Möller-Trumbore: each ray vs each triangle in the batch.

    Args:
        origins: (R, 3)
        dirs:    (R, 3)
        v0:      (T, 3) triangle vertex 0
        edge1:   (T, 3) = v1 - v0
        edge2:   (T, 3) = v2 - v0
        _chunk_threshold: chunk over rays when R*T exceeds this (default 10M)
                          to keep peak memory below ~250 MB regardless of input

    Returns:
        t_hit:  (R,) — nearest hit distance per ray; INF if miss
        f_idx:  (R,) — index of winning triangle in the batch; -1 if miss

    Memory note: the unchunked implementation allocates (R, T, 3) work arrays.
    For R=50000 and T=5000, that's 6 GB per array — this function chunks the
    ray axis automatically when the product exceeds _chunk_threshold.
    """
    R, T = len(origins), len(v0)
    if R == 0 or T == 0:
        return np.full(R, INF), np.full(R, -1, dtype=np.int64)

    # Chunk over rays when the work tensor would exceed the threshold
    if R * T > _chunk_threshold:
        out_t = np.full(R, INF)
        out_f = np.full(R, -1, dtype=np.int64)
        chunk = max(1, _chunk_threshold // max(1, T))
        for i in range(0, R, chunk):
            sl = slice(i, i + chunk)
            out_t[sl], out_f[sl] = ray_triangle_batch(
                origins[sl], dirs[sl], v0, edge1, edge2,
                _chunk_threshold=_chunk_threshold,
            )
        return out_t, out_f

    # h = dirs × edge2 → (R, T, 3)
    h = np.cross(dirs[:, None, :], edge2[None, :, :])
    a = np.einsum("tk,rtk->rt", edge1, h)  # (R, T)
    parallel = np.abs(a) < EPS
    a_safe = np.where(parallel, 1.0, a)
    f = 1.0 / a_safe

    s = origins[:, None, :] - v0[None, :, :]            # (R, T, 3)
    u = f * np.einsum("rtk,rtk->rt", s, h)              # (R, T)

    q = np.cross(s, edge1[None, :, :])                   # (R, T, 3)
    v_bary = f * np.einsum("rk,rtk->rt", dirs, q)        # (R, T)

    t = f * np.einsum("tk,rtk->rt", edge2, q)            # (R, T)

    valid = (~parallel) & (u >= 0) & (u <= 1) & \
            (v_bary >= 0) & (u + v_bary <= 1) & (t > EPS)
    t = np.where(valid, t, INF)
    best_f = np.argmin(t, axis=1)
    best_t = t[np.arange(R), best_f]
    hit = best_t < INF
    return np.where(hit, best_t, INF), np.where(hit, best_f, -1)


# ──────────────────────────────────────────────────────────
# BVH (axis-aligned bounding volume hierarchy)
# ──────────────────────────────────────────────────────────
@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left: object = None       # BVHNode
    right: object = None      # BVHNode
    item_indices: np.ndarray = None  # leaf only: indices into source array

    @property
    def is_leaf(self) -> bool:
        return self.item_indices is not None


def build_bvh(aabbs: np.ndarray, max_leaf: int = 4) -> BVHNode:
    """Build a BVH over a set of AABBs using simple median-split on the
    longest axis. Returns the root node.

    Args:
        aabbs: (N, 2, 3) — aabbs[i,0]=min, aabbs[i,1]=max
        max_leaf: max items per leaf
    """
    N = len(aabbs)
    if N == 0:
        return BVHNode(aabb_min=np.zeros(3), aabb_max=np.zeros(3),
                       item_indices=np.array([], dtype=np.int64))
    indices = np.arange(N)

    def _recurse(idx):
        if len(idx) <= max_leaf:
            mn = aabbs[idx, 0].min(axis=0)
            mx = aabbs[idx, 1].max(axis=0)
            return BVHNode(aabb_min=mn, aabb_max=mx,
                           item_indices=idx.astype(np.int64))
        # Split along longest axis at the median of centroids
        centers = (aabbs[idx, 0] + aabbs[idx, 1]) / 2
        extents = aabbs[idx, 1].max(axis=0) - aabbs[idx, 0].min(axis=0)
        axis = int(np.argmax(extents))
        order = np.argsort(centers[:, axis])
        idx_sorted = idx[order]
        mid = len(idx_sorted) // 2
        left = _recurse(idx_sorted[:mid])
        right = _recurse(idx_sorted[mid:])
        mn = np.minimum(left.aabb_min, right.aabb_min)
        mx = np.maximum(left.aabb_max, right.aabb_max)
        return BVHNode(aabb_min=mn, aabb_max=mx, left=left, right=right)

    return _recurse(indices)


def _bvh_aabb_test_batch(node: BVHNode, origins: np.ndarray, dirs: np.ndarray,
                         current_best_t: np.ndarray) -> np.ndarray:
    """Return a boolean mask: which rays could still hit this node's AABB
    before their current best_t. Uses >= for tangent-ray acceptance so that
    zero-extent AABBs (e.g. coplanar triangles) don't get falsely rejected.
    """
    if len(origins) == 0:
        return np.array([], dtype=bool)
    inv = _safe_inverse(dirs)
    t1 = (node.aabb_min - origins) * inv
    t2 = (node.aabb_max - origins) * inv
    t_near = np.minimum(t1, t2).max(axis=1)
    t_far = np.maximum(t1, t2).min(axis=1)
    return (t_far >= t_near) & (t_far > EPS) & (t_near < current_best_t)


# ──────────────────────────────────────────────────────────
# SCENE CONTAINER + UNIFIED CAST
# ──────────────────────────────────────────────────────────
@dataclass
class Scene:
    """Container for primitives + meshes + cached top-level BVH."""
    primitives: List[Primitive] = field(default_factory=list)
    meshes: List[Mesh] = field(default_factory=list)
    _bvh: BVHNode = None  # built lazily; items 0..N_prims-1 are primitives,
                          # items N_prims..N_prims+N_meshes-1 are meshes

    def build_bvh(self):
        """Build the top-level BVH over all primitives + meshes."""
        all_aabbs = []
        for p in self.primitives:
            all_aabbs.append(_primitive_aabb(p))
        for m in self.meshes:
            all_aabbs.append(np.stack([m.aabb_min, m.aabb_max]))
        if not all_aabbs:
            self._bvh = build_bvh(np.zeros((0, 2, 3)))
            return
        aabbs = np.stack(all_aabbs)
        self._bvh = build_bvh(aabbs)

    @property
    def bvh(self):
        if self._bvh is None:
            self.build_bvh()
        return self._bvh

    @property
    def n_primitives(self):
        return len(self.primitives)


def _primitive_aabb(p: Primitive) -> np.ndarray:
    """World-space AABB for a primitive. Uses the primitive's bounding sphere
    (max half-extent) as a conservative bound; tight for axis-aligned shapes.
    """
    # Conservative bound: use the max half-extent as a sphere radius
    if p.shape == "sphere":
        r = float(p.half_extents[0])
        return np.stack([p.center - r, p.center + r])
    elif p.shape == "cylinder":
        r = float(p.half_extents[0])
        h = float(p.half_extents[1])
        # Loose for rotated cylinders; tight for axis-aligned
        bound = max(r, h)
        return np.stack([p.center - bound, p.center + bound])
    else:  # box — use diagonal as conservative bound
        bound = float(np.linalg.norm(p.half_extents))
        return np.stack([p.center - bound, p.center + bound])


def _cast_primitive(origins, dirs, prim, ray_indices,
                    best_t, best_color, best_normal, best_pid, rng=None):
    """Test a subset of rays against one primitive; update best_* in place."""
    if len(ray_indices) == 0:
        return
    o = origins[ray_indices]
    d = dirs[ray_indices]
    if prim.shape == "sphere":
        t, normal = ray_sphere(o, d, prim.center, prim.half_extents[0])
    else:
        o_local = (o - prim.center) @ prim.rotation_matrix.T
        d_local = d @ prim.rotation_matrix.T
        if prim.shape == "box":
            t, n_local = ray_box_local(o_local, d_local, prim.half_extents)
        elif prim.shape == "cylinder":
            t, n_local = ray_cylinder_local(
                o_local, d_local,
                r=float(prim.half_extents[0]),
                half_h=float(prim.half_extents[1]),
            )
        else:
            return
        normal = n_local @ prim.inv_rotation_matrix.T
    # Statistical transparency: skip this primitive's hit for a random
    # subset of rays, letting them continue to find the next surface.
    if prim.transparency > 0 and rng is not None:
        skip = rng.random(t.shape) < prim.transparency
        t = np.where(skip, INF, t)
    color = np.array(prim.color)
    closer = t < best_t[ray_indices]
    upd_idx = ray_indices[closer]
    best_t[upd_idx] = t[closer]
    best_color[upd_idx] = color
    best_normal[upd_idx] = normal[closer]
    best_pid[upd_idx] = prim.piece_id


def _cast_mesh(origins, dirs, mesh, ray_indices,
               best_t, best_color, best_normal, best_pid):
    """Test a subset of rays against a mesh (via the mesh's internal BVH).
    Uses Mesh's cached triangle edges (built once in __post_init__).
    """
    if len(ray_indices) == 0:
        return
    # Build mesh BVH lazily over triangles
    if mesh._bvh is None:
        tri_aabbs = _triangle_aabbs(mesh.vertices, mesh.faces)
        mesh._bvh = build_bvh(tri_aabbs, max_leaf=8)

    v0, edge1, edge2 = mesh._tri_v0, mesh._tri_e1, mesh._tri_e2
    color = np.array(mesh.color)

    def _walk(node, ray_idx):
        if len(ray_idx) == 0:
            return
        mask = _bvh_aabb_test_batch(node, origins[ray_idx], dirs[ray_idx], best_t[ray_idx])
        sur = ray_idx[mask]
        if len(sur) == 0:
            return
        if node.is_leaf:
            fi = node.item_indices
            t_hit, f_hit = ray_triangle_batch(
                origins[sur], dirs[sur], v0[fi], edge1[fi], edge2[fi]
            )
            closer = t_hit < best_t[sur]
            upd = sur[closer]
            best_t[upd] = t_hit[closer]
            best_color[upd] = color
            # Map local face index back to global face index
            winning_global = fi[f_hit[closer]]
            best_normal[upd] = mesh.face_normals[winning_global]
            best_pid[upd] = mesh.piece_id
        else:
            _walk(node.left, sur)
            _walk(node.right, sur)

    _walk(mesh._bvh, ray_indices)


def cast_rays_scene(origins: np.ndarray, dirs: np.ndarray, scene: Scene, rng=None):
    """BVH-accelerated ray cast against a Scene (primitives + meshes).

    Pass rng to enable statistical transparency on primitives that have
    `transparency > 0`.  Without rng, transparency is ignored (rays always
    register a hit on every primitive they intersect).
    """
    N = len(origins)
    best_t = np.full(N, INF)
    best_color = np.zeros((N, 3))
    best_normal = np.zeros((N, 3))
    best_pid = np.full(N, -1, dtype=np.int64)

    if N == 0 or (scene.n_primitives == 0 and len(scene.meshes) == 0):
        return best_t, best_color, best_normal, best_pid

    n_prims = scene.n_primitives

    def _walk(node, ray_idx):
        if len(ray_idx) == 0:
            return
        mask = _bvh_aabb_test_batch(node, origins[ray_idx], dirs[ray_idx], best_t[ray_idx])
        sur = ray_idx[mask]
        if len(sur) == 0:
            return
        if node.is_leaf:
            for item_idx in node.item_indices:
                if item_idx < n_prims:
                    _cast_primitive(origins, dirs, scene.primitives[item_idx], sur,
                                    best_t, best_color, best_normal, best_pid, rng=rng)
                else:
                    mesh = scene.meshes[item_idx - n_prims]
                    _cast_mesh(origins, dirs, mesh, sur,
                               best_t, best_color, best_normal, best_pid)
        else:
            _walk(node.left, sur)
            _walk(node.right, sur)

    _walk(scene.bvh, np.arange(N))
    return best_t, best_color, best_normal, best_pid


# ──────────────────────────────────────────────────────────
# STL LOADER (binary + ASCII)
# ──────────────────────────────────────────────────────────
def load_stl(path: str, color=(0.7, 0.7, 0.7), piece_id: int = 1000) -> Mesh:
    """Load an STL file (binary or ASCII) as a Mesh.

    Auto-detects format by looking at the file header AND scanning for the
    'facet normal' landmark (which appears in ASCII STLs but is unlikely
    as a coincidence in binary data).
    """
    with open(path, "rb") as f:
        all_bytes = f.read()
    is_ascii = (all_bytes.lstrip().startswith(b"solid")
                and b"facet normal" in all_bytes[:10000])
    if is_ascii:
        return _load_stl_ascii(all_bytes, color, piece_id)
    # Binary: skip the 80-byte header
    return _load_stl_binary(all_bytes[80:], color, piece_id)


def _load_stl_binary(buf: bytes, color, piece_id) -> Mesh:
    n_tri = struct.unpack("<I", buf[:4])[0]
    verts = np.zeros((n_tri * 3, 3), dtype=np.float64)
    faces = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    off = 4
    for i in range(n_tri):
        # Per-triangle layout: 12 bytes normal + 36 bytes (3 vertices × 3 floats)
        # + 2 bytes attribute count = 50 bytes. Read just the 9 vertex floats.
        v = struct.unpack("<9f", buf[off + 12 : off + 48])
        verts[i * 3]     = v[0:3]
        verts[i * 3 + 1] = v[3:6]
        verts[i * 3 + 2] = v[6:9]
        off += 50
    return Mesh(vertices=verts, faces=faces, color=color, piece_id=piece_id)


def _load_stl_ascii(buf: bytes, color, piece_id) -> Mesh:
    text = buf.decode("utf-8", errors="replace")
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts, dtype=np.float64)
    n_tri = len(verts) // 3
    faces = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    return Mesh(vertices=verts[:n_tri * 3], faces=faces, color=color, piece_id=piece_id)


# ──────────────────────────────────────────────────────────
# BACKWARDS-COMPATIBLE cast_rays
# ──────────────────────────────────────────────────────────
_legacy_cast_rays = cast_rays  # save the original linear-scan implementation


def cast_rays(origins, dirs, scene_or_prims, rng=None):
    """Cast rays against a Scene (BVH-accelerated) or a list of Primitives
    (legacy linear scan; for big scenes wrap in Scene() for BVH speedup).

    Pass `rng` to enable statistical transparency on primitives.
    """
    if isinstance(scene_or_prims, Scene):
        return cast_rays_scene(origins, dirs, scene_or_prims, rng=rng)
    if isinstance(scene_or_prims, list):
        # Auto-wrap into a Scene only if it's worth it (≥8 primitives).
        # Smaller scenes are faster without BVH overhead.
        if len(scene_or_prims) >= 8:
            scene = Scene(primitives=list(scene_or_prims))
            return cast_rays_scene(origins, dirs, scene, rng=rng)
        return _legacy_cast_rays(origins, dirs, scene_or_prims, rng=rng)
    raise TypeError(f"cast_rays expects Scene or list of Primitive, got {type(scene_or_prims)}")


# ── LENS EXTENSION (merged from lenses.py) ──────────────────────────────





def _halton(n: int, base: int) -> np.ndarray:
    """First n values of the Halton low-discrepancy sequence in given base.

    Vectorized — processes all indices in parallel until they're all zero.
    Cost is ~log_base(n) iterations of n-wide numpy ops, much faster than
    a per-index loop.
    """
    indices = np.arange(1, n + 1, dtype=np.int64)
    out = np.zeros(n, dtype=np.float64)
    f = 1.0
    while np.any(indices > 0):
        f /= base
        out += f * (indices % base)
        indices //= base
    return out


def _stratified_pixels(W: int, H: int, n_samples: int, rng,
                       mode: str = "halton") -> Tuple[np.ndarray, np.ndarray]:
    """Shared stratified pixel sampler.  Returns (px, py).

    The image is divided into a grid of (grid_w × grid_h) cells with one ray
    per cell.  `mode` controls where within each cell the ray sits:

        "halton"   — scrambled Halton(base=2,3) low-discrepancy sequence
                     with per-burst Cranley-Patterson rotation.  Quasi-
                     random, space-filling, no aliasing.  ~45% lower noise
                     on coherence channels than 'jittered'.  The
                     recommended default — best signal quality for most
                     scenes at marginal extra cost.  Per-burst rotation
                     also makes burst-stacking effective (each stack gets
                     independent sample positions).
        "grid"     — cell centers (no jitter).  Fastest (no RNG calls), and
                     deterministic.  BUT produces visible aliasing stripes
                     on smooth surfaces aligned with the pixel grid.  Use
                     for fast iteration / preview, not final output.
        "jittered" — random uniform within each cell.  Original engine
                     behavior — maximum aliasing protection, but ~2× higher
                     noise floor on wave channels than 'halton'.  Keep for
                     reproducing older results or when you want pure noise
                     rather than any structured residual.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    aspect = W / H
    grid_w = max(1, int(math.sqrt(n_samples * aspect)))
    grid_h = max(1, n_samples // grid_w)
    gx, gy = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    if mode == "halton":
        n = gx.size
        # Halton(2,3) + per-burst Cranley-Patterson rotation.  Pure
        # Halton has a visible (but low-amplitude) diamond pattern on
        # flat surfaces because the deterministic sequence aligns with
        # the carrier-phase formula.  The pattern is cosmetic — the
        # actual noise floor and false-edge count are both about half
        # of 'jittered' — so we keep pure Halton as the default and
        # accept the visual artifact in exchange for the cleaner signal.
        # CP rotation per burst gives stacking proper independence.
        # If the diamond pattern bothers you, use sampling_mode="jittered"
        # for unstructured-but-noisier output.
        jx = ((_halton(n, base=2) + rng.random()) % 1.0).reshape(gx.shape)
        jy = ((_halton(n, base=3) + rng.random()) % 1.0).reshape(gx.shape)
    elif mode == "grid":
        jx = np.full(gx.shape, 0.5)
        jy = np.full(gy.shape, 0.5)
    elif mode == "jittered":
        jx = rng.random(gx.shape); jy = rng.random(gy.shape)
    else:
        raise ValueError(f"unknown sampling mode: {mode!r}. "
                          f"Use 'halton', 'grid', or 'jittered'.")
    px = ((gx + jx) / grid_w * W).flatten()
    py = ((gy + jy) / grid_h * H).flatten()
    return px, py


# ──────────────────────────────────────────────────────────
# LENS: PINHOLE (the original)
# ──────────────────────────────────────────────────────────
def rays_pinhole(cam: Camera, n_samples: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W, H = cam.width, cam.height
    aspect = W / H
    fov_rad = math.radians(cam.fov_deg)
    half_h = math.tan(fov_rad / 2)
    half_w = half_h * aspect
    px, py = _stratified_pixels(W, H, n_samples, rng, mode=cam.sampling_mode)
    ndc_x = (px / W - 0.5) * 2 * half_w
    ndc_y = (0.5 - py / H) * 2 * half_h
    forward, right, up = _camera_basis(cam)
    dirs = (forward[None, :]
            + ndc_x[:, None] * right[None, :]
            + ndc_y[:, None] * up[None, :])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam.position[None, :], (len(dirs), 1))
    return origins, dirs, np.stack([px, py], axis=1)


# ──────────────────────────────────────────────────────────
# LENS: FISHEYE (equidistant projection)
# ──────────────────────────────────────────────────────────
def rays_fisheye(cam: Camera, n_samples: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equidistant fisheye. r_image = f · θ, so angle from optical axis is
    linear in image radius. 180° gives a hemisphere in one frame."""
    W, H = cam.width, cam.height
    max_theta = math.radians(cam.fisheye_fov_deg / 2)
    px, py = _stratified_pixels(W, H, n_samples, rng, mode=cam.sampling_mode)

    # NDC, then make it a square in the smaller dimension so the fisheye
    # circle fits inside the image
    if W >= H:
        ndc_x = (px / W - 0.5) * 2 * (W / H)
        ndc_y = (0.5 - py / H) * 2
    else:
        ndc_x = (px / W - 0.5) * 2
        ndc_y = (0.5 - py / H) * 2 * (H / W)

    r = np.sqrt(ndc_x*ndc_x + ndc_y*ndc_y)
    inside = r <= 1.0
    ndc_x, ndc_y, r = ndc_x[inside], ndc_y[inside], r[inside]
    px, py = px[inside], py[inside]

    theta = r * max_theta
    phi = np.arctan2(ndc_y, ndc_x)
    forward, right, up = _camera_basis(cam)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    dirs = (cos_t[:, None] * forward[None, :]
            + (sin_t * cos_p)[:, None] * right[None, :]
            + (sin_t * sin_p)[:, None] * up[None, :])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam.position[None, :], (len(dirs), 1))
    return origins, dirs, np.stack([px, py], axis=1)


# ──────────────────────────────────────────────────────────
# LENS: ORTHOGRAPHIC (parallel rays)
# ──────────────────────────────────────────────────────────
def rays_orthographic(cam: Camera, n_samples: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel rays. ortho_size is the half-width of the view frame in
    world units. Useful for floor plans (top-down) and elevations (side)."""
    W, H = cam.width, cam.height
    aspect = W / H
    half_w = cam.ortho_size
    half_h = cam.ortho_size / aspect
    px, py = _stratified_pixels(W, H, n_samples, rng, mode=cam.sampling_mode)
    ndc_x = (px / W - 0.5) * 2 * half_w
    ndc_y = (0.5 - py / H) * 2 * half_h
    forward, right, up = _camera_basis(cam)
    n_rays = len(px)
    dirs = np.tile(forward[None, :], (n_rays, 1))
    origins = (cam.position[None, :]
               + ndc_x[:, None] * right[None, :]
               + ndc_y[:, None] * up[None, :])
    return origins, dirs, np.stack([px, py], axis=1)


# ──────────────────────────────────────────────────────────
# LENS: EQUIRECTANGULAR (360°×180° spherical)
# ──────────────────────────────────────────────────────────
def rays_equirectangular(cam: Camera, n_samples: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sphere projection. X axis = longitude [-π, π], Y axis =
    latitude [-π/2, π/2]. The 'forward' direction is at the image center."""
    W, H = cam.width, cam.height
    px, py = _stratified_pixels(W, H, n_samples, rng, mode=cam.sampling_mode)
    lon = (px / W - 0.5) * 2 * math.pi
    lat = (0.5 - py / H) * math.pi

    forward, right, up = _camera_basis(cam)
    cos_lat = np.cos(lat); sin_lat = np.sin(lat)
    cos_lon = np.cos(lon); sin_lon = np.sin(lon)
    dirs = (cos_lat[:, None] * (cos_lon[:, None] * forward[None, :]
                                + sin_lon[:, None] * right[None, :])
            + sin_lat[:, None] * up[None, :])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam.position[None, :], (len(dirs), 1))
    return origins, dirs, np.stack([px, py], axis=1)


# ──────────────────────────────────────────────────────────
# REGISTRY + DISPATCH
# ──────────────────────────────────────────────────────────
LENS_REGISTRY = {
    "pinhole":         rays_pinhole,
    "telephoto":       rays_pinhole,   # alias: pinhole with narrow fov_deg
    "fisheye":         rays_fisheye,
    "orthographic":    rays_orthographic,
    "equirectangular": rays_equirectangular,
}


def fire_burst(cam: Camera, prims: List[Primitive], n_samples: int, seed: int) -> Burst:
    """Fire a burst using the camera's lens type."""
    rng = np.random.default_rng(seed)
    gen = LENS_REGISTRY.get(cam.lens, rays_pinhole)
    origins, dirs, pixels = gen(cam, n_samples, rng)
    t, color, normal, pid = cast_rays(origins, dirs, prims, rng=rng)
    hit = t < INF
    coverage = float(hit.mean()) if len(t) else 0.0
    unique_pieces = int(len(np.unique(pid[hit]))) if hit.any() else 0
    return Burst(
        cam=cam, pixels=pixels, origins=origins, dirs=dirs, depths=t, colors=color,
        normals=normal, piece_ids=pid,
        coverage=coverage, unique_pieces=unique_pieces,
    )


# ──────────────────────────────────────────────────────────
# CUBEMAP HELPER (6 perpendicular pinholes from one point)
# ──────────────────────────────────────────────────────────
def cubemap_cameras(position, width: int = 400, height: int = 400) -> List[Tuple[str, Camera]]:
    """Return 6 cameras forming a cubemap rig at `position`.
    Each is a 90° FOV pinhole pointing at one face of the cube."""
    p = np.asarray(position, dtype=float)
    face_specs = [
        ("+X", [ 1,  0,  0], [0, 1, 0]),
        ("-X", [-1,  0,  0], [0, 1, 0]),
        ("+Y", [ 0,  1,  0], [0, 0, 1]),
        ("-Y", [ 0, -1,  0], [0, 0,-1]),
        ("+Z", [ 0,  0,  1], [0, 1, 0]),
        ("-Z", [ 0,  0, -1], [0, 1, 0]),
    ]
    cams = []
    for name, d, up in face_specs:
        d = np.array(d, float); up = np.array(up, float)
        cams.append((name, Camera(
            position=p.copy(),
            target=p + d,
            up=up,
            fov_deg=90.0,
            width=width, height=height,
            lens="pinhole",
        )))
    return cams


# ──────────────────────────────────────────────────────────
# LENS-AWARE PROJECTION (inverse of each ray generator)
# ──────────────────────────────────────────────────────────
def _project_pinhole(points: np.ndarray, cam: Camera):
    """Inverse of rays_pinhole: world point → (sx, sy, depth, in_bounds)."""
    forward, right, up = _camera_basis(cam)
    W, H = cam.width, cam.height
    aspect = W / H
    fov_rad = math.radians(cam.fov_deg)
    half_h = math.tan(fov_rad / 2)
    half_w = half_h * aspect

    rel = points - cam.position
    depth = rel @ forward
    in_front = depth > 0.05
    x_cam = rel @ right
    y_cam = rel @ up

    sx = np.full(len(points), -1.0)
    sy = np.full(len(points), -1.0)
    safe_d = np.where(in_front, depth, 1.0)
    sx[in_front] = (0.5 + x_cam[in_front] / (safe_d[in_front] * 2 * half_w)) * W
    sy[in_front] = (0.5 - y_cam[in_front] / (safe_d[in_front] * 2 * half_h)) * H
    in_bounds = in_front & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    return sx, sy, depth, in_bounds


def _project_fisheye(points: np.ndarray, cam: Camera):
    """Inverse of rays_fisheye (equidistant projection)."""
    forward, right, up = _camera_basis(cam)
    W, H = cam.width, cam.height
    max_theta = math.radians(cam.fisheye_fov_deg / 2)

    rel = points - cam.position
    dist = np.linalg.norm(rel, axis=1).clip(EPS)
    dirs_unit = rel / dist[:, None]

    cos_theta = (dirs_unit @ forward).clip(-1, 1)
    theta = np.arccos(cos_theta)
    in_view = theta <= max_theta

    x_proj = dirs_unit @ right
    y_proj = dirs_unit @ up
    phi = np.arctan2(y_proj, x_proj)

    r = theta / max_theta
    ndc_x = r * np.cos(phi)
    ndc_y = r * np.sin(phi)

    # Inverse of rays_fisheye NDC scaling
    if W >= H:
        sx = ((ndc_x / (W / H)) * 0.5 + 0.5) * W
        sy = (-ndc_y * 0.5 + 0.5) * H
    else:
        sx = (ndc_x * 0.5 + 0.5) * W
        sy = ((-ndc_y / (H / W)) * 0.5 + 0.5) * H

    in_bounds = in_view & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    return sx, sy, dist, in_bounds


def _project_orthographic(points: np.ndarray, cam: Camera):
    """Inverse of rays_orthographic. Depth = along-forward distance."""
    forward, right, up = _camera_basis(cam)
    W, H = cam.width, cam.height
    aspect = W / H
    half_w = cam.ortho_size
    half_h = cam.ortho_size / aspect

    rel = points - cam.position
    depth = rel @ forward
    x_cam = rel @ right
    y_cam = rel @ up

    sx = (x_cam / (2 * half_w) + 0.5) * W
    sy = (-y_cam / (2 * half_h) + 0.5) * H

    in_bounds = (depth > 0.05) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    return sx, sy, depth, in_bounds


def _project_equirectangular(points: np.ndarray, cam: Camera):
    """Inverse of rays_equirectangular. Depth = distance from camera."""
    forward, right, up = _camera_basis(cam)
    W, H = cam.width, cam.height

    rel = points - cam.position
    dist = np.linalg.norm(rel, axis=1).clip(EPS)
    dirs_unit = rel / dist[:, None]

    sin_lat = (dirs_unit @ up).clip(-1, 1)
    lat = np.arcsin(sin_lat)
    fwd_comp = dirs_unit @ forward
    rt_comp = dirs_unit @ right
    lon = np.arctan2(rt_comp, fwd_comp)

    sx = (lon / math.pi * 0.5 + 0.5) * W
    sy = (-lat / (math.pi / 2) * 0.5 + 0.5) * H

    in_bounds = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    return sx, sy, dist, in_bounds


LENS_PROJECTORS = {
    "pinhole":         _project_pinhole,
    "telephoto":       _project_pinhole,
    "fisheye":         _project_fisheye,
    "orthographic":    _project_orthographic,
    "equirectangular": _project_equirectangular,
}


def project_to_camera(points: np.ndarray, cam: Camera):
    """Lens-aware world→pixel projection. Dispatches on cam.lens."""
    projector = LENS_PROJECTORS.get(cam.lens, _project_pinhole)
    return projector(points, cam)


# ──────────────────────────────────────────────────────────
# LENS-AWARE FUSION
# ──────────────────────────────────────────────────────────
def fuse_bursts_pointcloud(bursts, canonical_cam: Camera, mode: str = "shaded",
                           dot_size: int = 1, bg: str = "#08101a"):
    """Multi-view fusion: aggregate hits as world point cloud, project through
    canonical_cam (any lens). Back-to-front splat."""
    all_pts, all_cols, all_norms = [], [], []
    for b in bursts:
        pts, cols, norms, _, _ = _burst_world_points(b)
        if len(pts):
            all_pts.append(pts); all_cols.append(cols); all_norms.append(norms)
    if not all_pts:
        return Image.new("RGB", (canonical_cam.width, canonical_cam.height), bg)
    pts = np.vstack(all_pts); cols = np.vstack(all_cols); norms = np.vstack(all_norms)

    sx, sy, depth, ok = project_to_camera(pts, canonical_cam)
    sx, sy = sx[ok].astype(int), sy[ok].astype(int)
    cols, norms, depth = cols[ok], norms[ok], depth[ok]

    order = np.argsort(-depth)
    sx, sy, cols, norms = sx[order], sy[order], cols[order], norms[order]

    if mode == "shaded":
        light = np.array([0.4, 0.7, 0.3]); light /= np.linalg.norm(light)
        ndotl = np.clip((norms * light).sum(axis=1), 0.3, 1.0)
        out_col = cols * ndotl[:, None]
    elif mode == "lidar":
        out_col = depth_to_color(depth[order], 0.5, 20.0)
    else:
        out_col = cols

    W, H = canonical_cam.width, canonical_cam.height
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    out_col_int = (np.clip(out_col, 0, 1) * 255).astype(np.uint8)
    for i in range(len(sx)):
        c = tuple(int(v) for v in out_col_int[i])
        x, y = int(sx[i]), int(sy[i])
        if dot_size == 1:
            img.putpixel((x, y), c)
        else:
            draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=c)
    return img


def fuse_bursts_attention(bursts, canonical_cam: Camera,
                          depth_bin: float = 0.30,
                          confidence_threshold: float = 0.15,
                          tile: int = 2,
                          alpha_gain: float = 1.5,
                          smooth_heatmap: bool = False,
                          bg_rgb: tuple = (8, 16, 26)
                          ) -> tuple:
    """Lens-aware attention-weighted fusion. Returns (color, heatmap).
    The canonical_cam can use any lens — projection dispatches on cam.lens."""
    tile = max(1, int(tile))
    depth_bin = max(float(depth_bin), EPS)
    alpha_gain = max(float(alpha_gain), EPS)
    confidence_threshold = max(float(confidence_threshold), 0.0)

    W, H = canonical_cam.width, canonical_cam.height

    all_pts, all_cols, all_weights = [], [], []
    for b in bursts:
        pts, cols, norms, depths_b, hit_idx = _burst_world_points(b)
        if not len(pts):
            continue
        source_dirs = b.dirs[hit_idx]
        align = np.abs((norms * -source_dirs).sum(axis=1)).clip(0.1, 1.0)
        depth_w = 1.0 / (1.0 + 0.05 * depths_b)
        pilot_w = max(0.1, b.pilot_score)
        weights = align * depth_w * pilot_w

        light = np.array([0.4, 0.7, 0.3]); light /= np.linalg.norm(light)
        ndotl = np.clip((norms * light).sum(axis=1), 0.3, 1.0)
        shaded = cols * ndotl[:, None]

        all_pts.append(pts); all_cols.append(shaded)
        all_weights.append(weights)

    if not all_pts:
        empty = Image.new("RGB", (W, H), bg_rgb)
        return empty, empty

    pts = np.vstack(all_pts); cols = np.vstack(all_cols)
    weights = np.concatenate(all_weights)

    # Lens-aware projection
    sx, sy, can_depth, ok = project_to_camera(pts, canonical_cam)
    sx_i = sx[ok].astype(int); sy_i = sy[ok].astype(int)
    cols, weights, can_depth = cols[ok], weights[ok], can_depth[ok]
    bucket = (can_depth / depth_bin).astype(int)

    tile_x = sx_i // tile
    tile_y = sy_i // tile

    accum = {}
    for i in range(len(sx_i)):
        key = (int(tile_x[i]), int(tile_y[i]), int(bucket[i]))
        w = weights[i]
        if key in accum:
            acc_c, acc_w, acc_d = accum[key]
            accum[key] = (acc_c + w * cols[i], acc_w + w, min(acc_d, can_depth[i]))
        else:
            accum[key] = (w * cols[i], w, can_depth[i])

    img_arr = np.full((H, W, 3), bg_rgb, dtype=np.float64) / 255.0
    conf_arr = np.zeros((H, W), dtype=np.float64)
    best_depth = {}
    bg_norm = np.array(bg_rgb) / 255.0

    for (tx, ty, _), (sw_c, sw, d) in accum.items():
        if sw < confidence_threshold:
            continue
        if (tx, ty) in best_depth and best_depth[(tx, ty)] <= d:
            continue
        color = sw_c / sw
        alpha = 1.0 - math.exp(-alpha_gain * sw)
        blended = color * alpha + bg_norm * (1.0 - alpha)
        x0, y0 = tx * tile, ty * tile
        x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
        if 0 <= x0 < W and 0 <= y0 < H:
            img_arr[y0:y1, x0:x1] = blended
            conf_arr[y0:y1, x0:x1] = sw
            best_depth[(tx, ty)] = d

    color_img = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
    heat_img = _confidence_heatmap(conf_arr, bg_rgb=bg_rgb, smooth=smooth_heatmap)
    return color_img, heat_img


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def scene_radius(prims) -> float:
    """Return a radius bounding all primitives (centers + extents) with 10% margin.
    Useful for auto-picking ortho_size so content doesn't silently clip.

    Note: uses the maximum half-extent of each primitive as a conservative
    bound. Exact for axis-aligned shapes; slightly loose for rotated boxes.
    """
    if not prims:
        return 1.0
    lows = np.array([p.center - p.half_extents.max() for p in prims])
    highs = np.array([p.center + p.half_extents.max() for p in prims])
    bb_min = lows.min(axis=0)
    bb_max = highs.max(axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    return float(diag / 2 * 1.1)


# ═════════════════════════════════════════════════════════════════════════
# WAVE EXTENSION  (v0.3.0)
# ─────────────────────────────────────────────────────────────────────────
# Sound + light + coherence + anti-coherence + cross-modal classification,
# layered onto the engine as pure post-processors over an existing Burst.
#
# DESIGN PHILOSOPHY
#   fire_burst() generates per-ray data (depth, hit primitive, pixel).
#   compute_wave_channels() takes that Burst and produces 4 phase channels
#   plus 2 intensity channels — without re-casting rays.  Subsequent
#   analyses (classify_pixels, render_channel, focal stack) are cheap.
#
# CHANNELS (all are per-pixel HxW arrays in [0,1] unless noted):
#
#   light_coh           : |Σ exp(iφ_d)|² / N²
#                         positive interference; uniform optical surfaces bright
#   light_anti          : |Σ exp(iφ_d + iφ_c)|² / N²
#                         destructive carrier; edges & far surfaces bright
#   sound_coh           : |Σ a_k exp(iφ_d)|² / (Σ|a_k|)²
#                         positive interference weighted by acoustic amplitude
#   sound_anti          : |Σ a_k exp(iφ_d + iφ_c)|² / (Σ|a_k|)²
#                         destructive carrier with acoustic weighting
#   acoustic_intensity  : mean per-pixel acoustic impedance ∈ [0,1]
#                         high = solid material, low = foliage / fabric
#   light_intensity     : per-pixel hit fraction ∈ [0,1]
#   hit_count           : raw ray hit counts per pixel
#
# CARRIER PHASE
#   φ_carrier(L) = 2π · (L_ref / L) · (frac_x − 0.5)
#   At L = L_ref the carrier sweeps exactly one cycle per pixel → rays
#   destructively cancel on uniform surfaces.  Closer → more cycles, more
#   cancellation.  Farther → sub-cycle, signal preserved.  Tune L_ref to
#   the depth band you want to *hide* (so further stuff and edges survive).
#
# WAVELENGTH TUNING
#   depth_wavelength must satisfy
#     within-pixel-depth-variation « λ_d « scene-depth-discontinuities
#   For 10 m cabin-scale scenes, λ_d in [0.5, 2.0] m works well.
# ═════════════════════════════════════════════════════════════════════════
import math as _math  # local alias used throughout the wave section


# ─────────────────────────────────────────────────────────────────────────
#  ACOUSTIC IMPEDANCE — heuristic mapping from primitive to reflectivity
#
#  Two bands are modeled here.  Audible (~1 kHz, ~34 cm wavelength) and
#  ultrasonic (~40 kHz, ~8 mm wavelength) probe different surface scales:
#
#  - HARD RIGID surfaces (metal, glass, dense stone) — reflect strongly at
#    BOTH bands; ultrasonic often slightly stronger (less surface scatter).
#  - WOOD / posts / logs — reflect well at audible, less at ultrasonic
#    (some absorption + scattering off grain).
#  - FOLIAGE / leaves — weak at both; even weaker at ultrasonic (small
#    leaves scatter or transmit short wavelengths).
#  - SOFT MATERIALS (fabric, skin, carpet) — moderate at audible, much
#    weaker at ultrasonic (fibers absorb high frequencies).
#
#  The DIFFERENCE between the two bands (audible − ultrasonic) is then a
#  *softness* signal: high difference means absorbent/fibrous material;
#  low or negative difference means hard smooth surface.
# ─────────────────────────────────────────────────────────────────────────
def acoustic_impedance(prim: Primitive) -> float:
    """Audible-band (~1 kHz) acoustic reflectivity in [0,1].

    Heuristic based on shape + color.  Override in user code by passing
    `impedance_fn=your_function` to compute_wave_channels.
    """
    r, g, b = prim.color
    if prim.shape == "sphere":
        # green-dominated → foliage → sound passes through
        if g > 1.2 * r and g > 1.2 * b:
            return 0.08
        # skin-toned (figure head) → soft fabric → weak
        if r > 0.6 and 0.4 < g < 0.7 and b < 0.55:
            return 0.20
        return 0.75  # rocks, ornamental, etc.
    if prim.shape == "cylinder":
        return 0.85  # logs, posts, trunks
    if prim.shape == "box":
        return 0.80  # roof, walls, chimney, paths
    return 0.5


def acoustic_impedance_ultrasonic(prim: Primitive) -> float:
    """Ultrasonic-band (~40 kHz) acoustic reflectivity in [0,1].

    Differs from the audible band: fibrous and soft materials absorb more
    at high frequencies; hard rigid surfaces reflect even more strongly.
    Override via `impedance_ult_fn=` on compute_wave_channels.
    """
    r, g, b = prim.color
    if prim.shape == "sphere":
        # foliage: even more transparent — short wavelengths slip through gaps
        if g > 1.2 * r and g > 1.2 * b:
            return 0.03
        # skin/fabric: fibers absorb ultrasonic
        if r > 0.6 and 0.4 < g < 0.7 and b < 0.55:
            return 0.10
        return 0.85  # rocks: slightly more reflective than audible
    if prim.shape == "cylinder":
        return 0.65  # wood absorbs more high frequencies
    if prim.shape == "box":
        return 0.92  # rigid flat surfaces near-perfect reflectors
    return 0.5


def polarization_preservation(prim: Primitive) -> float:
    """Fraction of incident polarization that the surface preserves on return.

    1.0 = perfectly polarized return (mirror, retroreflector, metal)
    0.0 = completely depolarized (heavy scattering, fibrous surface)

    This is a NEW physical axis independent of impedance: a painted metal
    panel and a varnished wooden plank can have similar acoustic signatures
    but very different polarization signatures.  Combined with the impedance
    channels, polarization helps distinguish:
      - wet vs dry roads (water has high polarization)
      - metal vs plastic (metal preserves polarization)
      - glass vs matte surfaces
      - retroreflective signs / safety vests (very high)
    """
    r, g, b = prim.color
    if prim.shape == "sphere":
        # foliage: leaves scramble polarization heavily
        if g > 1.2 * r and g > 1.2 * b:
            return 0.10
        # skin / fabric: fibrous, scrambles
        if r > 0.6 and 0.4 < g < 0.7 and b < 0.55:
            return 0.20
        return 0.55  # rocks: partial preservation
    if prim.shape == "cylinder":
        return 0.35  # wood grain partially scrambles
    if prim.shape == "box":
        return 0.65  # rigid flat surfaces preserve well
    return 0.50


# ─────────────────────────────────────────────────────────────────────────
#  MATERIAL PRIORS — optional sourced-ish defaults for test scenes
# ─────────────────────────────────────────────────────────────────────────
# These are intentionally coarse synthetic priors, not calibrated physics.
# They are useful for "material_targets" test scenes and for initializing
# custom impedance functions before empirical channel diagnostics tune them.
MATERIAL_PRIORS = {
    # piece_type: audible, ultrasonic, polarization, transparency
    "metal":          dict(audible=0.92, ultrasonic=0.96, polarization=0.85, transparency=0.00),
    "brass":          dict(audible=0.90, ultrasonic=0.94, polarization=0.78, transparency=0.00),
    "glass":          dict(audible=0.86, ultrasonic=0.94, polarization=0.80, transparency=0.25),
    "stone":          dict(audible=0.88, ultrasonic=0.92, polarization=0.45, transparency=0.00),
    "concrete":       dict(audible=0.88, ultrasonic=0.92, polarization=0.40, transparency=0.00),
    "brick":          dict(audible=0.82, ultrasonic=0.86, polarization=0.35, transparency=0.00),
    "ground":         dict(audible=0.60, ultrasonic=0.50, polarization=0.25, transparency=0.00),
    "wood":           dict(audible=0.74, ultrasonic=0.58, polarization=0.30, transparency=0.00),
    "painted_wood":   dict(audible=0.78, ultrasonic=0.64, polarization=0.42, transparency=0.00),
    "plastic":        dict(audible=0.55, ultrasonic=0.50, polarization=0.35, transparency=0.00),
    "rubber":         dict(audible=0.38, ultrasonic=0.20, polarization=0.18, transparency=0.00),
    "fabric":         dict(audible=0.30, ultrasonic=0.10, polarization=0.15, transparency=0.00),
    "cloth":          dict(audible=0.30, ultrasonic=0.10, polarization=0.15, transparency=0.00),
    "carpet":         dict(audible=0.26, ultrasonic=0.08, polarization=0.12, transparency=0.00),
    "curtain":        dict(audible=0.24, ultrasonic=0.07, polarization=0.12, transparency=0.15),
    "foliage":        dict(audible=0.08, ultrasonic=0.03, polarization=0.08, transparency=0.65),
    "canopy":         dict(audible=0.08, ultrasonic=0.03, polarization=0.08, transparency=0.65),
    "smoke":          dict(audible=0.05, ultrasonic=0.02, polarization=0.03, transparency=0.85),
}

def material_prior_value(prim: Primitive, key: str, fallback: float) -> float:
    """Return MATERIAL_PRIORS[piece_type][key] if present, else fallback."""
    rec = MATERIAL_PRIORS.get(str(prim.piece_type).lower())
    if rec is None:
        return fallback
    return float(rec.get(key, fallback))

def material_prior_acoustic(prim: Primitive) -> float:
    return material_prior_value(prim, "audible", acoustic_impedance(prim))

def material_prior_ultrasonic(prim: Primitive) -> float:
    return material_prior_value(prim, "ultrasonic", acoustic_impedance_ultrasonic(prim))

def material_prior_polarization(prim: Primitive) -> float:
    return material_prior_value(prim, "polarization", polarization_preservation(prim))

def apply_material_prior_transparency(prims: List[Primitive]) -> List[Primitive]:
    """Mutate primitives in place so known porous/transparent material types get transparency."""
    for p in prims:
        p.transparency = material_prior_value(p, "transparency", getattr(p, "transparency", 0.0))
    return prims


# ─────────────────────────────────────────────────────────────────────────
#  REALISM HELPERS — sensor noise + temporal averaging
# ─────────────────────────────────────────────────────────────────────────
def add_sensor_noise(burst: Burst,
                     range_noise_at_0m: float = 0.005,
                     range_noise_per_meter: float = 0.001,
                     dropout_fraction: float = 0.0,
                     rng=None) -> Burst:
    """Return a copy of `burst` with realistic sensor noise injected.

    range_noise_at_0m: Gaussian sigma on depth at zero range (e.g. 5 mm
                      detector jitter).
    range_noise_per_meter: additional Gaussian sigma added linearly with
                            range — real LiDAR's range noise typically
                            grows with distance.
    dropout_fraction: fraction of hits randomly converted to misses
                       (atmospheric occlusion, weak returns).

    Coherence channels degrade gracefully with this noise — the demo
    pipeline already uses these noise values cleanly.
    """
    rng = rng if rng is not None else np.random.default_rng()
    hit = burst.depths < INF
    t_new = burst.depths.copy()
    sigma = range_noise_at_0m + range_noise_per_meter * np.where(hit, t_new, 0.0)
    perturbation = rng.normal(0.0, 1.0, t_new.shape) * sigma
    t_new = t_new + perturbation * hit
    if dropout_fraction > 0:
        drop = rng.random(t_new.shape) < dropout_fraction
        t_new = np.where(hit & drop, INF, t_new)
    new_hit = t_new < INF
    new_coverage = float(new_hit.mean()) if len(t_new) else 0.0
    new_uniq = int(len(np.unique(burst.piece_ids[new_hit]))) if new_hit.any() else 0
    return Burst(
        cam=burst.cam, pixels=burst.pixels, origins=burst.origins,
        dirs=burst.dirs, depths=t_new, colors=burst.colors,
        normals=burst.normals, piece_ids=burst.piece_ids,
        coverage=new_coverage, unique_pieces=new_uniq,
    )


def stack_channels(channel_dicts: List[Dict[str, np.ndarray]]
                    ) -> Dict[str, np.ndarray]:
    """Average wave channels across multiple bursts pixelwise.

    Use this to reduce Monte-Carlo noise without firing more rays per
    burst: fire N bursts with different seeds, compute channels for each,
    then stack.  SNR improves like √N on incoherent channels.

    Channel-specific aggregation:
      hit_count        : summed (total rays across bursts)
      depth_per_pixel  : hit-count-weighted mean (bursts with more hits
                          in a pixel contribute proportionally)
      depth_variance   : unweighted mean — a small approximation since
                          the mathematically correct merge would stack
                          the raw second moments and recompute.  Fine
                          for similar-coverage bursts; worth revisiting
                          if partial_occluder becomes mission-critical.
      everything else  : unweighted pixelwise mean
    """
    if not channel_dicts:
        return {}
    keys = list(channel_dicts[0].keys())
    n = len(channel_dicts)
    out = {}
    has_hits = "hit_count" in keys
    total_hits = (sum(ch["hit_count"] for ch in channel_dicts)
                  if has_hits else None)
    for k in keys:
        if k == "hit_count":
            out[k] = total_hits
        elif k == "depth_per_pixel" and has_hits:
            # weighted by per-burst hit count — bursts with no hits in a
            # pixel contribute zero, not a misleading zero-depth average
            weighted = sum(ch["depth_per_pixel"] * ch["hit_count"]
                            for ch in channel_dicts)
            out[k] = np.where(total_hits > 0,
                               weighted / np.maximum(total_hits, 1),
                               0.0)
        else:
            out[k] = sum(ch[k] for ch in channel_dicts) / n
    return out


# ─────────────────────────────────────────────────────────────────────────
#  WAVE CHANNELS — pure post-processor over a Burst
# ─────────────────────────────────────────────────────────────────────────
def compute_wave_channels(burst: Burst,
                          prims: Optional[List[Primitive]] = None,
                          depth_wavelength: float = 1.0,
                          L_ref: float = 14.0,
                          carrier_strength: float = 1.0,
                          carrier_mode: str = "ensemble",
                          carrier_angles: Optional[List[float]] = None,
                          include_sound: bool = True,
                          include_ultrasonic: bool = False,
                          include_polarization: bool = False,
                          attenuation_per_meter: float = 0.0,
                          impedance_fn=acoustic_impedance,
                          impedance_ult_fn=acoustic_impedance_ultrasonic,
                          polarization_fn=polarization_preservation
                          ) -> Dict[str, np.ndarray]:
    """Compute coherence + anti-coherence channels from an existing Burst.

    Args:
        burst, prims, depth_wavelength, L_ref, carrier_strength: as before.
        carrier_mode: "x" for the legacy x-only carrier, "ensemble" for
            median-combined rotated carriers that suppress carrier striping
            in light_anti / sound_anti while preserving real depth edges.
        carrier_angles: optional list of carrier angles in radians used when
            carrier_mode="ensemble". Defaults to 0°, 45°, 90°, 135°.
        include_sound, include_ultrasonic, impedance_fn, impedance_ult_fn:
            audible and ultrasonic acoustic channels.
        include_polarization: add a per-pixel optical polarization-preservation
            channel.  Independent of acoustic impedance — detects
            metals/glass/retroreflectors that other channels miss.
        polarization_fn: callable(prim) → preservation ∈ [0,1].
        attenuation_per_meter: atmospheric exponential attenuation coefficient
            applied to ALL channel amplitudes.  Real LiDAR signals fall off
            with range; setting α > 0 makes far surfaces register weaker than
            near ones.  Typical values: 0.0 (no atmosphere), 0.02 (haze),
            0.10 (light fog).

    Returns dict of (H, W) numpy arrays.  Keys present:
        light_coh, light_anti, light_intensity, hit_count            (always)
        sound_coh, sound_anti, acoustic_intensity                    (sound)
        ultrasonic_coh, ultrasonic_anti, ultrasonic_intensity,
        acoustic_softness, acoustic_texture                          (ultrasonic)
        polarization                                                 (polarization)
    """
    cam = burst.cam
    W, H = cam.width, cam.height
    pixels = burst.pixels
    t = burst.depths
    pid = burst.piece_ids
    hit = t < INF

    # ── phases ──
    L = np.where(hit, t, 0.0)
    phi_depth = (2 * _math.pi * 2 * L) / depth_wavelength
    frac_x = pixels[:, 0] - np.floor(pixels[:, 0]) - 0.5
    frac_y = pixels[:, 1] - np.floor(pixels[:, 1]) - 0.5
    L_safe = np.where(hit, np.maximum(t, 0.5), L_ref)
    n_cycles = carrier_strength * L_ref / L_safe

    # Anti-coherence carrier.
    #
    # Legacy mode used only frac_x, which is fast and easy to inspect but can
    # leave visible carrier striping on broad flat surfaces.  The default
    # ensemble mode evaluates several rotated carriers and median-combines the
    # anti response.  Real depth/occlusion edges survive; carrier stripes rotate
    # with each basis and get suppressed.
    if carrier_mode == "x":
        carrier_coords = [frac_x]
    elif carrier_mode == "ensemble":
        if carrier_angles is None:
            inv_sqrt2 = 1.0 / _math.sqrt(2.0)
            carrier_coords = [
                frac_x,
                frac_y,
                (frac_x + frac_y) * inv_sqrt2,
                (frac_x - frac_y) * inv_sqrt2,
            ]
        else:
            carrier_coords = [
                _math.cos(float(theta)) * frac_x + _math.sin(float(theta)) * frac_y
                for theta in carrier_angles
            ]
            if not carrier_coords:
                carrier_coords = [frac_x]
    else:
        raise ValueError("carrier_mode must be 'x' or 'ensemble'")

    # Keep a legacy carrier available for backwards-compatible intermediate
    # variables below; final anti maps are recomputed with the ensemble after
    # pixel aggregation helpers are defined.
    phi_carrier = 2 * _math.pi * n_cycles * carrier_coords[0]

    # ── ray amplitudes ──
    # Atmospheric attenuation: each ray's amplitude decays exp(-α·t).
    # When α = 0, atten = 1 for hits and the formulas reduce to the
    # original count-normalized coherence.
    if attenuation_per_meter > 0:
        atten = np.where(hit, np.exp(-attenuation_per_meter * t), 0.0)
    else:
        atten = hit.astype(np.float64)

    light_weight = atten  # |a_k| for the light channel
    hit_complex = atten.astype(np.complex128)
    a_l_coh  = hit_complex * np.exp(1j * phi_depth)
    a_l_anti = hit_complex * np.exp(1j * (phi_depth + phi_carrier))

    if include_sound and prims is not None:
        # Per-piece impedance (a piece may contain multiple primitives —
        # we average their impedances and key by piece_id)
        id2imp_list: Dict[int, List[float]] = {}
        for p in prims:
            id2imp_list.setdefault(p.piece_id, []).append(impedance_fn(p))
        id2imp = {k: float(np.mean(v)) for k, v in id2imp_list.items()}
        acoustic_amp = np.array([id2imp.get(int(p), 0.5) for p in pid])
        # Combine intrinsic impedance with atmospheric attenuation
        sound_weight = atten * acoustic_amp
        a_s_coh  = sound_weight * np.exp(1j * phi_depth)
        a_s_anti = sound_weight * np.exp(1j * (phi_depth + phi_carrier))

        if include_ultrasonic:
            id2imp_ult_list: Dict[int, List[float]] = {}
            for p in prims:
                id2imp_ult_list.setdefault(p.piece_id, []).append(
                    impedance_ult_fn(p))
            id2imp_ult = {k: float(np.mean(v))
                           for k, v in id2imp_ult_list.items()}
            ult_amp = np.array([id2imp_ult.get(int(p), 0.5) for p in pid])
            sound_weight_ult = atten * ult_amp
            a_u_coh  = sound_weight_ult * np.exp(1j * phi_depth)
            a_u_anti = sound_weight_ult * np.exp(1j * (phi_depth + phi_carrier))
    else:
        sound_weight = None

    if include_polarization and prims is not None:
        id2pol_list: Dict[int, List[float]] = {}
        for p in prims:
            id2pol_list.setdefault(p.piece_id, []).append(polarization_fn(p))
        id2pol = {k: float(np.mean(v)) for k, v in id2pol_list.items()}
        pol_amp = np.array([id2pol.get(int(p), 0.5) for p in pid])
        pol_weight = atten * pol_amp  # attenuated polarization preservation

    # ── aggregate to pixel grid ──
    px_i = pixels[:, 0].astype(int)
    py_i = pixels[:, 1].astype(int)
    in_img = (px_i >= 0) & (px_i < W) & (py_i >= 0) & (py_i < H)
    idx = py_i[in_img] * W + px_i[in_img]

    def agg_complex(amp):
        sr = np.zeros((H, W)); si = np.zeros((H, W))
        np.add.at(sr.ravel(), idx, amp[in_img].real)
        np.add.at(si.ravel(), idx, amp[in_img].imag)
        return sr, si

    def agg_real(val):
        out = np.zeros((H, W))
        np.add.at(out.ravel(), idx, val[in_img])
        return out

    count = agg_real(hit.astype(np.float64))
    sum_light_weight = agg_real(light_weight)   # for proper normalization
    # Mean depth per pixel (only over rays that actually hit something).
    # Needed by range-aware classifiers to compensate for atmospheric
    # attenuation: closer pixels are brighter, far pixels are dimmer, and
    # the classifier can recover the original surface intensity if it
    # knows the depth.
    sum_depth = agg_real(np.where(hit, t, 0.0))
    depth_per_pixel = np.where(count > 0, sum_depth / np.maximum(count, 1), 0.0)
    # Variance of hit-ray depths within each pixel.  High variance means
    # rays in this pixel reached very different distances — a signature of
    # PARTIAL OCCLUDERS (foliage canopy with ground visible through it,
    # wire fence with stuff behind it, smoke with surface behind it).
    # Computed as E[t²] − (E[t])² using vectorized aggregation.
    sum_depth_sq = agg_real(np.where(hit, t * t, 0.0))
    mean_t_sq = np.where(count > 0, sum_depth_sq / np.maximum(count, 1), 0.0)
    depth_variance = np.maximum(mean_t_sq - depth_per_pixel ** 2, 0.0)
    lr_c, li_c = agg_complex(a_l_coh)
    norm_l = np.maximum(sum_light_weight ** 2, 1e-9)

    def anti_map(weight: np.ndarray, norm: np.ndarray) -> np.ndarray:
        """Median-combine anti-coherence maps across rotated carriers.

        This is a post-burst fix: it changes only the wave-channel math, not
        the ray hits, camera, materials, or sampling.  It suppresses carrier
        striping while preserving edge responses that are stable across carrier
        directions.
        """
        maps = []
        w_complex = weight.astype(np.complex128)
        for carrier_coord in carrier_coords:
            phi_c = 2 * _math.pi * n_cycles * carrier_coord
            amp = w_complex * np.exp(1j * (phi_depth + phi_c))
            ar, ai = agg_complex(amp)
            maps.append(np.clip((ar**2 + ai**2) / norm, 0, 1))
        if len(maps) == 1:
            return maps[0]
        return np.median(np.stack(maps, axis=0), axis=0)

    out = {
        "light_coh":       np.clip((lr_c**2 + li_c**2) / norm_l, 0, 1),
        "light_anti":      anti_map(light_weight, norm_l),
        # When attenuation is active, intensity drops with range.
        # When α = 0, sum_light_weight == count and this equals the
        # geometric hit fraction (same as before).
        "light_intensity": sum_light_weight / max(sum_light_weight.max(), 1),
        "hit_count":       count,
        "depth_per_pixel": depth_per_pixel,
        "depth_variance":  depth_variance,
    }
    if include_sound and prims is not None:
        sum_imp = agg_real(sound_weight)
        sr_c, si_c = agg_complex(a_s_coh)
        imp2 = np.maximum(sum_imp ** 2, 1e-9)
        out["sound_coh"]  = np.clip((sr_c**2 + si_c**2) / imp2, 0, 1)
        out["sound_anti"] = anti_map(sound_weight, imp2)
        out["acoustic_intensity"] = np.where(count > 0,
                                              sum_imp / np.maximum(count, 1),
                                              0.0)
        if include_ultrasonic:
            sum_imp_ult = agg_real(sound_weight_ult)
            uc_r, uc_i = agg_complex(a_u_coh)
            imp2_ult = np.maximum(sum_imp_ult ** 2, 1e-9)
            out["ultrasonic_coh"]  = np.clip((uc_r**2 + uc_i**2) / imp2_ult,
                                              0, 1)
            out["ultrasonic_anti"] = anti_map(sound_weight_ult, imp2_ult)
            out["ultrasonic_intensity"] = np.where(
                count > 0, sum_imp_ult / np.maximum(count, 1), 0.0)

            # Softness = audible − ultrasonic intensity, centered to [0,1].
            # High = soft/absorbent (foliage, fabric, skin)
            # Low  = hard smooth (metal, glass, rigid box)
            diff = out["acoustic_intensity"] - out["ultrasonic_intensity"]
            out["acoustic_softness"] = np.clip(diff + 0.5, 0, 1)

            # Texture = normalized ratio in [0,1].  0.5 = balanced.
            tot = out["acoustic_intensity"] + out["ultrasonic_intensity"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(tot > 1e-6,
                                  out["acoustic_intensity"] / np.maximum(tot, 1e-6),
                                  0.5)
            out["acoustic_texture"] = np.clip(ratio, 0, 1)

    if include_polarization and prims is not None:
        sum_pol = agg_real(pol_weight)
        out["polarization"] = np.where(count > 0,
                                        sum_pol / np.maximum(count, 1),
                                        0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────
#  CLASSIFICATION — per-pixel labels from the 4-channel signature
# ─────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "sky":              (10, 14, 22),
    "solid_surface":    (220, 200, 130),
    "geom_edge":        (230,  60,  60),
    "foliage":          ( 70, 200,  90),
    "partial_occluder": (140, 220, 100),  # like foliage but lighter
    "acoustic_only":    ( 80, 160, 230),
    "optical_only":     (210, 130, 220),
    "hard_smooth":      (200, 220, 235),
    "soft_material":    (230, 170, 110),
    "wood_material":    (160, 105,  55),
    "metal_or_glass":   (170, 220, 255),
    "stone_hard":       (150, 150, 145),
    "uncertain":        (120, 120, 120),
}


def classify_pixels(channels: Dict[str, np.ndarray],
                    foliage_intensity_max: float = 0.25,
                    foliage_texture_min: float = 0.70,
                    light_coh_min: float = 0.15,
                    edge_anti_min: float = 0.20,
                    adaptive_edge_percentile: Optional[float] = 90.0,
                    solid_coh_min: float = 0.40,
                    solid_intensity_min: float = 0.50,
                    acoustic_only_coh_min: float = 0.30,
                    optical_only_anti_min: float = 0.25,
                    hard_smooth_ult_min: float = 0.70,
                    metal_glass_ult_min: float = 0.78,
                    metal_glass_pol_min: float = 0.65,
                    stone_hard_ult_min: float = 0.76,
                    stone_hard_pol_max: float = 0.55,
                    wood_acoustic_min: float = 0.55,
                    wood_acoustic_max: float = 0.86,
                    wood_ultrasonic_min: float = 0.54,
                    wood_ultrasonic_max: float = 0.75,
                    wood_texture_min: float = 0.50,
                    wood_texture_max: float = 0.66,
                    wood_polarization_max: float = 0.55,
                    soft_material_texture_min: float = 0.60,
                    soft_material_intensity_min: float = 0.15,
                    soft_material_ult_max: float = 0.30,
                    soft_material_pol_max: float = 0.35,
                    partial_occluder_var_min: float = 0.5,
                    partial_occluder_intensity_max: float = 0.55,
                    material_labels: bool = False,
                    attenuation_per_meter: float = 0.0
                    ) -> Tuple[Image.Image, Dict[str, int]]:
    """Per-pixel classification using the wave channel signature.

    Categories in priority order (later overwrites earlier, geom_edge wins):
        sky            : no hit
        solid_surface  : both modalities return strongly and uniformly
        soft_material  : audible-to-ultrasonic texture ratio is high     [ultrasonic only]
                         — i.e., much more audible than ultrasonic return,
                         characteristic of fibrous / soft / skin / fabric
        hard_smooth    : ultrasonic intensity very high                  [ultrasonic only]
        foliage        : light returns but acoustic intensity is low
        optical_only   : light edge without acoustic edge (paint, color)
        acoustic_only  : sound returns but light coherence is weak
        geom_edge      : both anti-coherences fire (wins ties)
        uncertain      : has hit but doesn't fit any category

    Range-aware compensation:
        If `attenuation_per_meter > 0` AND the channels dict has a
        `depth_per_pixel` array, the classifier recovers the original
        surface intensity by multiplying observed A_i and U_i by
        `exp(+α·depth)`.  This stops far surfaces from being misclassified
        as "uncertain" just because atmospheric attenuation dimmed them.
        The texture ratio is already attenuation-invariant (the factor
        cancels in the ratio), so it needs no compensation.

    Adaptive edge threshold:
        If adaptive_edge_percentile is not None, the effective edge
        threshold becomes max(edge_anti_min, percentile(min(light_anti,
        sound_anti), adaptive_edge_percentile) over hit pixels).  This
        preserves the strongest anti-coherence edges while suppressing
        scene-wide carrier residue, especially in dense interiors.
    """
    L_c = channels["light_coh"]
    L_a = channels["light_anti"]
    S_c = channels.get("sound_coh")
    S_a = channels.get("sound_anti")
    A_i = channels.get("acoustic_intensity")
    U_i = channels.get("ultrasonic_intensity")
    A_tex = channels.get("acoustic_texture")
    P_i = channels.get("polarization")
    depth = channels.get("depth_per_pixel")
    hits = channels["hit_count"]
    has_hit = hits > 0

    # v0.6.2: adaptive anti-edge threshold.
    # The ensemble carrier greatly reduces striping, but dense interiors can
    # still have broad low-level anti response.  A percentile floor makes
    # geom_edge a "top anti-coherence evidence" label instead of "anything
    # above a fixed global number."  Set adaptive_edge_percentile=None to
    # recover the old fixed-threshold behavior.
    edge_anti_min_eff = float(edge_anti_min)
    if adaptive_edge_percentile is not None and np.any(has_hit):
        if S_a is not None:
            edge_score = np.minimum(L_a, S_a)
        else:
            edge_score = L_a
        valid_edge = has_hit & np.isfinite(edge_score)
        if np.any(valid_edge):
            try:
                auto_thr = float(np.percentile(edge_score[valid_edge],
                                                float(adaptive_edge_percentile)))
                edge_anti_min_eff = max(edge_anti_min_eff, auto_thr)
            except Exception:
                pass

    # Range-aware intensity compensation
    if attenuation_per_meter > 0 and depth is not None:
        comp = np.exp(attenuation_per_meter * depth)
        if A_i is not None:
            A_i = A_i * comp
        if U_i is not None:
            U_i = U_i * comp

    H, W = L_c.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., :] = CLASS_COLORS["sky"]

    if S_c is None:
        # Light-only mode
        solid = has_hit & (L_c > solid_coh_min)
        edge = has_hit & (L_a > edge_anti_min_eff)
        out[solid] = CLASS_COLORS["solid_surface"]
        out[edge]  = CLASS_COLORS["geom_edge"]
        counts = {
            "solid_surface": int(solid.sum()),
            "geom_edge":     int(edge.sum()),
            "sky":           int((~has_hit).sum()),
        }
        return Image.fromarray(out), counts

    # Full classification with cross-modal info
    # FOLIAGE: low acoustic intensity AND highly lopsided spectrum.
    # When ultrasonic data is present, additionally require A_tex > 0.70
    # so we don't sweep in low-intensity-but-not-fibrous things (like the
    # head of a figure, which sits around 0.66).
    if A_tex is not None:
        foliage = (has_hit & (A_i < foliage_intensity_max) &
                    (L_c > light_coh_min) & (A_tex > foliage_texture_min))
    else:
        foliage = has_hit & (A_i < foliage_intensity_max) & (L_c > light_coh_min)

    optical_only = has_hit & (L_a > optical_only_anti_min) & (S_a < edge_anti_min_eff) & (A_i > foliage_intensity_max)
    acoustic_only = has_hit & (S_c > acoustic_only_coh_min) & (L_c < light_coh_min) & (A_i > foliage_intensity_max)
    edge = has_hit & (L_a > edge_anti_min_eff) & (S_a > edge_anti_min_eff)
    solid = has_hit & (L_c > solid_coh_min) & (S_c > 0.10) & (A_i > solid_intensity_min)

    # PARTIAL OCCLUDER: high within-pixel depth variance AND a "diluted
    # porous" acoustic signature.  The variance gate catches "rays in
    # this pixel hit very different distances," but that condition also
    # fires on ordinary object silhouettes where rays at the boundary
    # hit foreground vs. background.  To distinguish penetrable porous
    # material (foliage, mesh, smoke) from clean geometric edges, also
    # require that the per-pixel acoustic intensity is BELOW what two
    # solid surfaces meeting at an edge would produce.
    #
    # The threshold sits between "pure leaves" (~0.10) and "edge between
    # two solid surfaces" (~0.70+).  Mixed partial-occluder pixels
    # typically have A_i around 0.30-0.55 because the low-impedance
    # canopy contribution pulls down the otherwise high
    # ground/wall intensity.  Clean cabin edges land at 0.70+
    # (high-impedance wall + ground), so they fall through to
    # geom_edge correctly.
    depth_variance = channels.get("depth_variance")
    if depth_variance is not None and A_i is not None:
        partial_occluder = (has_hit
                             & (depth_variance > partial_occluder_var_min)
                             & (A_i < partial_occluder_intensity_max))
    else:
        partial_occluder = np.zeros_like(has_hit)

    # v0.6.2 material categories.
    # The v0.6.1 soft/hard split was too blunt for material_targets: cloth
    # had high acoustic_texture like foliage, while metal/glass/stone all
    # collapsed into hard_smooth.  These rules use the diagnostic channel
    # ranges directly:
    #   cloth/fabric  -> low ultrasonic, high texture, low polarization
    #   wood          -> medium-high acoustic, medium ultrasonic/texture, low-mid polarization
    #   metal/glass   -> high ultrasonic + high polarization
    #   stone/concrete-> high ultrasonic + lower polarization than metal/glass
    if U_i is not None and A_tex is not None:
        pol = P_i if P_i is not None else np.zeros_like(U_i)

        # Generic hard-smooth stays available for all presets, but the
        # fine-grained material labels are gated by `material_labels`.
        # This prevents material-specific classes (especially stone_hard)
        # from stealing ordinary indoor/outdoor structure pixels.
        hard_smooth_base = (has_hit
                            & (U_i > hard_smooth_ult_min)
                            & (pol > 0.55)
                            & ~foliage)

        if bool(material_labels):
            metal_or_glass = (has_hit
                              & (U_i > metal_glass_ult_min)
                              & (pol > metal_glass_pol_min)
                              & ~foliage)

            stone_hard = (has_hit
                          & (U_i > stone_hard_ult_min)
                          & (pol <= stone_hard_pol_max)
                          & (A_i > solid_intensity_min)
                          & ~foliage)

            soft_material = (has_hit
                              & (A_tex > soft_material_texture_min)
                              & (A_i > soft_material_intensity_min)
                              & (A_i < foliage_intensity_max + 0.35)
                              & (U_i < soft_material_ult_max)
                              & (pol < soft_material_pol_max)
                              & ~foliage)

            wood_material = (has_hit
                             & (A_i >= wood_acoustic_min)
                             & (A_i <= wood_acoustic_max)
                             & (U_i >= wood_ultrasonic_min)
                             & (U_i <= wood_ultrasonic_max)
                             & (A_tex >= wood_texture_min)
                             & (A_tex <= wood_texture_max)
                             & (pol <= wood_polarization_max)
                             & ~foliage
                             & ~metal_or_glass
                             & ~stone_hard)
        else:
            metal_or_glass = np.zeros_like(has_hit)
            stone_hard = np.zeros_like(has_hit)
            soft_material = np.zeros_like(has_hit)
            wood_material = np.zeros_like(has_hit)

        hard_smooth = (hard_smooth_base
                       & ~metal_or_glass
                       & ~stone_hard
                       & ~soft_material
                       & ~wood_material)
    else:
        hard_smooth = np.zeros_like(has_hit)
        soft_material = np.zeros_like(has_hit)
        wood_material = np.zeros_like(has_hit)
        metal_or_glass = np.zeros_like(has_hit)
        stone_hard = np.zeros_like(has_hit)

    # ── EXCLUSIVE PRIORITY RESOLUTION ──
    # Each pixel gets exactly one label; later categories must mask out
    # earlier ones so counts are mutually exclusive and add up correctly.
    # Priority (highest first):
    #   partial_occluder → geom_edge → foliage → metal_or_glass →
    #   stone_hard → soft_material → wood_material → hard_smooth →
    #   solid_surface → optical_only → acoustic_only
    # partial_occluder wins over geom_edge because it's the more specific
    # physical interpretation of "rays in this pixel hit very different
    # distances AND the material signature is porous."  Ordinary
    # geometric edges fail the porous-material gate and fall through to
    # geom_edge correctly.
    assigned = np.zeros_like(has_hit)
    final = {}
    for name, mask in [
        ("partial_occluder", partial_occluder),  # more specific than edge
        ("geom_edge",        edge),
        ("foliage",          foliage),
        ("metal_or_glass",   metal_or_glass),
        ("stone_hard",       stone_hard),
        ("soft_material",    soft_material),
        ("wood_material",    wood_material),
        ("hard_smooth",      hard_smooth),
        ("solid_surface",    solid),
        ("optical_only",     optical_only),
        ("acoustic_only",    acoustic_only),
    ]:
        m = mask & ~assigned
        final[name] = m
        assigned |= m
    final["uncertain"] = has_hit & ~assigned

    # Apply colors — exclusive masks, so order no longer matters for output
    for name, m in final.items():
        out[m] = CLASS_COLORS[name]

    counts = {name: int(m.sum()) for name, m in final.items()}
    counts["sky"] = int((~has_hit).sum())
    return Image.fromarray(out), counts


# ─────────────────────────────────────────────────────────────────────────
#  RENDERING — channels and classification
# ─────────────────────────────────────────────────────────────────────────
def render_channel(arr: np.ndarray, smooth: bool = True,
                   gamma: float = 0.6,
                   bg: Tuple[int, int, int] = (8, 16, 26)) -> Image.Image:
    """Render any [0,1] scalar map as a viridis-style heatmap."""
    return _confidence_heatmap(np.power(np.clip(arr, 0, 1), gamma),
                                bg_rgb=bg, smooth=smooth)


def render_intensity(arr: np.ndarray, smooth: bool = True, gamma: float = 0.5,
                     color_lo: Tuple[int, int, int] = (20, 20, 60),
                     color_hi: Tuple[int, int, int] = (255, 240, 200)
                     ) -> Image.Image:
    """Render a scalar map with a custom blue→cream gradient
    (used for acoustic intensity — emphasizes solid vs transparent)."""
    a = np.power(np.clip(arr / max(arr.max(), 1e-9), 0, 1), gamma)
    lo = np.array(color_lo, float) / 255.0
    hi = np.array(color_hi, float) / 255.0
    rgb = lo[None, None, :] * (1 - a[..., None]) + hi[None, None, :] * a[..., None]
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if smooth:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def render_classification_with_legend(label_img: Image.Image,
                                       counts: Dict[str, int]) -> Image.Image:
    """Add a legend strip under the classification image.

    Uses actual text metrics (not character-count estimates) and wraps
    entries to a second row if they overflow.  The strip is sized to the
    label image's width, so the full legend is always visible.
    """
    draw_test = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    swatch_w = 14
    row_h = 22
    pad_x = 6

    # Pre-measure every entry that will be drawn
    entries = []
    for name, count in counts.items():
        if name == "sky" or count == 0:
            continue
        text = f"{name} {count}"
        try:
            bbox = draw_test.textbbox((0, 0), text)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = 8 * len(text)  # fallback for older PIL
        entries.append((name, text, text_w))

    # Layout entries into rows that fit in label_img.width
    width = label_img.width
    rows: List[List[Tuple[str, str, int]]] = []
    cur: List[Tuple[str, str, int]] = []
    cur_w = pad_x
    item_pad = 12
    for entry in entries:
        item_w = swatch_w + 4 + entry[2] + item_pad
        if cur and cur_w + item_w > width:
            rows.append(cur)
            cur = [entry]; cur_w = pad_x + item_w
        else:
            cur.append(entry); cur_w += item_w
    if cur:
        rows.append(cur)

    legend_h = max(row_h * len(rows) + 6, 28)
    out = Image.new("RGB", (width, label_img.height + legend_h), (10, 12, 18))
    out.paste(label_img, (0, 0))
    draw = ImageDraw.Draw(out)

    for r, row in enumerate(rows):
        y = label_img.height + 4 + r * row_h
        x = pad_x
        for name, text, text_w in row:
            color = CLASS_COLORS.get(name, (200, 200, 200))
            draw.rectangle([x, y, x + swatch_w, y + swatch_w], fill=color)
            draw.text((x + swatch_w + 4, y + 1), text, fill=(220, 220, 220))
            x += swatch_w + 4 + text_w + item_pad
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────
def fire_burst_wave(cam: Camera, prims: List[Primitive], n_samples: int,
                    seed: int,
                    depth_wavelength: float = 1.0,
                    L_ref: float = 14.0,
                    carrier_strength: float = 1.0,
                    carrier_mode: str = "ensemble",
                    include_sound: bool = True,
                    include_ultrasonic: bool = False,
                    include_polarization: bool = False,
                    attenuation_per_meter: float = 0.0,
                    sensor_noise: bool = False,
                    range_noise_at_0m: float = 0.005,
                    range_noise_per_meter: float = 0.001,
                    n_stacked_bursts: int = 1
                    ) -> Tuple[Burst, Dict[str, np.ndarray]]:
    """Fire one or more bursts and compute wave channels.

    Args:
        ...standard wave-channel params...
        sensor_noise: if True, inject Gaussian range jitter into depths.
        range_noise_at_0m, range_noise_per_meter: noise model parameters.
        n_stacked_bursts: if > 1, fire that many bursts (different seeds),
            compute channels for each, and average them.  Reduces
            Monte-Carlo + sensor noise like √N.

    Returns: (last_burst, channels).  When n_stacked_bursts > 1, the
    returned burst is the most-recent one (for rendering convenience);
    the channels dict is the average across all bursts.
    """
    channel_list = []
    last_burst = None
    for k in range(n_stacked_bursts):
        b = fire_burst(cam, prims, n_samples, seed=seed + k)
        if sensor_noise:
            b = add_sensor_noise(
                b, range_noise_at_0m=range_noise_at_0m,
                range_noise_per_meter=range_noise_per_meter,
                rng=np.random.default_rng(seed + k + 10_000))
        ch = compute_wave_channels(
            b, prims=prims,
            depth_wavelength=depth_wavelength,
            L_ref=L_ref, carrier_strength=carrier_strength,
            carrier_mode=carrier_mode,
            include_sound=include_sound,
            include_ultrasonic=include_ultrasonic,
            include_polarization=include_polarization,
            attenuation_per_meter=attenuation_per_meter,
        )
        channel_list.append(ch)
        last_burst = b
    channels = stack_channels(channel_list) if n_stacked_bursts > 1 else channel_list[0]
    return last_burst, channels


def focal_stack(burst: Burst, prims: List[Primitive],
                L_refs: List[float] = (4.0, 8.0, 14.0, 20.0),
                depth_wavelength: float = 1.0
                ) -> List[Dict[str, np.ndarray]]:
    """Compute one set of wave channels per L_ref value.  All from the
    same Burst — cost is just per-pixel phase math per layer."""
    return [
        compute_wave_channels(burst, prims=prims,
                              depth_wavelength=depth_wavelength,
                              L_ref=L)
        for L in L_refs
    ]


def fuse_bursts_wave_attention(burst_channel_pairs:
                                    List[Tuple[Burst, Dict[str, np.ndarray]]],
                                canonical_cam: Camera,
                                **kwargs):
    """Coherence-weighted multi-burst fusion.  Each input is a (burst,
    channels) pair from fire_burst_wave.  Bursts with cleaner geometry
    (higher mean light_coh on hit pixels) contribute more.

    Returns (composite_image, confidence_heatmap) — same shape as the
    engine's standard fuse_bursts_attention output.
    """
    for burst, ch in burst_channel_pairs:
        mask = ch["hit_count"] > 0
        mean_coh = float(ch["light_coh"][mask].mean()) if mask.any() else 0.0
        burst.pilot_score = burst.pilot_score * (0.3 + 1.7 * mean_coh)
    return fuse_bursts_attention([p[0] for p in burst_channel_pairs],
                                  canonical_cam, **kwargs)


# end of WAVE EXTENSION


# ═════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED DEMO  —  no external scene file or loader required.
# Run:   python lidar_lenses_wave.py
# Produces a contact sheet showing every wave channel + classification.
# ═════════════════════════════════════════════════════════════════════════

def _build_demo_scene() -> List[Primitive]:
    """Procedurally construct a log-cabin-style scene with trees, a path,
    and a figure — entirely from native Primitive objects.  No HTML,
    no external scraping, no STL loads.
    """
    R_id = make_rotation_matrix(0, 0, 0)
    prims: List[Primitive] = []
    next_pid = [0]

    def add(shape, center, half_extents, color, piece_type, rotation=None,
            transparency=0.0):
        rot = rotation if rotation is not None else R_id
        prims.append(Primitive(
            shape=shape,
            center=np.array(center, dtype=np.float64),
            half_extents=np.array(half_extents, dtype=np.float64),
            rotation_matrix=rot.T, inv_rotation_matrix=rot,
            color=tuple(color),
            piece_id=next_pid[0], piece_type=piece_type,
            transparency=transparency,
        ))
        next_pid[0] += 1

    # Ground plane
    add("box", [0, -0.05, 0], [12, 0.05, 12], (0.34, 0.30, 0.22), "ground")

    # Cabin walls — stacks of horizontal logs (cylinders) on four sides
    log_r = 0.16
    n_logs = 7
    wood_a = (0.42, 0.27, 0.16)
    wood_b = (0.48, 0.32, 0.20)

    for i in range(n_logs):
        y = log_r + i * (2 * log_r) * 0.92
        col = wood_a if i % 2 == 0 else wood_b
        # South wall (z = -2) with door cutout in lower 3 logs
        if i < 3:
            add("cylinder", [-1.2, y, -2.0], [log_r, 0.8, 0.0], col, "log",
                make_rotation_matrix(0, 0, 90))
            add("cylinder", [ 1.2, y, -2.0], [log_r, 0.8, 0.0], col, "log",
                make_rotation_matrix(0, 0, 90))
        else:
            add("cylinder", [0, y, -2.0], [log_r, 2.0, 0.0], col, "log",
                make_rotation_matrix(0, 0, 90))
        # North wall (z = +2) — full length
        add("cylinder", [0, y, 2.0], [log_r, 2.0, 0.0], col, "log",
            make_rotation_matrix(0, 0, 90))
        # West wall (x = -2) — logs along z, rotated about y by 90°
        col2 = wood_b if i % 2 == 0 else wood_a
        add("cylinder", [-2.0, y, 0], [log_r, 2.0, 0.0], col2, "log",
            make_rotation_matrix(90, 0, 90))
        # East wall (x = +2) — window cutout in middle of log 4
        if i == 4:
            add("cylinder", [2.0, y, -1.3], [log_r, 0.7, 0.0], col2, "log",
                make_rotation_matrix(90, 0, 90))
            add("cylinder", [2.0, y,  1.3], [log_r, 0.7, 0.0], col2, "log",
                make_rotation_matrix(90, 0, 90))
        else:
            add("cylinder", [2.0, y, 0], [log_r, 2.0, 0.0], col2, "log",
                make_rotation_matrix(90, 0, 90))

    # A-frame roof — two slanted boxes
    roof_y_base = log_r + n_logs * (2 * log_r) * 0.92
    ridge_h = 1.2
    roof_color = (0.30, 0.16, 0.10)
    add("box", [-1.0, roof_y_base + ridge_h/2, 0], [1.4, 0.08, 2.2],
        roof_color, "roof", make_rotation_matrix(0, 0, 30))
    add("box", [ 1.0, roof_y_base + ridge_h/2, 0], [1.4, 0.08, 2.2],
        roof_color, "roof", make_rotation_matrix(0, 0, -30))

    # Chimney
    add("box", [0.8, roof_y_base + ridge_h + 0.3, -0.8], [0.16, 0.40, 0.16],
        (0.55, 0.55, 0.55), "chimney")

    # Four trees with GREEN-DOMINANT canopies so the acoustic foliage
    # heuristic identifies them as low-impedance.
    for x, z, h, r in [(-5.0, 1.5, 2.0, 1.0),
                       (-4.0, -3.0, 1.6, 0.8),
                       ( 4.5, 3.0, 1.8, 0.95),
                       ( 5.0, -2.0, 2.2, 1.1)]:
        # Trunk (cylinder) — opaque
        add("cylinder", [x, h*0.5, z], [0.18, h*0.5, 0.0],
            (0.32, 0.20, 0.12), "trunk")
        # Canopy — three offset green spheres for organic shape.
        # transparency=0.65 means ~65% of rays pass through (foliage with
        # plenty of gaps between leaves), continuing on to whatever's
        # behind the canopy.  This is what makes trees register as
        # partial_occluders in the classifier.
        add("sphere", [x,         h + 0.5,       z       ], [r, r, r],
            (0.18, 0.48, 0.20), "canopy", transparency=0.65)
        add("sphere", [x + 0.5,   h + 0.7,       z + 0.3 ], [r*0.65]*3,
            (0.20, 0.50, 0.22), "canopy", transparency=0.65)
        add("sphere", [x - 0.4,   h + 0.4,       z - 0.3 ], [r*0.7]*3,
            (0.17, 0.45, 0.19), "canopy", transparency=0.65)

    # A standing figure in front of the cabin
    fx, fz = 0.0, -3.5
    add("sphere",   [fx, 1.8, fz], [0.16]*3, (0.78, 0.62, 0.50), "head")
    add("cylinder", [fx, 1.25, fz], [0.20, 0.42, 0.0],
        (0.30, 0.45, 0.62), "torso")
    add("cylinder", [fx-0.10, 0.45, fz], [0.08, 0.42, 0.0],
        (0.18, 0.20, 0.30), "leg")
    add("cylinder", [fx+0.10, 0.45, fz], [0.08, 0.42, 0.0],
        (0.18, 0.20, 0.30), "leg")

    # Stepping stones on the ground in front of the door
    for i, sz in enumerate([0.18, 0.22, 0.16, 0.20]):
        z = -2.5 - 0.6 * i
        add("sphere", [0, sz*0.5, z], [sz]*3, (0.40, 0.40, 0.42), "rock")

    # Peripheral fence posts
    for x, z in [(-6.5, 4), (-3, 5.5), (3, 5.5), (6.5, 4),
                 (-6.5, -4.5), (3.5, -5)]:
        add("cylinder", [x, 0.6, z], [0.10, 0.6, 0.0],
            (0.25, 0.18, 0.10), "post")

    return prims




def _build_material_target_board_scene() -> List[Primitive]:
    """Build a dedicated material-target board with direct visibility.

    v0.6.4: The earlier notebook material_targets scene could under-expose
    wood/metal/glass/plastic because auto-framing and dominant-hit reporting
    often saw only cloth/stone/ground/foliage.  This board presents each
    material as a front-facing slab or object with enough pixel area and mild
    depth staggering so material_channel_report can verify every material.

    Intended test materials:
      wood, metal, glass, plastic, cloth, carpet, stone, foliage, ground
    """
    R_id = make_rotation_matrix(0, 0, 0)
    prims: List[Primitive] = []
    next_pid = [0]

    def add(shape, center, half_extents, color, piece_type, rotation=None,
            transparency=0.0):
        rot = rotation if rotation is not None else R_id
        prims.append(Primitive(
            shape=shape,
            center=np.array(center, dtype=np.float64),
            half_extents=np.array(half_extents, dtype=np.float64),
            rotation_matrix=rot.T,
            inv_rotation_matrix=rot,
            color=tuple(color),
            piece_id=next_pid[0],
            piece_type=str(piece_type),
            transparency=float(transparency),
        ))
        next_pid[0] += 1

    # Low ground plane gives a stable reference but should not dominate the
    # material report because each target has a large visible face.
    add("box", [0.0, -0.05, 0.0], [10.5, 0.05, 5.5],
        (0.30, 0.30, 0.28), "ground")

    # A muted rear board adds depth but is intentionally below the material
    # thresholds and mostly hidden behind the panels.
    add("box", [0.0, 1.05, -0.65], [10.4, 1.05, 0.06],
        (0.18, 0.18, 0.20), "backdrop")

    # Front-facing target slabs, separated in X and gently staggered in Z.
    # Shape/roughness variety is included without sacrificing visibility.
    panel_y = 0.95
    panel_h = 1.30
    panel_w = 0.72
    panel_d = 0.08
    panels = [
        ("wood",    -4.20, -0.05, (0.47, 0.28, 0.12), "box"),
        ("metal",   -3.00,  0.35, (0.74, 0.76, 0.78), "box"),
        ("glass",   -1.80, -0.30, (0.62, 0.82, 0.96), "box"),
        ("plastic", -0.60,  0.55, (0.86, 0.70, 0.20), "box"),
        ("cloth",    0.60, -0.55, (0.64, 0.28, 0.24), "box"),
        ("carpet",   1.80,  0.20, (0.32, 0.22, 0.52), "box"),
        ("stone",    3.00, -0.20, (0.48, 0.48, 0.50), "box"),
    ]
    for name, x, z, color, shape in panels:
        add(shape, [x, panel_y, z], [panel_w, panel_h, panel_d],
            color, name)

    # Foliage target is a porous sphere plus a small trunk, offset so the
    # canopy remains visible and exercises transparency/depth variance.
    add("sphere", [4.20, 1.05, 0.40], [0.62, 0.62, 0.62],
        (0.16, 0.52, 0.19), "foliage", transparency=0.65)
    add("cylinder", [4.20, 0.45, 0.40], [0.08, 0.45, 0.0],
        (0.30, 0.17, 0.08), "wood")

    # Small shape samples at the foot of a few panels verify that the material
    # prior keys, not primitive shape alone, drive the material channels.
    add("cylinder", [-4.20, 0.24, 1.15], [0.18, 0.24, 0.0],
        (0.47, 0.28, 0.12), "wood")
    add("sphere", [3.00, 0.24, 1.00], [0.24, 0.24, 0.24],
        (0.48, 0.48, 0.50), "stone")
    add("box", [-3.00, 0.22, 1.05], [0.24, 0.22, 0.24],
        (0.74, 0.76, 0.78), "metal")

    apply_material_prior_transparency(prims)
    return prims


QUALITY_PRESETS = {
    "draft": dict(
        width=320, height=240, rays_per_pixel=4,
        edge_anti_min=0.30,
        label="draft (fast iteration, ~0.3M rays, tuned anti threshold)",
    ),
    "standard": dict(
        width=480, height=320, rays_per_pixel=8,
        edge_anti_min=0.30,
        label="standard (~1.2M rays, tuned anti threshold)",
    ),
    "production": dict(
        width=640, height=480, rays_per_pixel=16,
        edge_anti_min=0.35,
        label="production (clean, slow, ~4.9M rays, tuned anti threshold)",
    ),
}


def _run_demo(quality: str = "standard",
              out_dir: Optional[str] = None,
              width: Optional[int] = None,
              height: Optional[int] = None,
              rays_per_pixel: Optional[int] = None,
              L_ref: float = 14.0,
              depth_wavelength: float = 1.0,
              seed: int = 42,
              classify: bool = True,
              save_channels: bool = False,
              sensor_noise: bool = False,
              attenuation_per_meter: float = 0.0,
              n_stacked_bursts: int = 1,
              include_polarization: bool = False) -> None:
    """Build the scene, fire one (or more) bursts, render every channel.

    Realism options:
        sensor_noise: inject Gaussian range jitter (5mm + 1mm/m by default).
        attenuation_per_meter: atmospheric attenuation α (0 = none, 0.04 ≈ haze).
        n_stacked_bursts: fire N bursts and average (SNR ~√N).
        include_polarization: add a polarization-preservation channel.

    All four are independent and additive.  Run with no flags for the
    idealized previous behavior; turn any on for more realistic sensor output.
    """
    import os, time

    if quality not in QUALITY_PRESETS:
        raise ValueError(f"quality must be one of {list(QUALITY_PRESETS)}, "
                          f"got {quality!r}")
    preset = QUALITY_PRESETS[quality]
    width = width if width is not None else preset["width"]
    height = height if height is not None else preset["height"]
    rays_per_pixel = rays_per_pixel if rays_per_pixel is not None else preset["rays_per_pixel"]
    edge_anti_min = preset["edge_anti_min"]
    if out_dir is None:
        out_dir = ("wave_demo_out" if quality == "standard"
                   else f"wave_demo_out_{quality}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"quality: {preset['label']}")
    print(f"  resolution {width}×{height}, {rays_per_pixel} rays/px, "
          f"edge_anti_min={edge_anti_min}")
    print(f"  classifier: {'on' if classify else 'OFF (raw channels only)'}")
    flags = []
    if sensor_noise: flags.append("sensor_noise")
    if attenuation_per_meter > 0: flags.append(f"atten=α{attenuation_per_meter}")
    if n_stacked_bursts > 1: flags.append(f"stack={n_stacked_bursts}")
    if include_polarization: flags.append("polarization")
    if save_channels: flags.append("save_channels")
    if flags:
        print(f"  realism flags: {', '.join(flags)}")
    print(f"building procedural scene…")
    prims = _build_demo_scene()
    centers = np.array([p.center for p in prims])
    cabin_center = (centers.min(axis=0) + centers.max(axis=0)) / 2
    print(f"  {len(prims)} primitives, center {cabin_center.round(2)}")

    cam = Camera(
        position=np.array([cabin_center[0] + 12, 5.5, cabin_center[2] + 12]),
        target=cabin_center,
        fov_deg=45, width=width, height=height, lens="pinhole",
    )
    n_rays = width * height * rays_per_pixel

    AUDIBLE = {
        "ground": 0.60, "log": 0.85, "post": 0.85, "trunk": 0.85,
        "roof":   0.85, "chimney": 0.85, "rock": 0.85,
        "canopy": 0.08, "head": 0.20, "torso": 0.20, "leg": 0.20,
    }
    ULTRASONIC = {
        "ground": 0.50, "log": 0.65, "post": 0.65, "trunk": 0.65,
        "roof":   0.92, "chimney": 0.92, "rock": 0.92,
        "canopy": 0.03, "head": 0.10, "torso": 0.10, "leg": 0.10,
    }
    POLARIZATION = {
        # 1.0 = mirror, 0.0 = fully scrambled
        "ground": 0.25, "log": 0.30, "post": 0.30, "trunk": 0.30,
        "roof":   0.55, "chimney": 0.65, "rock": 0.45,
        "canopy": 0.08, "head": 0.18, "torso": 0.18, "leg": 0.18,
    }
    impedance_aud = lambda p: AUDIBLE.get(p.piece_type, 0.50)
    impedance_ult = lambda p: ULTRASONIC.get(p.piece_type, 0.50)
    pol_fn = lambda p: POLARIZATION.get(p.piece_type, 0.40)

    total_rays = n_rays * n_stacked_bursts
    print(f"firing {total_rays:,} rays "
          f"({n_stacked_bursts} burst{'s' if n_stacked_bursts>1 else ''})"
          f" + computing wave channels…")
    t0 = time.time()
    # We need polarization_fn override → can't use the convenience fire_burst_wave
    # for n_stacked_bursts > 1 with overrides, so we do the loop here.
    channel_list = []
    burst = None
    for k in range(n_stacked_bursts):
        b = fire_burst(cam, prims, n_rays, seed=seed + k)
        if sensor_noise:
            b = add_sensor_noise(b,
                rng=np.random.default_rng(seed + k + 10_000))
        ch = compute_wave_channels(
            b, prims=prims, depth_wavelength=depth_wavelength,
            L_ref=L_ref,
            include_sound=True, include_ultrasonic=True,
            include_polarization=include_polarization,
            attenuation_per_meter=attenuation_per_meter,
            impedance_fn=impedance_aud, impedance_ult_fn=impedance_ult,
            polarization_fn=pol_fn,
        )
        channel_list.append(ch)
        burst = b
    channels = stack_channels(channel_list) if n_stacked_bursts > 1 else channel_list[0]
    print(f"  burst + channels: {time.time()-t0:.2f}s")
    print(f"  coverage: {burst.coverage:.1%}")

    # Render channels (always)
    shaded   = render_burst(burst, mode="shaded")
    depth    = render_burst(burst, mode="lidar", near=5, far=20)
    l_coh    = render_channel(channels["light_coh"])
    l_anti   = render_channel(channels["light_anti"])
    s_coh    = render_channel(channels["sound_coh"])
    s_anti   = render_channel(channels["sound_anti"])
    a_int    = render_intensity(channels["acoustic_intensity"])
    u_int    = render_intensity(channels["ultrasonic_intensity"])
    soft     = render_channel(channels["acoustic_softness"])

    panels = [
        (shaded,  "1 shaded"),
        (depth,   "2 depth"),
        (l_coh,   "3 light_coh"),
        (l_anti,  "4 light_anti"),
        (s_coh,   "5 sound_coh"),
        (s_anti,  "6 sound_anti"),
        (a_int,   "7 acoustic_intensity (audible)"),
        (u_int,   "8 ultrasonic_intensity (40 kHz)"),
        (soft,    "9 acoustic_softness (audible - ultra)"),
    ]
    file_pairs = [
        ("00_shaded", shaded), ("01_depth", depth),
        ("02_light_coh", l_coh), ("03_light_anti", l_anti),
        ("04_sound_coh", s_coh), ("05_sound_anti", s_anti),
        ("06_acoustic_intensity", a_int),
        ("07_ultrasonic_intensity", u_int),
        ("08_acoustic_softness", soft),
    ]

    # Polarization panel
    if include_polarization:
        pol_img = render_channel(channels["polarization"])
        panels.append((pol_img, "POLARIZATION (mirror=high, matte=low)"))
        file_pairs.append(("polarization", pol_img))

    if classify:
        labels, counts = classify_pixels(
            channels,
            edge_anti_min=edge_anti_min,
            attenuation_per_meter=attenuation_per_meter,
        )
        cls_img = render_classification_with_legend(labels, counts)
        print(f"  classification: {counts}")
        panels.append((cls_img, "classification + legend"))
        file_pairs.append(("classification", cls_img))
    else:
        print(f"  (classification skipped — raw channels available in `channels` dict)")

    for name, img in file_pairs:
        img.save(f"{out_dir}/{name}.png")

    cols = 4 if len(panels) >= 8 else 3
    sheet = composite_grid([p[0] for p in panels],
                            [p[1] for p in panels], cols=cols)
    sheet.save(f"{out_dir}/contact_sheet.png")
    print(f"saved → {out_dir}/contact_sheet.png")

    if save_channels:
        out_path = f"{out_dir}/channels.npz"
        np.savez(out_path, **{k: v for k, v in channels.items()})
        print(f"saved → {out_path}")
        print(f"  arrays: {list(channels.keys())}")


# ═════════════════════════════════════════════════════════════════════════
# v0.6.2 SENSOR PRESETS — consistent scene test packs
# ─────────────────────────────────────────────────────────────────────────
# These helpers are intentionally optional.  The core engine remains:
#   fire_burst() → compute_wave_channels() → classify_pixels()
# Presets simply bundle camera choice, sampling, stack count, channel panels,
# and classifier thresholds for common use cases.
# ═════════════════════════════════════════════════════════════════════════

def _copy_preset(d: Dict) -> Dict:
    """Tiny dependency-free deep-ish copy for preset dictionaries."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _copy_preset(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


SENSOR_PRESETS: Dict[str, Dict] = {
    "overhead_layout": {
        "description": "Clean top-down map for layout, spacing, navigation, and scene repair.",
        "camera": "topdown",
        "wave": False,
        "lens": "orthographic",
        "sampling_mode": "grid",
        "width": 900,
        "height": 900,
        "rays_per_pixel": 1,
        "render_modes": ["material", "shaded"],
        "auto_frame": False,
    },
    "indoor_structure": {
        "description": "Eye-level interior sensing. Emphasizes light_anti + sound_anti for dense furniture/room structure.",
        "camera": "front_eyelevel",
        "auto_frame": True,
        "min_coverage": 0.35,
        "min_depth_span": 3.0,
        "wave": True,
        "lens": "pinhole",
        "sampling_mode": "halton",
        "width": 480,
        "height": 320,
        "fov_deg": 70.0,
        "rays_per_pixel": 8,
        "stack": 4,
        "carrier_mode": "ensemble",
        "carrier_strength": 0.50,
        "depth_wavelength": 1.0,
        "edge_anti_min": 0.38,
        "adaptive_edge_percentile": 94.0,
        "include_ultrasonic": True,
        "include_polarization": False,
        "attenuation_per_meter": 0.0,
        "material_labels": False,
        "panels": [
            "shaded", "depth", "light_anti", "sound_anti",
            "acoustic_intensity", "ultrasonic_intensity",
            "depth_variance", "classification",
        ],
    },
    "outdoor_occlusion": {
        "description": "Exterior/foliage/fence/smoke style sensing. Highlights partial occluders and mixed-depth pixels.",
        "camera": "corner_eyelevel",
        "auto_frame": True,
        "min_coverage": 0.35,
        "min_depth_span": 3.0,
        "wave": True,
        "lens": "pinhole",
        "sampling_mode": "halton",
        "width": 480,
        "height": 320,
        "fov_deg": 55.0,
        "rays_per_pixel": 8,
        "stack": 4,
        "carrier_mode": "ensemble",
        "carrier_strength": 0.50,
        "depth_wavelength": 1.0,
        "edge_anti_min": 0.34,
        "adaptive_edge_percentile": 92.0,
        "include_ultrasonic": True,
        "include_polarization": False,
        "attenuation_per_meter": 0.0,
        "material_labels": False,
        "panels": [
            "shaded", "depth", "light_anti", "sound_anti",
            "acoustic_intensity", "acoustic_softness",
            "depth_variance", "classification",
        ],
    },
    "material_scan": {
        "description": "Material board / object inspection. Emphasizes acoustic, ultrasonic, softness, and polarization cues.",
        "camera": "front_eyelevel",
        "auto_frame": True,
        "min_coverage": 0.35,
        "min_depth_span": 3.0,
        "wave": True,
        "lens": "pinhole",
        "sampling_mode": "halton",
        "width": 480,
        "height": 320,
        "fov_deg": 50.0,
        "rays_per_pixel": 8,
        "stack": 4,
        "carrier_mode": "ensemble",
        "carrier_strength": 0.30,
        "depth_wavelength": 1.0,
        "edge_anti_min": 0.30,
        "adaptive_edge_percentile": 94.0,
        "include_ultrasonic": True,
        "include_polarization": True,
        "attenuation_per_meter": 0.0,
        "material_labels": True,
        "panels": [
            "shaded", "depth", "acoustic_intensity", "ultrasonic_intensity",
            "acoustic_softness", "acoustic_texture", "polarization", "classification",
        ],
    },
    "full_diagnostic": {
        "description": "Large all-channel diagnostic sheet. Best for debugging, not compact agent input.",
        "camera": "corner_eyelevel",
        "auto_frame": True,
        "min_coverage": 0.35,
        "min_depth_span": 3.0,
        "wave": True,
        "lens": "pinhole",
        "sampling_mode": "halton",
        "width": 640,
        "height": 480,
        "fov_deg": 55.0,
        "rays_per_pixel": 12,
        "stack": 4,
        "carrier_mode": "ensemble",
        "carrier_strength": 1.0,
        "depth_wavelength": 1.0,
        "edge_anti_min": 0.35,
        "adaptive_edge_percentile": 92.0,
        "include_ultrasonic": True,
        "include_polarization": True,
        "attenuation_per_meter": 0.0,
        "material_labels": False,
        "panels": [
            "shaded", "depth", "light_coh", "light_anti",
            "sound_coh", "sound_anti", "acoustic_intensity", "ultrasonic_intensity",
            "acoustic_softness", "acoustic_texture", "depth_variance",
            "polarization", "classification",
        ],
    },
}


def list_sensor_presets() -> List[str]:
    """Return available preset names."""
    return sorted(SENSOR_PRESETS.keys())


def get_sensor_preset(name: str = "indoor_structure", **overrides) -> Dict:
    """Return a copy of a sensor preset, with optional key overrides.

    Example:
        cfg = get_sensor_preset("indoor_structure", edge_anti_min=0.38)
    """
    if name not in SENSOR_PRESETS:
        raise KeyError(f"unknown sensor preset {name!r}; available: {list_sensor_presets()}")
    cfg = _copy_preset(SENSOR_PRESETS[name])
    cfg.update(overrides)
    return cfg


def scene_bounds(prims: List[Primitive]) -> Dict[str, np.ndarray]:
    """Scene bounds for primitives. Returns min/max/center/span.

    v0.6.2 note: this uses per-axis extents instead of max-half-extent as a
    sphere on every axis.  The old conservative sphere made flat ground planes
    look dozens of meters tall, which could place auto-framing cameras below
    the scene.  Rotated boxes/cylinders still use conservative transformed
    local AABBs, but without inflating the vertical span unnecessarily.
    """
    if not prims:
        mn = np.array([-1.0, 0.0, -1.0])
        mx = np.array([ 1.0, 1.0,  1.0])
    else:
        lows = []
        highs = []
        for p in prims:
            if p.shape == "sphere":
                ext = np.array([p.half_extents[0]] * 3, dtype=np.float64)
            elif p.shape == "cylinder":
                # local bounding box for a vertical cylinder: radius in X/Z, half-height in Y
                r = float(p.half_extents[0]); hh = float(p.half_extents[1])
                local_ext = np.array([r, hh, r], dtype=np.float64)
                ext = np.abs(p.inv_rotation_matrix) @ local_ext
            else:
                local_ext = np.asarray(p.half_extents, dtype=np.float64)
                ext = np.abs(p.inv_rotation_matrix) @ local_ext
            lows.append(p.center - ext)
            highs.append(p.center + ext)
        mn = np.vstack(lows).min(axis=0)
        mx = np.vstack(highs).max(axis=0)
    center = (mn + mx) / 2.0
    span = np.maximum(mx - mn, 1e-6)
    return {"min": mn, "max": mx, "center": center, "span": span}


def burst_depth_stats(burst: Burst) -> Dict[str, float]:
    """Coverage and robust depth stats for diagnostics."""
    hit = burst.depths < INF
    out = {
        "coverage": float(hit.mean()) if len(hit) else 0.0,
        "hit_rays": int(hit.sum()),
        "total_rays": int(len(hit)),
        "depth_p05": None,
        "depth_p50": None,
        "depth_p95": None,
        "depth_span_p05_p95": 0.0,
    }
    if hit.any():
        d = burst.depths[hit]
        p05, p50, p95 = np.percentile(d, [5, 50, 95])
        out.update({
            "depth_p05": float(p05),
            "depth_p50": float(p50),
            "depth_p95": float(p95),
            "depth_span_p05_p95": float(p95 - p05),
        })
    return out


def estimate_L_ref_from_burst(burst: Burst, fallback: float = 8.0,
                              percentile: float = 50.0) -> float:
    """Pick a carrier reference depth from visible hit depths."""
    hit = burst.depths < INF
    if not hit.any():
        return float(fallback)
    return float(np.percentile(burst.depths[hit], percentile))


def make_camera_for_preset(prims: List[Primitive],
                           preset_name: str = "indoor_structure",
                           width: Optional[int] = None,
                           height: Optional[int] = None,
                           position: Optional[np.ndarray] = None,
                           target: Optional[np.ndarray] = None,
                           up: Optional[np.ndarray] = None,
                           **preset_overrides) -> Camera:
    """Build a camera from a preset and scene bounds.

    The camera choices are intentionally simple and robust:
      - topdown: orthographic map camera, excellent for layout
      - front_eyelevel: human-height view from +Z looking inward
      - corner_eyelevel: diagonal/corner perspective for depth variation
    """
    cfg = get_sensor_preset(preset_name, **preset_overrides)
    b = scene_bounds(prims)
    mn, mx, c, span = b["min"], b["max"], b["center"], b["span"]
    W = int(width if width is not None else cfg.get("width", 480))
    H = int(height if height is not None else cfg.get("height", 320))
    cam_kind = cfg.get("camera", "front_eyelevel")
    lens = cfg.get("lens", "pinhole")
    sampling = cfg.get("sampling_mode", "halton")
    fov = float(cfg.get("fov_deg", 60.0))

    if target is None:
        target = np.array([c[0], mn[1] + min(max(span[1] * 0.35, 1.0), 1.8), c[2]], dtype=np.float64)
    else:
        target = np.asarray(target, dtype=np.float64)

    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.asarray(up, dtype=np.float64)

    if cam_kind == "topdown":
        max_xz = float(max(span[0], span[2]))
        ortho = max(1.0, max_xz * 0.58)
        pos = np.array([c[0], mx[1] + max(max_xz, span[1]) + 4.0, c[2]], dtype=np.float64)
        tgt = np.array([c[0], mn[1], c[2]], dtype=np.float64)
        return Camera(
            position=pos, target=tgt, up=np.array([0.0, 0.0, -1.0]),
            width=W, height=H, lens="orthographic", ortho_size=ortho,
            sampling_mode=sampling,
        )

    if position is not None:
        pos = np.asarray(position, dtype=np.float64)
    elif cam_kind == "corner_eyelevel":
        pos = np.array([
            mx[0] + max(span[0] * 0.45, 2.5),
            mn[1] + 1.65,
            mx[2] + max(span[2] * 0.55, 3.5),
        ], dtype=np.float64)
    else:  # front_eyelevel
        pos = np.array([
            c[0],
            mn[1] + 1.65,
            mx[2] + max(span[2] * 0.70, 4.0),
        ], dtype=np.float64)

    return Camera(
        position=pos, target=target, up=up, fov_deg=fov,
        width=W, height=H, lens=lens, sampling_mode=sampling,
    )


def _normalize_channel_for_render(arr: np.ndarray) -> np.ndarray:
    """Robust [0,1] normalization for depth-like arrays with zero background."""
    arr = np.asarray(arr, dtype=np.float64)
    mask = arr > 0
    if not mask.any():
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[mask], [2, 98])
    if hi <= lo + EPS:
        return np.where(mask, 1.0, 0.0)
    return np.clip((arr - lo) / (hi - lo), 0, 1)



# Classifier thresholds that can be exposed in notebooks/spreadsheets.
CLASSIFIER_KWARG_NAMES = [
    "foliage_intensity_max",
    "foliage_texture_min",
    "light_coh_min",
    "edge_anti_min",
    "adaptive_edge_percentile",
    "solid_coh_min",
    "solid_intensity_min",
    "acoustic_only_coh_min",
    "optical_only_anti_min",
    "hard_smooth_ult_min",
    "metal_glass_ult_min",
    "metal_glass_pol_min",
    "stone_hard_ult_min",
    "stone_hard_pol_max",
    "wood_acoustic_min",
    "wood_acoustic_max",
    "wood_ultrasonic_min",
    "wood_ultrasonic_max",
    "wood_texture_min",
    "wood_texture_max",
    "wood_polarization_max",
    "soft_material_texture_min",
    "soft_material_intensity_min",
    "soft_material_ult_max",
    "soft_material_pol_max",
    "partial_occluder_var_min",
    "partial_occluder_intensity_max",
    "material_labels",
    "attenuation_per_meter",
]

def classifier_kwargs_from_config(cfg: Dict) -> Dict:
    """Extract valid classify_pixels kwargs from a preset/sweep config dict."""
    out = {}
    for k in CLASSIFIER_KWARG_NAMES:
        if k in cfg and cfg[k] is not None:
            if k == "material_labels":
                v = cfg[k]
                if isinstance(v, str):
                    out[k] = v.strip().lower() in ("1", "true", "yes", "y", "on")
                else:
                    out[k] = bool(v)
                continue
            try:
                out[k] = float(cfg[k])
            except (TypeError, ValueError):
                out[k] = cfg[k]
    # Preserve old defaults when not specified.
    out.setdefault("edge_anti_min", float(cfg.get("edge_anti_min", 0.30)))
    out.setdefault("adaptive_edge_percentile", cfg.get("adaptive_edge_percentile", 90.0))
    out.setdefault("material_labels", bool(cfg.get("material_labels", False)))
    out.setdefault("attenuation_per_meter", float(cfg.get("attenuation_per_meter", 0.0)))
    return out


def dominant_piece_type_masks(burst: Burst,
                              prims: List[Primitive],
                              min_count: int = 1) -> Dict[str, np.ndarray]:
    """Return per-piece_type pixel masks based on dominant hit counts in a burst.

    This is intended for diagnostics, not rendering.  It lets test notebooks
    compute per-material channel means without adding new arrays to Burst.
    """
    W, H = burst.cam.width, burst.cam.height
    hit = burst.depths < INF
    if not hit.any():
        return {}
    pid_to_type = {int(p.piece_id): str(p.piece_type) for p in prims}
    piece_types = sorted(set(pid_to_type.values()))
    if not piece_types:
        return {}
    type_counts = {t: np.zeros((H, W), dtype=np.float64) for t in piece_types}
    px = burst.pixels[hit, 0].astype(int)
    py = burst.pixels[hit, 1].astype(int)
    pid = burst.piece_ids[hit].astype(int)
    in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, pid = px[in_img], py[in_img], pid[in_img]
    for t in piece_types:
        mask = np.array([pid_to_type.get(int(x), None) == t for x in pid], dtype=bool)
        if mask.any():
            np.add.at(type_counts[t], (py[mask], px[mask]), 1.0)
    stack = np.stack([type_counts[t] for t in piece_types], axis=0)
    max_count = stack.max(axis=0)
    winner = stack.argmax(axis=0)
    masks = {}
    for i, t in enumerate(piece_types):
        masks[t] = (winner == i) & (max_count >= min_count)
    return masks


def material_channel_report(burst: Burst,
                            channels: Dict[str, np.ndarray],
                            prims: List[Primitive],
                            min_pixels: int = 5) -> List[Dict]:
    """Summarize raw channel values by dominant piece_type/material.

    Returns a list of dicts suitable for CSV/JSON.  If materials overlap too
    much here, threshold sweeps cannot separate them reliably.
    """
    masks = dominant_piece_type_masks(burst, prims)
    rows = []
    channel_names = [
        "acoustic_intensity", "ultrasonic_intensity", "acoustic_texture",
        "acoustic_softness", "polarization", "depth_variance",
        "light_coh", "light_anti", "sound_coh", "sound_anti",
    ]
    for piece_type, mask in sorted(masks.items()):
        n = int(mask.sum())
        if n < int(min_pixels):
            continue
        row = {"piece_type": piece_type, "pixel_count": n}
        for ch_name in channel_names:
            if ch_name in channels:
                vals = np.asarray(channels[ch_name])[mask]
                row[ch_name + "_mean"] = float(np.nanmean(vals)) if vals.size else 0.0
                row[ch_name + "_p50"] = float(np.nanpercentile(vals, 50)) if vals.size else 0.0
                row[ch_name + "_p90"] = float(np.nanpercentile(vals, 90)) if vals.size else 0.0
        rows.append(row)
    return rows


def _render_panel(panel_name: str,
                  burst: Burst,
                  channels: Optional[Dict[str, np.ndarray]],
                  prims: List[Primitive],
                  cfg: Dict) -> Tuple[Image.Image, str]:
    """Render one named preset panel."""
    name = panel_name
    if name == "shaded":
        return render_burst(burst, mode="shaded", bg="#08101a"), "shaded"
    if name == "material":
        return render_burst(burst, mode="material", bg="#08101a"), "material"
    if name == "depth":
        return render_burst(burst, mode="lidar", bg="#08101a", auto_calibrate=True), "depth"
    if channels is None:
        return render_burst(burst, mode="shaded", bg="#08101a"), name
    if name == "classification":
        labels, counts = classify_pixels(channels, **classifier_kwargs_from_config(cfg))
        return render_classification_with_legend(labels, counts), "classification"
    if name in ("acoustic_intensity", "ultrasonic_intensity"):
        return render_intensity(channels.get(name, np.zeros_like(channels["hit_count"]))), name
    if name in channels:
        arr = channels[name]
        if name in ("depth_per_pixel", "depth_variance"):
            arr = _normalize_channel_for_render(arr)
        return render_channel(arr), name
    # Unknown/missing channel: return blank but label it clearly
    W, H = burst.cam.width, burst.cam.height
    img = Image.new("RGB", (W, H), (8, 16, 26))
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), f"missing: {name}", fill=(220, 220, 220))
    return img, f"missing {name}"




def _camera_score_from_burst(burst: Burst,
                             min_coverage: float = 0.35,
                             min_depth_span: float = 3.0) -> Dict:
    """Compact camera health score used by preset auto-framing."""
    stats = burst_depth_stats(burst)
    cov = float(stats.get("coverage", 0.0))
    span = float(stats.get("depth_span_p05_p95", 0.0))
    uniq = int(getattr(burst, "unique_pieces", 0))
    # Coverage is primary; depth span and unique piece count break ties.
    cov_bonus = min(cov / max(min_coverage, EPS), 1.5)
    span_bonus = min(span / max(min_depth_span, EPS), 1.5)
    score = 0.62 * cov_bonus + 0.28 * span_bonus + 0.10 * min(_math.log1p(uniq) / 4.0, 1.0)
    # Penalize extremely low coverage hard; mostly-empty contact sheets are useless.
    if cov < min_coverage * 0.5:
        score *= 0.45
    return {
        "score": float(score),
        "coverage": cov,
        "depth_span_p05_p95": span,
        "unique_pieces": uniq,
    }


def _make_candidate_camera(prims: List[Primitive],
                           cfg: Dict,
                           position: np.ndarray,
                           target: np.ndarray,
                           width: int,
                           height: int,
                           fov_deg: float) -> Camera:
    """Internal candidate pinhole camera for auto-framing."""
    return Camera(
        position=np.asarray(position, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=float(fov_deg),
        width=int(width),
        height=int(height),
        lens=cfg.get("lens", "pinhole"),
        sampling_mode=cfg.get("sampling_mode", "halton"),
    )


def auto_frame_camera_for_preset(prims: List[Primitive],
                                 preset_name: str = "indoor_structure",
                                 seed: int = 42,
                                 pilot_rays: int = 30000,
                                 **preset_overrides) -> Tuple[Camera, Dict]:
    """Choose a robust preset camera by trying a few cheap pilot bursts.

    This is intentionally conservative: it does not change the raw engine or
    wave math.  It only prevents bad preset contact sheets caused by a camera
    that under-frames a scene.  If the first preset camera already has healthy
    coverage/depth span, it wins.  Otherwise, the best candidate from several
    front/back/side/corner eye-level views is chosen.

    Returns: (best_camera, diagnostics_dict)
    """
    cfg = get_sensor_preset(preset_name, **preset_overrides)
    base_cam = make_camera_for_preset(prims, preset_name, **preset_overrides)
    if not bool(cfg.get("auto_frame", False)) or not bool(cfg.get("wave", True)):
        return base_cam, {"enabled": False, "reason": "disabled_or_nonwave"}

    b = scene_bounds(prims)
    mn, mx, c, span = b["min"], b["max"], b["center"], b["span"]
    W, H = int(base_cam.width), int(base_cam.height)
    fov0 = float(cfg.get("fov_deg", getattr(base_cam, "fov_deg", 60.0)))
    min_cov = float(cfg.get("min_coverage", 0.35))
    min_span = float(cfg.get("min_depth_span", 3.0))
    pilot_rays = int(max(4000, min(pilot_rays, W * H)))

    target_y_options = [
        mn[1] + min(max(span[1] * 0.35, 1.0), 1.8),
        mn[1] + min(max(span[1] * 0.50, 1.1), 2.2),
    ]
    max_xz = float(max(span[0], span[2], 1.0))
    eye_y = float(mn[1] + 1.65)

    # Direction vectors in XZ. Start with the preset camera direction first,
    # then try cardinal and diagonal views.  This lets saloons/interiors find
    # an open wall/doorway view without scene-specific code.
    base_dir = np.array([base_cam.position[0] - c[0], 0.0, base_cam.position[2] - c[2]], dtype=np.float64)
    if np.linalg.norm(base_dir[[0,2]]) < EPS:
        base_dir = np.array([0.0, 0.0, 1.0])
    directions = [
        base_dir,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, 0.0, -1.0]),
    ]
    dist_factors = [0.65, 0.90, 1.20]
    fovs = sorted(set([fov0, min(max(fov0 + 12.0, 55.0), 85.0), 85.0]))

    candidates: List[Camera] = [base_cam]
    for ty in target_y_options:
        tgt = np.array([c[0], ty, c[2]], dtype=np.float64)
        for d in directions:
            d = np.asarray(d, dtype=np.float64)
            d[1] = 0.0
            n = np.linalg.norm(d)
            if n < EPS:
                continue
            d = d / n
            for df in dist_factors:
                dist = max(max_xz * df, 2.5)
                pos = np.array([c[0] + d[0] * dist, eye_y, c[2] + d[2] * dist], dtype=np.float64)
                for fov in fovs:
                    candidates.append(_make_candidate_camera(prims, cfg, pos, tgt, W, H, fov))

    # Deduplicate near-identical candidates.
    unique = []
    seen = set()
    for cam in candidates:
        key = tuple(np.round(np.r_[cam.position, cam.target, cam.fov_deg], 3))
        if key not in seen:
            seen.add(key); unique.append(cam)

    best_cam = None
    best_info = None
    trials = []
    for i, cam in enumerate(unique):
        try:
            pilot = fire_burst(cam, prims, n_samples=pilot_rays, seed=seed + i)
            info = _camera_score_from_burst(pilot, min_cov, min_span)
            info.update({
                "index": i,
                "position": [float(x) for x in cam.position],
                "target": [float(x) for x in cam.target],
                "fov_deg": float(cam.fov_deg),
            })
            trials.append(info)
            if best_info is None or info["score"] > best_info["score"]:
                best_info = info; best_cam = cam
        except Exception as exc:
            trials.append({"index": i, "error": str(exc), "score": -1.0})

    if best_cam is None:
        return base_cam, {"enabled": True, "error": "all_candidates_failed"}

    passed = (best_info["coverage"] >= min_cov and
              best_info["depth_span_p05_p95"] >= min_span)
    return best_cam, {
        "enabled": True,
        "selected": best_info,
        "passed": bool(passed),
        "min_coverage": min_cov,
        "min_depth_span": min_span,
        "pilot_rays": pilot_rays,
        "n_candidates": len(unique),
        "top_trials": sorted(trials, key=lambda x: x.get("score", -1.0), reverse=True)[:8],
    }

def run_sensor_preset(prims: List[Primitive],
                      preset_name: str = "indoor_structure",
                      scene_name: str = "scene",
                      out_dir: Optional[str] = None,
                      seed: int = 42,
                      L_ref: str = "auto",
                      impedance_fn=acoustic_impedance,
                      impedance_ult_fn=acoustic_impedance_ultrasonic,
                      polarization_fn=polarization_preservation,
                      **preset_overrides) -> Dict:
    """Run one named sensor preset and optionally save a contact sheet + diagnostics.

    Returns a dict containing:
      contact_sheet, burst, channels, diagnostics, paths

    Example:
        result = run_sensor_preset(prims, "indoor_structure", out_dir="outputs")
        result["contact_sheet"].show()
    """
    cfg = get_sensor_preset(preset_name, **preset_overrides)
    cam = make_camera_for_preset(prims, preset_name, **preset_overrides)
    frame_diagnostics = {"enabled": False}
    if bool(cfg.get("auto_frame", False)) and bool(cfg.get("wave", True)):
        cam, frame_diagnostics = auto_frame_camera_for_preset(
            prims, preset_name=preset_name, seed=seed, **preset_overrides
        )
    n_samples = int(cam.width * cam.height * cfg.get("rays_per_pixel", 1))
    t0 = __import__("time").time()

    paths = {}
    channels = None
    channel_list = []
    last_burst = None

    if not bool(cfg.get("wave", True)):
        burst = fire_burst(cam, prims, n_samples=n_samples, seed=seed)
        last_burst = burst
        imgs, labels = [], []
        for mode in cfg.get("render_modes", ["material", "shaded"]):
            if mode == "material":
                imgs.append(render_burst(burst, mode="material", bg="#d8b487"))
                labels.append("topdown material")
            elif mode == "shaded":
                imgs.append(render_burst(burst, mode="shaded", bg="#d8b487"))
                labels.append("topdown shaded")
            else:
                imgs.append(render_burst(burst, mode=mode))
                labels.append(mode)
        sheet = composite_grid(imgs, labels, cols=len(imgs), bg="#101010")
    else:
        stack = int(cfg.get("stack", 1))
        # Pilot for a stable L_ref, then reuse it across the stack.
        pilot = fire_burst(cam, prims, n_samples=n_samples, seed=seed)
        lref_value = estimate_L_ref_from_burst(pilot) if L_ref == "auto" else float(L_ref)
        for k in range(stack):
            b = pilot if k == 0 else fire_burst(cam, prims, n_samples=n_samples, seed=seed + k)
            ch = compute_wave_channels(
                b, prims=prims,
                depth_wavelength=float(cfg.get("depth_wavelength", 1.0)),
                L_ref=lref_value,
                carrier_strength=float(cfg.get("carrier_strength", 1.0)),
                carrier_mode=cfg.get("carrier_mode", "ensemble"),
                include_sound=True,
                include_ultrasonic=bool(cfg.get("include_ultrasonic", True)),
                include_polarization=bool(cfg.get("include_polarization", False)),
                attenuation_per_meter=float(cfg.get("attenuation_per_meter", 0.0)),
                impedance_fn=impedance_fn,
                impedance_ult_fn=impedance_ult_fn,
                polarization_fn=polarization_fn,
            )
            channel_list.append(ch)
            last_burst = b
        channels = stack_channels(channel_list) if len(channel_list) > 1 else channel_list[0]

        imgs, labels = [], []
        for panel in cfg.get("panels", ["shaded", "depth", "light_anti", "sound_anti", "classification"]):
            im, lab = _render_panel(panel, last_burst, channels, prims, cfg)
            imgs.append(im); labels.append(lab)
        cols = 4 if len(imgs) >= 8 else min(3, max(1, len(imgs)))
        sheet = composite_grid(imgs, labels, cols=cols)

    stats = burst_depth_stats(last_burst)
    counts = None
    if channels is not None:
        _, counts = classify_pixels(channels, **classifier_kwargs_from_config(cfg))

    diagnostics = {
        "scene": scene_name,
        "preset": preset_name,
        "description": cfg.get("description", ""),
        "width": int(cam.width),
        "height": int(cam.height),
        "lens": cam.lens,
        "camera_position": [float(x) for x in cam.position],
        "camera_target": [float(x) for x in cam.target],
        "sampling_mode": cam.sampling_mode,
        "rays_per_pixel": int(cfg.get("rays_per_pixel", 1)),
        "stacked_bursts": int(cfg.get("stack", 1)) if cfg.get("wave", True) else 1,
        "n_samples_per_burst": int(n_samples),
        "wave": bool(cfg.get("wave", True)),
        "L_ref": (estimate_L_ref_from_burst(last_burst) if L_ref == "auto" and channels is None else
                  (estimate_L_ref_from_burst(last_burst) if L_ref == "auto" else float(L_ref))),
        "carrier_mode": cfg.get("carrier_mode", None),
        "edge_anti_min": cfg.get("edge_anti_min", None),
        "adaptive_edge_percentile": cfg.get("adaptive_edge_percentile", None),
        "classifier_kwargs": classifier_kwargs_from_config(cfg) if channels is not None else {},
        "auto_frame": frame_diagnostics,
        "depth_stats": stats,
        "classification_counts": counts,
        "runtime_seconds": round(__import__("time").time() - t0, 3),
    }

    if out_dir is not None:
        import os as _os
        import json as _json
        _os.makedirs(out_dir, exist_ok=True)
        safe_scene = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in scene_name)
        base = f"{safe_scene}_{preset_name}"
        img_path = _os.path.join(out_dir, base + "_contact_sheet.png")
        diag_path = _os.path.join(out_dir, base + "_diagnostics.json")
        sheet.save(img_path)
        with open(diag_path, "w", encoding="utf-8") as f:
            _json.dump(diagnostics, f, indent=2)
        paths["contact_sheet"] = img_path
        paths["diagnostics"] = diag_path
        if channels is not None:
            npz_path = _os.path.join(out_dir, base + "_channels.npz")
            np.savez(npz_path, **channels)
            paths["channels"] = npz_path

    return {
        "contact_sheet": sheet,
        "burst": last_burst,
        "channels": channels,
        "diagnostics": diagnostics,
        "material_report": material_channel_report(last_burst, channels, prims) if channels is not None else [],
        "paths": paths,
    }


def run_scene_preset_pack(prims: List[Primitive],
                          scene_name: str = "scene",
                          presets: Optional[List[str]] = None,
                          out_dir: Optional[str] = None,
                          seed: int = 42,
                          **overrides) -> Dict[str, Dict]:
    """Run several presets on one scene. Useful default pack for agent debugging."""
    if presets is None:
        presets = ["overhead_layout", "indoor_structure", "material_scan"]
    results = {}
    for i, preset in enumerate(presets):
        results[preset] = run_sensor_preset(
            prims, preset_name=preset, scene_name=scene_name,
            out_dir=out_dir, seed=seed + i * 100, **overrides
        )
    return results



if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print("Usage: python lidar_lenses_wave.py [QUALITY] [FLAGS]")
        print(f"  QUALITY: {list(QUALITY_PRESETS)}  (default: standard)")
        print("  FLAGS:")
        print("    --no-classify       skip the opinionated classifier")
        print("    --save-channels     also dump raw channels as .npz")
        print("    --noise             inject realistic sensor noise")
        print("    --atten=ALPHA       atmospheric attenuation /m (e.g. 0.04)")
        print("    --stack=N           fire N bursts and average (SNR ~√N)")
        print("    --polarization      add polarization-preservation channel")
        print("    --realistic         shorthand for --noise --atten=0.04 --stack=4")
        print("    --help, -h          this message")
        sys.exit(0)

    q = "standard"
    for a in args:
        if a in QUALITY_PRESETS:
            q = a
            break

    classify = "--no-classify" not in args
    save_channels = "--save-channels" in args
    noise = "--noise" in args
    polarization = "--polarization" in args
    atten = 0.0
    stack = 1
    realistic = "--realistic" in args
    if realistic:
        noise = True
        atten = 0.015                   # clear haze (~17% loss at 12m)
        stack = 4
        polarization = True
    for a in args:
        if a.startswith("--atten="):
            atten = float(a.split("=", 1)[1])
        if a.startswith("--stack="):
            stack = int(a.split("=", 1)[1])

    _run_demo(quality=q, classify=classify, save_channels=save_channels,
              sensor_noise=noise, attenuation_per_meter=atten,
              n_stacked_bursts=stack, include_polarization=polarization)
