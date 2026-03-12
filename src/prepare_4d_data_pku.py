"""Prepare training initialization data from SMPL-X fits.

Generates smplx_init.npz with initial Gaussian positions, colors, semantic
labels, and LBS skinning weights sampled from the SMPL-X mesh surface.
Also includes filtered COLMAP sparse points inside the canonical bounding box.

Usage:
    python prepare_4d_data_pku.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def sample_mesh_surface(vertices, faces, n_samples, rng=None):
    """Uniformly sample points on a triangle mesh surface.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        n_samples: number of points to sample
        rng: numpy random generator

    Returns:
        points: (n_samples, 3) sampled positions
        face_indices: (n_samples,) which face each point came from
        bary_coords: (n_samples, 3) barycentric coordinates
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()

    # Sample faces proportional to area
    face_indices = rng.choice(len(faces), size=n_samples, p=probs)

    # Random barycentric coordinates
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=1)

    # Interpolate positions
    f = faces[face_indices]
    points = (bary[:, 0:1] * vertices[f[:, 0]] +
              bary[:, 1:2] * vertices[f[:, 1]] +
              bary[:, 2:3] * vertices[f[:, 2]])

    return points, face_indices, bary


def interpolate_vertex_attributes(attr, faces, face_indices, bary):
    """Interpolate per-vertex attributes using barycentric coordinates.

    Args:
        attr: (V, ...) per-vertex attribute
        faces: (F, 3) face indices
        face_indices: (N,) sampled face indices
        bary: (N, 3) barycentric coordinates

    Returns:
        (N, ...) interpolated attributes
    """
    f = faces[face_indices]
    a0 = attr[f[:, 0]]
    a1 = attr[f[:, 1]]
    a2 = attr[f[:, 2]]

    if attr.ndim == 1:
        return bary[:, 0] * a0 + bary[:, 1] * a1 + bary[:, 2] * a2
    else:
        return (bary[:, 0:1] * a0 + bary[:, 1:2] * a1 + bary[:, 2:3] * a2)


def sample_colors_from_images(points_3d, calib, scene_dir, frame_idx=0, n_cams=8):
    """Sample RGB colors for 3D points by projecting into camera images.

    Uses median color across multiple views for robustness.

    Args:
        points_3d: (N, 3) world-space points
        calib: calibration dict
        scene_dir: Path to scene
        frame_idx: which frame to sample from
        n_cams: number of cameras to use

    Returns:
        (N, 3) float32 colors in [0, 1]
    """
    w2c = calib["w2c"]
    K_all = calib["K"]
    registered = calib["registered"]
    cam_names = calib["cam_names"]
    reg_indices = np.where(registered)[0]

    per_view = scene_dir / "per_view"
    frame_str = f"{frame_idx:06d}"

    N = len(points_3d)

    # Select evenly spaced cameras
    step = max(1, len(reg_indices) // n_cams)
    selected = reg_indices[::step][:n_cams]

    all_colors = []  # list of (N, 3) arrays

    for cam_idx in selected:
        cam_name = cam_names[cam_idx]
        img_path = per_view / cam_name / "images" / f"{frame_str}.png"
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W = img.shape[:2]

        # Project points
        R = w2c[cam_idx][:3, :3]
        t = w2c[cam_idx][:3, 3]
        pts_cam = (R @ points_3d.T + t[:, None]).T
        K = K_all[cam_idx]
        pts_2d = (K @ pts_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3].clip(min=1e-6)

        # Check visibility
        visible = pts_cam[:, 2] > 0.01
        px = pts_2d[:, 0].astype(np.int32)
        py = pts_2d[:, 1].astype(np.int32)
        in_bounds = visible & (px >= 0) & (px < W) & (py >= 0) & (py < H)

        colors = np.full((N, 3), np.nan, dtype=np.float32)
        valid = np.where(in_bounds)[0]
        if len(valid) > 0:
            colors[valid] = img[py[valid], px[valid]]
        all_colors.append(colors)

    if not all_colors:
        return np.full((N, 3), 0.5, dtype=np.float32)

    # Median across views (ignoring NaN)
    stacked = np.stack(all_colors, axis=0)  # (n_views, N, 3)
    with np.errstate(all="ignore"):
        median_colors = np.nanmedian(stacked, axis=0)

    # Fill remaining NaN with gray
    nan_mask = np.isnan(median_colors).any(axis=1)
    median_colors[nan_mask] = 0.5

    return median_colors.astype(np.float32)


def filter_points_in_bbox(points, origin, axes, half_extents):
    """Filter points to those inside the canonical bounding box.

    Args:
        points: (N, 3) world-space points
        origin: (3,) bbox center
        axes: (3, 3) bbox axes
        half_extents: (3,) half-widths

    Returns:
        mask: (N,) bool, True for points inside bbox
    """
    # Transform to canonical local coordinates
    local = (points - origin) @ axes.T  # (N, 3) in [right, up, forward] space
    inside = np.all(np.abs(local) <= half_extents, axis=1)
    return inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--n_surface_samples", type=int, default=30000,
                        help="Number of points to sample on SMPL-X surface")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Which frame's SMPL-X to use for initialization")
    parser.add_argument("--include_sfm", action="store_true",
                        help="Include filtered COLMAP sparse points")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    # ── Load SMPL-X params for reference frame ────────────────────────────────
    smplx_path = scene_dir / "smplx_params" / f"frame_{args.frame_idx:06d}.npz"
    if not smplx_path.exists():
        raise FileNotFoundError(
            f"SMPL-X params not found: {smplx_path}\n"
            "Run fit_smplx_canonical.py first."
        )

    smplx_data = np.load(str(smplx_path), allow_pickle=True)
    vertices = smplx_data["vertices"]          # (10475, 3)
    skinning_weights = smplx_data["skinning_weights"]  # (10475, 55)
    semantic_labels = smplx_data["semantic_labels"]    # (10475,)

    print(f"SMPL-X mesh: {vertices.shape[0]} vertices")

    # ── Load calibration and canonical frame ──────────────────────────────────
    calib = dict(np.load(str(scene_dir / "calibration.npz"), allow_pickle=True))
    canonical = dict(np.load(str(scene_dir / "canonical_frame.npz"), allow_pickle=True))

    # ── Get SMPL-X faces (need the model for this) ────────────────────────────
    # Try loading from smplx package, fall back to generating from vertices
    faces = None
    try:
        import smplx as smplx_pkg
        candidates = [
            Path("/workspace/body_models/smplx"),
            Path("/workspace/smplx_models"),
            Path("body_models/smplx"),
            Path.home() / ".smplx" / "models",
        ]
        for p in candidates:
            if p.exists():
                model = smplx_pkg.create(str(p.parent), model_type="smplx",
                                          gender="neutral", batch_size=1)
                faces = model.faces.astype(np.int64)
                break
    except (ImportError, Exception):
        pass

    if faces is None:
        # Fallback: use vertices directly (no surface sampling, just use vertices)
        print("WARNING: SMPL-X model not available, using vertices directly")
        init_points = vertices
        init_weights = skinning_weights
        init_semantics = semantic_labels
    else:
        # ── Sample points on mesh surface ─────────────────────────────────────
        print(f"Sampling {args.n_surface_samples} points on SMPL-X surface...")
        rng = np.random.default_rng(42)

        surface_pts, face_idx, bary = sample_mesh_surface(
            vertices, faces, args.n_surface_samples, rng
        )

        # Also include all mesh vertices
        init_points = np.concatenate([vertices, surface_pts], axis=0)

        # Interpolate skinning weights for surface samples
        surface_weights = interpolate_vertex_attributes(
            skinning_weights, faces, face_idx, bary
        )
        init_weights = np.concatenate([skinning_weights, surface_weights], axis=0)

        # Interpolate semantic labels (nearest vertex for discrete labels)
        f = faces[face_idx]
        nearest_vert = f[np.arange(len(face_idx)), bary.argmax(axis=1)]
        surface_semantics = semantic_labels[nearest_vert]
        init_semantics = np.concatenate([semantic_labels, surface_semantics], axis=0)

    # ── Sample colors from images ─────────────────────────────────────────────
    print("Sampling colors from multi-view images...")
    init_colors = sample_colors_from_images(
        init_points, calib, scene_dir, args.frame_idx
    )

    n_total = len(init_points)
    print(f"SMPL-X init points: {n_total}")

    # ── Optionally add COLMAP sparse points inside canonical bbox ──────────────
    if args.include_sfm and "points3d" in calib:
        sfm_pts = calib["points3d"]
        sfm_cols = calib.get("colors3d", np.full_like(sfm_pts, 0.5))

        if len(sfm_pts) > 0:
            origin = canonical["origin"]
            axes = canonical["axes"]
            half_ext = canonical["bbox_half_extents"] * 1.5  # slightly expanded
            inside = filter_points_in_bbox(sfm_pts, origin, axes, half_ext)

            sfm_inside = sfm_pts[inside]
            sfm_cols_inside = sfm_cols[inside]

            if len(sfm_inside) > 0:
                print(f"Adding {len(sfm_inside)} COLMAP points inside bbox "
                      f"(from {len(sfm_pts)} total)")

                # SfM points get zero skinning weights (free-floating) and bg label
                sfm_weights = np.zeros((len(sfm_inside), skinning_weights.shape[1]),
                                       dtype=np.float32)
                sfm_semantics = np.zeros(len(sfm_inside), dtype=np.uint8)

                init_points = np.concatenate([init_points, sfm_inside], axis=0)
                init_colors = np.concatenate([init_colors, sfm_cols_inside], axis=0)
                init_weights = np.concatenate([init_weights, sfm_weights], axis=0)
                init_semantics = np.concatenate([init_semantics, sfm_semantics], axis=0)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_final = len(init_points)
    unique, counts = np.unique(init_semantics[init_semantics > 0], return_counts=True)
    sem_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    print(f"\nFinal init: {n_final} points")
    print(f"  Semantic distribution: {sem_dist}")
    print(f"  Skinning weights shape: {init_weights.shape}")
    print(f"  Bounding box: [{init_points.min(0)}, {init_points.max(0)}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = scene_dir / "smplx_init.npz"
    np.savez(
        str(out_path),
        positions=init_points.astype(np.float32),
        colors=init_colors.astype(np.float32),
        skinning_weights=init_weights.astype(np.float32),
        semantic_labels=init_semantics.astype(np.uint8),
        # Also save reference frame SMPL-X params for training
        smplx_vertices=vertices.astype(np.float32),
        smplx_betas=smplx_data["betas"].astype(np.float32),
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
