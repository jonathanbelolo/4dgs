"""Establish canonical reference frame from multi-view calibration + masks.

Triangulates the person's 3D center from alpha mask centroids across all cameras,
estimates person extent, and defines a fixed canonical coordinate frame with
per-frame centroid tracking.

Usage:
    python canonical_bbox.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def triangulate_point_dlt(projections, points_2d):
    """Triangulate a 3D point from N camera projections using DLT.

    Args:
        projections: (N, 3, 4) projection matrices P = K @ w2c[:3]
        points_2d: (N, 2) 2D observations [x, y]

    Returns:
        (3,) 3D point in world space
    """
    N = len(projections)
    A = np.zeros((2 * N, 4))
    for i in range(N):
        P = projections[i]
        x, y = points_2d[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def mask_centroid(mask_path):
    """Compute centroid of foreground region in a mask image.

    Returns (cx, cy) or None if mask is empty.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = mask > 127
    if fg.sum() == 0:
        return None
    ys, xs = np.where(fg)
    return np.array([xs.mean(), ys.mean()])


def mask_bbox(mask_path):
    """Get bounding box [x_min, y_min, x_max, y_max] of foreground.

    Returns bbox or None if mask is empty.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = mask > 127
    if fg.sum() == 0:
        return None
    ys, xs = np.where(fg)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def estimate_up_direction(c2w_all, registered):
    """Estimate up direction from camera arrangement.

    For a ring of cameras, the up direction is approximately the direction
    perpendicular to the plane containing all camera centers.
    """
    positions = c2w_all[registered][:, :3, 3]
    center = positions.mean(axis=0)
    centered = positions - center

    # PCA: smallest eigenvector of the camera positions is normal to the ring plane
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # smallest singular value direction

    # Make sure it points "up" (positive y in most conventions)
    # Use camera up vectors as reference
    cam_ups = c2w_all[registered][:, :3, 1].mean(axis=0)
    if np.dot(normal, cam_ups) < 0:
        normal = -normal

    return normal / np.linalg.norm(normal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--padding", type=float, default=0.2,
                        help="Padding factor beyond estimated person extent")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    # ── Load calibration ──────────────────────────────────────────────────────
    calib = np.load(str(scene_dir / "calibration.npz"), allow_pickle=True)
    w2c = calib["w2c"]
    c2w = calib["c2w"]
    K = calib["K"]
    registered = calib["registered"]
    cam_names = calib["cam_names"]

    n_cams = len(cam_names)
    reg_indices = np.where(registered)[0]
    print(f"Using {len(reg_indices)}/{n_cams} registered cameras")

    # ── Compute projection matrices for registered cameras ────────────────────
    P = np.zeros((n_cams, 3, 4))
    for i in range(n_cams):
        P[i] = K[i] @ w2c[i][:3]

    # ── Triangulate person center from frame 0 mask centroids ─────────────────
    per_view = scene_dir / "per_view"
    centroids_2d = []
    centroid_cams = []

    for idx in reg_indices:
        cam_name = cam_names[idx]
        mask_path = per_view / cam_name / "pha" / "000000.png"
        if not mask_path.exists():
            continue
        c = mask_centroid(mask_path)
        if c is not None:
            centroids_2d.append(c)
            centroid_cams.append(idx)

    centroids_2d = np.array(centroids_2d)
    centroid_P = P[centroid_cams]

    person_center = triangulate_point_dlt(centroid_P, centroids_2d)
    print(f"Person center (frame 0): [{person_center[0]:.3f}, "
          f"{person_center[1]:.3f}, {person_center[2]:.3f}]")

    # ── Estimate person height from mask bounding boxes ───────────────────────
    heights_3d = []
    for idx in reg_indices:
        cam_name = cam_names[idx]
        mask_path = per_view / cam_name / "pha" / "000000.png"
        if not mask_path.exists():
            continue
        bbox = mask_bbox(mask_path)
        if bbox is None:
            continue

        # Back-project top and bottom of bounding box through camera
        # Use person center's depth to estimate 3D extent
        cam_w2c = w2c[idx]
        cam_K = K[idx]

        # Person center in camera space
        pc_cam = cam_w2c[:3, :3] @ person_center + cam_w2c[:3, 3]
        depth = pc_cam[2]
        if depth <= 0:
            continue

        # Top and bottom pixel y → 3D height
        y_top = bbox[1]
        y_bot = bbox[3]
        fy = cam_K[1, 1]

        height_pixels = y_bot - y_top
        height_3d = height_pixels * depth / fy
        heights_3d.append(height_3d)

    median_height = np.median(heights_3d) if heights_3d else 1.7
    print(f"Estimated person height: {median_height:.3f} (from {len(heights_3d)} cameras)")

    # ── Define canonical frame ────────────────────────────────────────────────
    up = estimate_up_direction(c2w, registered)

    # Forward: arbitrary horizontal direction (perpendicular to up)
    # Use direction from camera ring center to first camera as initial
    cam_positions = c2w[registered][:, :3, 3]
    cam_center = cam_positions.mean(axis=0)
    forward_init = person_center - cam_center
    forward_init = forward_init - np.dot(forward_init, up) * up
    if np.linalg.norm(forward_init) < 1e-6:
        # Person is at ring center — use direction to first camera instead
        forward_init = cam_positions[0] - cam_center
        forward_init = forward_init - np.dot(forward_init, up) * up
    forward = forward_init / np.linalg.norm(forward_init)

    # Right = up × forward
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Recompute forward for orthogonality
    forward = np.cross(right, up)
    forward = forward / np.linalg.norm(forward)

    axes = np.stack([right, up, forward])  # (3, 3): rows are basis vectors

    # Bounding box half-extents
    width = median_height * 0.4   # rough human proportions
    depth_ext = median_height * 0.3
    half_extents = np.array([
        width / 2 * (1 + args.padding),
        median_height / 2 * (1 + args.padding),
        depth_ext / 2 * (1 + args.padding),
    ])

    print(f"Canonical frame:")
    print(f"  Origin: {person_center}")
    print(f"  Up:     {up}")
    print(f"  Right:  {right}")
    print(f"  Forward:{forward}")
    print(f"  Bbox half-extents: {half_extents}")

    # ── Per-frame centroid tracking ───────────────────────────────────────────
    # Count available frames
    sample_cam = cam_names[reg_indices[0]]
    frame_files = sorted((per_view / sample_cam / "pha").glob("*.png"))
    n_frames = len(frame_files)
    print(f"\nTracking person centroid across {n_frames} frames...")

    per_frame_translation = np.zeros((n_frames, 3))

    for f_idx in range(n_frames):
        frame_str = f"{f_idx:06d}"
        centroids_2d_f = []
        centroid_cams_f = []

        for idx in reg_indices:
            cam_name = cam_names[idx]
            mask_path = per_view / cam_name / "pha" / f"{frame_str}.png"
            if not mask_path.exists():
                continue
            c = mask_centroid(mask_path)
            if c is not None:
                centroids_2d_f.append(c)
                centroid_cams_f.append(idx)

        if len(centroids_2d_f) >= 2:
            center_f = triangulate_point_dlt(
                P[centroid_cams_f], np.array(centroids_2d_f)
            )
            per_frame_translation[f_idx] = center_f - person_center
        # else: keep zero offset

        if f_idx % 50 == 0:
            offset = per_frame_translation[f_idx]
            print(f"  Frame {f_idx}: offset [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")

    # Smooth per-frame translations (simple Gaussian smoothing)
    from scipy.ndimage import gaussian_filter1d
    for dim in range(3):
        per_frame_translation[:, dim] = gaussian_filter1d(
            per_frame_translation[:, dim], sigma=2.0
        )

    # ── Save canonical frame ──────────────────────────────────────────────────
    out_path = scene_dir / "canonical_frame.npz"
    np.savez(
        str(out_path),
        origin=person_center,
        axes=axes,
        bbox_half_extents=half_extents,
        per_frame_translation=per_frame_translation,
        up=up,
        person_height=median_height,
    )
    print(f"\nSaved canonical frame to {out_path}")


if __name__ == "__main__":
    main()
