"""Run COLMAP SfM on PKU-DyMVHumans frame 0 to calibrate all cameras.

Uses hloc (SuperPoint + LightGlue) for robust wide-baseline feature matching
in the dark studio environment, then runs COLMAP incremental SfM.

Outputs calibration.npz with intrinsics, extrinsics, and sparse 3D points.

Usage:
    python colmap_calibrate_pku.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import shutil
import numpy as np
from pathlib import Path

import pycolmap


def run_hloc_sfm(images_dir, output_dir):
    """Run hloc SuperPoint + LightGlue matching → COLMAP SfM.

    Args:
        images_dir: Path to directory of images
        output_dir: Path to output directory

    Returns:
        pycolmap.Reconstruction or None
    """
    from hloc import (
        extract_features,
        match_features,
        reconstruction,
    )

    # Paths for hloc outputs
    sfm_dir = output_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"

    # ── Generate exhaustive pairs ─────────────────────────────────────────
    image_names = sorted([f.name for f in images_dir.glob("*.png")])
    print(f"Generating exhaustive pairs for {len(image_names)} images...")
    with open(pairs_path, "w") as f:
        for i, name_i in enumerate(image_names):
            for name_j in image_names[i + 1:]:
                f.write(f"{name_i} {name_j}\n")

    n_pairs = len(image_names) * (len(image_names) - 1) // 2
    print(f"  {n_pairs} pairs")

    # ── SuperPoint feature extraction ─────────────────────────────────────
    print("\nExtracting SuperPoint features...")
    feature_conf = extract_features.confs["superpoint_max"]
    feature_conf["model"]["max_keypoints"] = 4096
    extract_features.main(
        conf=feature_conf,
        image_dir=images_dir,
        export_dir=output_dir,
        feature_path=features_path,
    )

    # ── LightGlue matching ────────────────────────────────────────────────
    print("Running LightGlue matching...")
    match_conf = match_features.confs["superglue"]
    match_features.main(
        conf=match_conf,
        pairs=pairs_path,
        features=features_path,
        export_dir=output_dir,
        matches=matches_path,
    )

    # ── COLMAP reconstruction from hloc features ──────────────────────────
    print("Running COLMAP incremental SfM from hloc features...")
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=images_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="SIMPLE_RADIAL",
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Which frame to use for calibration")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scene_dir / "colmap_calib"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_str = f"{args.frame_idx:06d}"

    # ── Collect images from all cameras ────────────────────────────────────
    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    per_view = scene_dir / "per_view"
    cam_dirs = sorted(
        [d for d in per_view.iterdir() if d.is_dir() and d.name.startswith("cam_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    cam_names = []

    for cam_dir in cam_dirs:
        src = cam_dir / "images" / f"{frame_str}.png"
        if not src.exists():
            print(f"  WARNING: No image for {cam_dir.name} frame {frame_str}")
            continue

        cam_name = cam_dir.name
        dst = images_dir / f"{cam_name}.png"
        shutil.copy2(str(src), str(dst))
        cam_names.append(cam_name)

    print(f"Collected {len(cam_names)} camera images from frame {frame_str}")

    # ── Run hloc SfM ──────────────────────────────────────────────────────
    recon = run_hloc_sfm(images_dir, output_dir)

    if recon is None:
        print("ERROR: Reconstruction failed!")
        return

    print(f"\nReconstruction: {recon.num_reg_images()} images registered, "
          f"{len(recon.points3D)} points")

    if recon.num_reg_images() < len(cam_names):
        print(f"  WARNING: Only {recon.num_reg_images()}/{len(cam_names)} cameras registered!")

    # ── Extract calibration ───────────────────────────────────────────────
    n_cams = len(cam_names)
    w2c_all = np.zeros((n_cams, 4, 4), dtype=np.float64)
    c2w_all = np.zeros((n_cams, 4, 4), dtype=np.float64)
    K_all = np.zeros((n_cams, 3, 3), dtype=np.float64)
    registered = np.zeros(n_cams, dtype=bool)

    name_to_idx = {name: i for i, name in enumerate(cam_names)}

    reproj_errors = []

    for img_id, image in recon.images.items():
        img_name = Path(image.name).stem
        if img_name not in name_to_idx:
            continue
        idx = name_to_idx[img_name]

        # World-to-camera
        cfw = image.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        w2c_all[idx] = w2c
        c2w_all[idx] = np.linalg.inv(w2c)

        # Intrinsics
        cam = recon.cameras[image.camera_id]
        params = cam.params
        model_name = str(cam.model).split(".")[-1]
        if model_name in ("OPENCV", "PINHOLE"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx = fy = params[0]
            cx, cy = cam.width / 2, cam.height / 2

        K_all[idx] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        registered[idx] = True

        for p2d in image.points2D:
            if p2d.has_point3D() and p2d.point3D_id in recon.points3D:
                reproj_errors.append(recon.points3D[p2d.point3D_id].error)

    n_registered = registered.sum()
    print(f"\nRegistered {n_registered}/{n_cams} cameras")

    if reproj_errors:
        mean_err = np.mean(reproj_errors)
        print(f"Mean reprojection error: {mean_err:.4f} px")

    # ── Extract sparse 3D points ──────────────────────────────────────────
    pts = []
    cols = []
    for pid, p in recon.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color / 255.0)

    points3d = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    colors3d = np.array(cols, dtype=np.float32) if cols else np.zeros((0, 3), dtype=np.float32)

    # ── Save calibration ──────────────────────────────────────────────────
    calib_path = scene_dir / "calibration.npz"
    np.savez(
        str(calib_path),
        w2c=w2c_all,
        c2w=c2w_all,
        K=K_all,
        cam_names=np.array(cam_names),
        registered=registered,
        points3d=points3d,
        colors3d=colors3d,
    )
    print(f"\nSaved calibration to {calib_path}")
    print(f"  {n_registered} cameras, {len(points3d)} sparse points")

    positions = c2w_all[registered][:, :3, 3]
    if len(positions) > 0:
        center = positions.mean(axis=0)
        radius = np.linalg.norm(positions - center, axis=1).mean()
        print(f"  Camera ring center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"  Camera ring radius: {radius:.2f}")


if __name__ == "__main__":
    main()
