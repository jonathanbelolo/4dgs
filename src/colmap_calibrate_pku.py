"""Run COLMAP SfM on PKU-DyMVHumans frame 0 to calibrate all cameras.

Uses composited images (com/, white background) for better feature matching.
Outputs calibration.npz with intrinsics, extrinsics, and sparse 3D points.

Usage:
    python colmap_calibrate_pku.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import shutil
import numpy as np
from pathlib import Path

import pycolmap


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

    # ── Collect composited images from all cameras ────────────────────────────
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
        # Use composited images (white bg) for better COLMAP features
        src = cam_dir / "com" / f"{frame_str}.png"
        if not src.exists():
            src = cam_dir / "images" / f"{frame_str}.png"
        if not src.exists():
            print(f"  WARNING: No image for {cam_dir.name} frame {frame_str}")
            continue

        cam_name = cam_dir.name  # e.g. "cam_0"
        dst = images_dir / f"{cam_name}.png"
        shutil.copy2(str(src), str(dst))
        cam_names.append(cam_name)

    print(f"Collected {len(cam_names)} camera images from frame {frame_str}")

    # ── COLMAP: Feature extraction ────────────────────────────────────────────
    db_path = output_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    print("\nExtracting SIFT features...")
    opts = pycolmap.FeatureExtractionOptions()
    opts.max_image_size = 2048  # 1080p is fine at this size
    opts.sift.max_num_features = 16384
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,  # all cameras share one model initially
        camera_model="OPENCV",
        extraction_options=opts,
    )

    # ── COLMAP: Exhaustive matching ───────────────────────────────────────────
    print("Running exhaustive matching...")
    pycolmap.match_exhaustive(database_path=str(db_path))

    # ── COLMAP: Incremental SfM ───────────────────────────────────────────────
    sparse_dir = output_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    print("Running incremental SfM...")
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_num_matches = 15
    mapper_opts.mapper.ba_global_max_num_iterations = 50
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
        options=mapper_opts,
    )

    if not reconstructions:
        print("ERROR: COLMAP reconstruction failed!")
        return

    # Use the largest reconstruction
    recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    print(f"\nReconstruction: {recon.num_reg_images()} images registered, "
          f"{len(recon.points3D)} points")

    if recon.num_reg_images() < len(cam_names):
        print(f"  WARNING: Only {recon.num_reg_images()}/{len(cam_names)} cameras registered!")

    # ── Extract calibration ───────────────────────────────────────────────────
    n_cams = len(cam_names)
    w2c_all = np.zeros((n_cams, 4, 4), dtype=np.float64)
    c2w_all = np.zeros((n_cams, 4, 4), dtype=np.float64)
    K_all = np.zeros((n_cams, 3, 3), dtype=np.float64)
    registered = np.zeros(n_cams, dtype=bool)

    # Build name → cam_idx mapping
    name_to_idx = {name: i for i, name in enumerate(cam_names)}

    # Mean reprojection error
    reproj_errors = []

    for img_id, image in recon.images.items():
        img_name = Path(image.name).stem  # "cam_0"
        if img_name not in name_to_idx:
            continue
        idx = name_to_idx[img_name]

        # World-to-camera: R | t
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        w2c_all[idx] = w2c
        c2w_all[idx] = np.linalg.inv(w2c)

        # Intrinsics
        cam = recon.cameras[image.camera_id]
        params = cam.params
        if cam.model_name == "OPENCV":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif cam.model_name == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif cam.model_name == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam.model_name == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx = fy = params[0]
            cx, cy = cam.width / 2, cam.height / 2

        K_all[idx] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        registered[idx] = True

        # Track reprojection error
        for p2d in image.points2D:
            if p2d.point3D_id >= 0 and p2d.point3D_id in recon.points3D:
                reproj_errors.append(recon.points3D[p2d.point3D_id].error)

    n_registered = registered.sum()
    print(f"\nRegistered {n_registered}/{n_cams} cameras")

    if reproj_errors:
        mean_err = np.mean(reproj_errors)
        print(f"Mean reprojection error: {mean_err:.4f} px")

    # ── Extract sparse 3D points ──────────────────────────────────────────────
    pts = []
    cols = []
    for pid, p in recon.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color / 255.0)

    points3d = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    colors3d = np.array(cols, dtype=np.float32) if cols else np.zeros((0, 3), dtype=np.float32)

    # ── Save calibration ──────────────────────────────────────────────────────
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

    # Print camera positions for sanity check
    positions = c2w_all[registered][:, :3, 3]
    if len(positions) > 0:
        center = positions.mean(axis=0)
        radius = np.linalg.norm(positions - center, axis=1).mean()
        print(f"  Camera ring center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"  Camera ring radius: {radius:.2f}")


if __name__ == "__main__":
    main()
