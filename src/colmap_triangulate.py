"""Run COLMAP point triangulation using known camera poses from poses_bounds.npy.

Uses pycolmap for headless SIFT extraction/matching, then fixes up the
database and reconstruction to use known poses before triangulating.
"""
import argparse
import shutil
import sqlite3
import struct
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pycolmap


def rotmat_to_qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # scipy returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def blob_to_array(blob, dtype, shape):
    return np.frombuffer(blob, dtype=dtype).reshape(shape)


def array_to_blob(arr):
    return arr.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scene_dir / "colmap"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load poses ────────────────────────────────────────────────────────────
    data = np.load(str(scene_dir / "poses_bounds.npy"))
    poses = data[:, :15].reshape(-1, 3, 5)

    mp4s = sorted(scene_dir.glob("cam*.mp4"))
    cam_names = [f.stem for f in mp4s]

    # ── Copy training images ──────────────────────────────────────────────────
    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    frames_dir = scene_dir / "frames" / "train"
    image_files = sorted(frames_dir.glob("cam*.jpg"))
    for img in image_files:
        shutil.copy2(str(img), str(images_dir / img.name))
    print(f"Copied {len(image_files)} training images")

    img_cam_pairs = []
    for img in image_files:
        cam_name = img.stem.split("_")[0] if "_" in img.stem else img.stem
        idx = cam_names.index(cam_name)
        img_cam_pairs.append((img.name, idx))
    img_cam_pairs.sort()

    from PIL import Image as PILImage
    sample_img = PILImage.open(image_files[0])
    W, H = sample_img.size
    print(f"Image dimensions: {W}x{H}")

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    db_path = output_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    print("\n── Extracting SIFT features...")
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.max_image_size = 4096
    extraction_opts.sift.max_num_features = 16384
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
        extraction_options=extraction_opts,
    )
    print("Feature extraction done.")

    # ── Step 2: Matching ──────────────────────────────────────────────────────
    print("\n── Running exhaustive matching...")
    pycolmap.match_exhaustive(database_path=str(db_path))
    print("Matching done.")

    # ── Step 3: Fix database cameras to use known intrinsics/poses ────────────
    print("\n── Updating database with known camera parameters...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get image names and IDs from database
    cursor.execute("SELECT image_id, name, camera_id FROM images")
    db_images = {row[1]: (row[0], row[2]) for row in cursor.fetchall()}

    # Update each camera to use our known intrinsics
    for img_name, cam_idx in img_cam_pairs:
        if img_name not in db_images:
            print(f"  WARNING: {img_name} not found in database!")
            continue

        db_image_id, db_camera_id = db_images[img_name]
        focal = poses[cam_idx, :, 4][2] / args.downsample

        # Update camera to PINHOLE with correct focal
        # PINHOLE model_id = 1, params = [fx, fy, cx, cy]
        params = np.array([focal, focal, W / 2.0, H / 2.0], dtype=np.float64)
        cursor.execute(
            "UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (W, H, array_to_blob(params), db_camera_id),
        )

    conn.commit()
    conn.close()
    print("Database updated.")

    # ── Step 4: Build reconstruction from database with known poses ───────────
    sparse_dir = output_dir / "sparse_known"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    # Read the database IDs
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    db_cameras = {}
    for row in cursor.fetchall():
        db_cameras[row[0]] = row

    cursor.execute("SELECT image_id, name, camera_id FROM images")
    db_imgs = {}
    for row in cursor.fetchall():
        db_imgs[row[1]] = (row[0], row[2])
    conn.close()

    # Write cameras.txt using database camera IDs
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list\n")
        for cam_id, (_, model, w, h, params_blob) in db_cameras.items():
            params = np.frombuffer(params_blob, dtype=np.float64)
            params_str = " ".join(f"{p:.6f}" for p in params)
            model_name = "PINHOLE"
            f.write(f"{cam_id} {model_name} {w} {h} {params_str}\n")

    # Write images.txt using database image IDs, with our known poses
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list\n")
        for img_name, cam_idx in img_cam_pairs:
            if img_name not in db_imgs:
                continue
            db_img_id, db_cam_id = db_imgs[img_name]

            R_llff = poses[cam_idx, :, :3]
            t_llff = poses[cam_idx, :, 3]

            # LLFF DRB → COLMAP RDF (c2w), then invert to w2c
            R_c2w = np.column_stack([R_llff[:, 1], R_llff[:, 0], -R_llff[:, 2]])
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_llff

            qvec = rotmat_to_qvec(R_w2c)
            f.write(f"{db_img_id} {qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} {db_cam_id} {img_name}\n")
            f.write("\n")

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# Empty\n")

    print(f"Wrote reconstruction to {sparse_dir}")

    # ── Step 5: Triangulate ───────────────────────────────────────────────────
    sparse_out = output_dir / "sparse_triangulated"
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True)

    print("\n── Triangulating points...")
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    result = pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_out),
    )
    print("Triangulation done.")

    # ── Step 6: Extract points ────────────────────────────────────────────────
    print("\n── Reading triangulated points...")
    final = pycolmap.Reconstruction(sparse_out)

    pts = []
    cols = []
    for pid, p in final.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color / 255.0)

    if len(pts) == 0:
        print("ERROR: No points triangulated!")
        return

    points = np.array(pts, dtype=np.float32)
    colors = np.array(cols, dtype=np.float32)

    np.save(str(output_dir / "points3d.npy"), points)
    np.save(str(output_dir / "colors3d.npy"), colors)

    center = points.mean(axis=0)
    extent = np.linalg.norm(points - center, axis=1).max()
    print(f"\nTriangulated {len(points):,} points!")
    print(f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"Extent: {extent:.2f}")
    print(f"Saved to {output_dir / 'points3d.npy'}")


if __name__ == "__main__":
    main()
