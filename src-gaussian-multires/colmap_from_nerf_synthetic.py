"""Run COLMAP triangulation on NeRF Synthetic scenes using known camera poses.

Produces points3d.npy and colors3d.npy for Gaussian initialization.
Since cameras are already known, we only need:
1. Feature extraction (SIFT)
2. Feature matching
3. Triangulation with fixed cameras

Usage:
    python colmap_from_nerf_synthetic.py --data_dir data/nerf_synthetic --scene lego
"""
import argparse
import json
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def rotmat_to_qvec(R):
    """Convert 3x3 rotation to COLMAP quaternion (w, x, y, z)."""
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # scipy: (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def create_colmap_cameras_and_images(scene_dir, colmap_dir, resolution=800):
    """Write COLMAP cameras.txt and images.txt from NeRF Synthetic transforms."""
    with open(scene_dir / "transforms_train.json") as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    focal = 0.5 * resolution / np.tan(0.5 * camera_angle_x)

    # Single shared camera model
    cameras_txt = colmap_dir / "cameras.txt"
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {resolution} {resolution} {focal} {focal} "
                f"{resolution/2} {resolution/2}\n")

    # Per-image poses: NeRF Synthetic is OpenGL, COLMAP expects w2c
    images_txt = colmap_dir / "images.txt"
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, frame in enumerate(meta["frames"]):
            c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)

            # OpenGL -> OpenCV: negate y and z
            c2w_cv = c2w_gl.copy()
            c2w_cv[:3, 1] *= -1
            c2w_cv[:3, 2] *= -1

            # c2w -> w2c for COLMAP
            w2c = np.linalg.inv(c2w_cv)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            qvec = rotmat_to_qvec(R)
            fname = Path(frame["file_path"]).name + ".png"

            image_id = idx + 1
            f.write(f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                    f"{t[0]} {t[1]} {t[2]} 1 {fname}\n")
            f.write("\n")  # empty points2d line

    # Empty points3D.txt (will be filled by triangulation)
    with open(colmap_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list (empty, to be triangulated)\n")

    return len(meta["frames"])


def prepare_images(scene_dir, image_dir, resolution=800):
    """Copy and optionally resize training images for COLMAP."""
    with open(scene_dir / "transforms_train.json") as f:
        meta = json.load(f)

    image_dir.mkdir(parents=True, exist_ok=True)

    for frame in meta["frames"]:
        src = scene_dir / f"{frame['file_path']}.png"
        fname = Path(frame["file_path"]).name + ".png"
        dst = image_dir / fname

        img = Image.open(src)
        if img.size[0] != resolution:
            img = img.resize((resolution, resolution), Image.LANCZOS)

        # Convert RGBA to RGB (white background)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        img.save(dst)


def run_colmap_triangulation(work_dir, image_dir):
    """Run COLMAP feature extraction, matching, and triangulation."""
    db_path = work_dir / "database.db"
    model_input = work_dir / "model_input"
    model_output = work_dir / "model_output"
    model_output.mkdir(parents=True, exist_ok=True)

    print("  COLMAP: Feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ], check=True, capture_output=True)

    print("  COLMAP: Feature matching (GPU)...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], check=True, capture_output=True)

    # Now we need to overwrite the camera params and image poses in the DB
    # to match our known cameras, then triangulate
    print("  COLMAP: Fixing cameras in database...")
    fix_database_cameras(db_path, model_input)

    print("  COLMAP: Point triangulation...")
    subprocess.run([
        "colmap", "point_triangulator",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--input_path", str(model_input),
        "--output_path", str(model_output),
        "--Mapper.tri_min_angle", "1.0",
    ], check=True, capture_output=True)

    return model_output


def fix_database_cameras(db_path, model_input):
    """Update COLMAP database camera IDs to match our known model."""
    # Read the known camera from cameras.txt
    cameras_txt = model_input / "cameras.txt"
    images_txt = model_input / "images.txt"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Read known image names and their intended camera_id=1
    # Update database: set all images to camera_id=1
    cur.execute("UPDATE images SET camera_id = 1")

    # Also ensure the camera params in DB match our known camera
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]  # PINHOLE
            w, h = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]

            # COLMAP PINHOLE: fx, fy, cx, cy
            model_id = 1  # PINHOLE
            params_blob = struct.pack(f"{len(params)}d", *params)
            cur.execute(
                "UPDATE cameras SET model = ?, width = ?, height = ?, params = ? "
                "WHERE camera_id = ?",
                (model_id, w, h, params_blob, cam_id)
            )

    conn.commit()
    conn.close()


def read_colmap_points3d(model_dir):
    """Read COLMAP points3D.bin and return points + colors."""
    points_path = model_dir / "points3D.bin"
    if not points_path.exists():
        # Try text format
        return read_colmap_points3d_txt(model_dir / "points3D.txt")

    points = []
    colors = []
    with open(points_path, "rb") as f:
        n_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_points):
            point3d_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            # Skip track entries
            f.read(track_length * 8)  # pairs of (image_id, point2d_idx)
            points.append(xyz)
            colors.append([c / 255.0 for c in rgb])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def read_colmap_points3d_txt(path):
    """Read COLMAP points3D.txt format."""
    points = []
    colors = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            rgb = [int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0]
            points.append(xyz)
            colors.append(rgb)
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--resolution", type=int, default=800,
                        help="Image resolution for feature extraction")
    args = parser.parse_args()

    scene_dir = Path(args.data_dir) / args.scene
    work_dir = scene_dir / "colmap_workspace"
    image_dir = work_dir / "images"
    model_input = work_dir / "model_input"

    # Clean previous run
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    model_input.mkdir(parents=True)

    print(f"Running COLMAP triangulation for {args.scene} at {args.resolution}px")

    # 1. Prepare images
    print("Preparing images...")
    prepare_images(scene_dir, image_dir, args.resolution)

    # 2. Create known camera model
    print("Creating camera model from transforms.json...")
    n_images = create_colmap_cameras_and_images(
        scene_dir, model_input, args.resolution
    )
    print(f"  {n_images} images, known camera poses")

    # 3. Run COLMAP
    model_output = run_colmap_triangulation(work_dir, image_dir)

    # 4. Extract points
    points, colors = read_colmap_points3d(model_output)
    print(f"\nTriangulated {len(points):,} 3D points")

    if len(points) > 0:
        # Transform points from COLMAP (OpenCV) back to our convention
        # (already in OpenCV since we converted poses to OpenCV for COLMAP)
        out_dir = scene_dir / "colmap"
        out_dir.mkdir(exist_ok=True)
        np.save(str(out_dir / "points3d.npy"), points)
        np.save(str(out_dir / "colors3d.npy"), colors)
        print(f"Saved to {out_dir}/points3d.npy ({len(points):,} points)")
        print(f"  Bounds: [{points.min(0)} .. {points.max(0)}]")
    else:
        print("ERROR: No points triangulated!")


if __name__ == "__main__":
    main()
