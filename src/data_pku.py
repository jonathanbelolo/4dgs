"""Data loading for PKU-DyMVHumans dataset.

Supports the Part 1 format:
  scene_dir/
  ├── cams/           # MVSNet cam_txt files (static camera rig)
  ├── per_frame/      # per-timestep directories
  │   ├── 000000/images/*.png
  │   ├── 000005/images/*.png
  │   └── ...
  └── data_COLMAP/    # per-frame COLMAP (for SfM points)

Camera format: cam_txt with 4x4 world-to-camera extrinsic, 3x3 intrinsic,
and 4 depth parameters.
"""
import struct
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def read_cam_txt(path):
    """Read an MVSNet-format camera txt file.

    Returns:
        extrinsic: (4, 4) world-to-camera matrix
        intrinsic: (3, 3) camera intrinsic matrix
        depth_params: (4,) depth range parameters [min, interval, samples, max]
    """
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Find extrinsic block
    ext_start = None
    int_start = None
    for i, line in enumerate(lines):
        if line.lower() == 'extrinsic':
            ext_start = i + 1
        elif line.lower() == 'intrinsic':
            int_start = i + 1

    extrinsic = np.array([
        [float(x) for x in lines[ext_start + j].split()]
        for j in range(4)
    ], dtype=np.float64)

    intrinsic = np.array([
        [float(x) for x in lines[int_start + j].split()]
        for j in range(3)
    ], dtype=np.float64)

    # Depth params on the last line
    depth_params = np.array([float(x) for x in lines[-1].split()], dtype=np.float32)

    return extrinsic, intrinsic, depth_params


def read_colmap_points3d_bin(path):
    """Read COLMAP points3D.bin file.

    Returns:
        points: (N, 3) float32
        colors: (N, 3) float32 in [0, 1]
    """
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        points = []
        colors = []
        for _ in range(num_points):
            point_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<ddd', f.read(24))
            rgb = struct.unpack('<BBB', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_len = struct.unpack('<Q', f.read(8))[0]
            # Skip track entries (image_id, point2d_idx pairs)
            f.read(track_len * 8)
            points.append(xyz)
            colors.append([c / 255.0 for c in rgb])
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def load_scene_4d_pku(scene_dir: str, num_frames: int = 0,
                      frame_stride: int = 1, test_every: int = 7,
                      device: str = "cuda"):
    """Load PKU-DyMVHumans scene for 4D training.

    Args:
        scene_dir: Path to scene (e.g. /workspace/Data/PKU-DyMVHumans/4K_Studios_Show_Pair_f16f17)
        num_frames: Max frames to use (0 = all available)
        frame_stride: Temporal subsampling stride
        test_every: Hold out every Nth camera for testing (0 = no holdout)
        device: torch device

    Returns:
        Same interface as load_scene_4d():
        frame_paths, cam_names, camtoworlds, K, timestamps, near, far, H, W
    """
    scene_dir = Path(scene_dir)
    cams_dir = scene_dir / "cams"
    per_frame_dir = scene_dir / "per_frame"

    # ── Read camera parameters from cam_txt files ──
    cam_files = sorted(cams_dir.glob("*_cam.txt"))
    if not cam_files:
        raise FileNotFoundError(f"No cam_txt files in {cams_dir}")

    all_cam_names = []
    all_c2w = []
    all_intrinsics = []
    all_depth_params = []

    for cam_file in cam_files:
        extrinsic, intrinsic, depth_params = read_cam_txt(str(cam_file))

        # Camera name from filename: 00000000_cam.txt → 00000000
        cam_name = cam_file.stem.replace("_cam", "")
        all_cam_names.append(cam_name)

        # Extrinsic is world-to-camera; invert to get camera-to-world
        c2w = np.linalg.inv(extrinsic)
        all_c2w.append(c2w.astype(np.float32))
        all_intrinsics.append(intrinsic.astype(np.float32))
        all_depth_params.append(depth_params)

    # ── Train/test split ──
    if test_every > 0:
        train_indices = [i for i in range(len(all_cam_names)) if i % test_every != 0]
        test_indices = [i for i in range(len(all_cam_names)) if i % test_every == 0]
    else:
        train_indices = list(range(len(all_cam_names)))
        test_indices = []

    cam_names = [all_cam_names[i] for i in train_indices]
    c2w_train = np.stack([all_c2w[i] for i in train_indices])
    intrinsics_train = [all_intrinsics[i] for i in train_indices]
    depth_params_train = np.stack([all_depth_params[i] for i in train_indices])
    C = len(cam_names)

    print(f"PKU cameras: {len(all_cam_names)} total, {C} train, "
          f"{len(test_indices)} test (every {test_every}th held out)")

    # ── Discover frame directories ──
    frame_dirs = sorted(per_frame_dir.glob("[0-9]*"))
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories in {per_frame_dir}")

    frame_dirs = frame_dirs[::frame_stride]
    if num_frames > 0:
        frame_dirs = frame_dirs[:num_frames]
    T = len(frame_dirs)

    # ── Build frame_paths[t][c] ──
    # Images: per_frame/FFFFFF/images/image_c_NNN_f_FFFFFF.png
    # Camera name 00000NNN maps to camera index NNN (3-digit zero-padded)
    frame_paths = []
    for fdir in frame_dirs:
        img_dir = fdir / "images"
        if not img_dir.exists():
            img_dir = fdir

        frame_id = fdir.name  # e.g. "000000"
        paths = []
        for i, cam_name in enumerate(cam_names):
            # Camera index from cam_name (e.g. "00000003" → 3)
            cam_idx = int(cam_name)
            cam_code = f"c_{cam_idx:03d}"  # "c_003"
            candidates = [
                img_dir / f"image_{cam_code}_f_{frame_id}.png",
                img_dir / f"image_{cam_name}.png",
                img_dir / f"{cam_name}.png",
                img_dir / f"image_{cam_name}.jpg",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = str(c)
                    break
            if found is None:
                raise FileNotFoundError(
                    f"No image for camera {cam_name} (idx {cam_idx}) in {img_dir}. "
                    f"Tried: {[c.name for c in candidates]}")
            paths.append(found)
        frame_paths.append(paths)

    # ── Get image dimensions ──
    sample = np.array(Image.open(frame_paths[0][0]))
    H, W = sample.shape[:2]

    # ── Build camera tensors ──
    camtoworlds = torch.tensor(c2w_train, device=device)

    # Build per-camera intrinsics
    K = torch.zeros((C, 3, 3), device=device)
    for i, intr in enumerate(intrinsics_train):
        K[i] = torch.tensor(intr, device=device)

    # ── Timestamps ──
    if T == 1:
        timestamps = torch.tensor([0.5], device=device)
    else:
        timestamps = torch.linspace(0.0, 1.0, T, device=device)

    # ── Depth bounds ──
    near = float(depth_params_train[:, 0].min())
    far = float(depth_params_train[:, 3].max()) if depth_params_train.shape[1] >= 4 else 100.0

    print(f"PKU scene: {T} frames x {C} cameras at {H}x{W}, "
          f"near={near:.2f}, far={far:.2f}")
    return frame_paths, cam_names, camtoworlds, K, timestamps, near, far, H, W


def load_sfm_points_pku(scene_dir: str, frame_idx: str = None,
                        device: str = "cuda"):
    """Load SfM points from PKU COLMAP data.

    Args:
        scene_dir: Path to PKU scene
        frame_idx: Which frame's COLMAP to use (default: first available)
        device: torch device

    Returns:
        points: (N, 3) tensor, or None
        colors: (N, 3) tensor, or None
    """
    scene_dir = Path(scene_dir)
    colmap_dir = scene_dir / "data_COLMAP"

    if not colmap_dir.exists():
        print(f"No COLMAP data at {colmap_dir}")
        return None, None

    # Find first available frame with points3D
    if frame_idx is not None:
        frame_dirs = [colmap_dir / frame_idx]
    else:
        frame_dirs = sorted(colmap_dir.glob("[0-9]*"))

    for fdir in frame_dirs:
        # Check common COLMAP layouts: sparse/0/, sparse/, or root
        pts_path = fdir / "sparse" / "0" / "points3D.bin"
        if not pts_path.exists():
            pts_path = fdir / "sparse" / "points3D.bin"
        if not pts_path.exists():
            pts_path = fdir / "points3D.bin"
        if pts_path.exists():
            points, colors = read_colmap_points3d_bin(str(pts_path))
            print(f"Loaded {len(points)} SfM points from {pts_path}")
            return (torch.tensor(points, device=device),
                    torch.tensor(colors, device=device))

    print(f"No points3D.bin found in {colmap_dir}")
    return None, None
