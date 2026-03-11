"""Data loading for Neu3D dataset with LLFF-format camera poses."""
import struct
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_llff_poses(poses_bounds_path: str):
    """Load and parse LLFF poses_bounds.npy.

    Returns:
        c2w: (N, 4, 4) camera-to-world matrices
        hwf: (N, 3) height, width, focal per camera
        bounds: (N, 2) near, far bounds
    """
    data = np.load(poses_bounds_path)  # (N, 17)
    poses = data[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
    bounds = data[:, 15:17]  # (N, 2)

    # LLFF format: columns are [R | t | hwf], stored as 3x5
    # R is 3x3 rotation, t is 3x1 translation
    hwf = poses[:, :, 4]  # (N, 3) — [H, W, F]

    # Build 4x4 camera-to-world matrices
    c2w = np.zeros((poses.shape[0], 4, 4), dtype=np.float64)
    c2w[:, :3, :4] = poses[:, :, :4]  # [R | t]
    c2w[:, 3, 3] = 1.0

    # LLFF stores rotation columns as DRB (Down, Right, Backwards).
    # gsplat expects OpenCV convention: RDF (Right, Down, Forward).
    # Convert: new_col0=old_col1 (Right), new_col1=old_col0 (Down),
    #          new_col2=-old_col2 (Forward = -Backwards). Translation unchanged.
    c2w_cv = np.zeros_like(c2w)
    c2w_cv[:, :3, 0] = c2w[:, :3, 1]    # Right
    c2w_cv[:, :3, 1] = c2w[:, :3, 0]    # Down
    c2w_cv[:, :3, 2] = -c2w[:, :3, 2]   # Forward = -Backwards
    c2w_cv[:, :3, 3] = c2w[:, :3, 3]    # translation unchanged
    c2w_cv[:, 3, 3] = 1.0

    return c2w_cv.astype(np.float32), hwf.astype(np.float32), bounds.astype(np.float32)


def load_scene(scene_dir: str, frames_subdir: str = "frames/train",
               downsample: int = 2, device: str = "cuda"):
    """Load images and cameras for a Neu3D scene.

    Args:
        scene_dir: Path to scene (e.g. /workspace/Data/Neu3D/coffee_martini)
        frames_subdir: Subdirectory containing extracted frames
        downsample: Downsample factor used during frame extraction
        device: torch device

    Returns:
        images: (N, H, W, 3) float32 tensor in [0, 1]
        camtoworlds: (N, 4, 4) float32 tensor
        K: (N, 3, 3) float32 intrinsics matrices
        near: float
        far: float
    """
    scene_dir = Path(scene_dir)
    frames_dir = scene_dir / frames_subdir

    # Load poses
    c2w, hwf, bounds = load_llff_poses(str(scene_dir / "poses_bounds.npy"))

    # Camera ordering: poses_bounds.npy is ordered by camera index
    # but cam00 is test, and some cameras may be missing
    # Match cameras to available images
    image_files = sorted(frames_dir.glob("cam*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    # Map camera names to indices in poses_bounds
    all_cams = sorted(scene_dir.glob("cam*.mp4"))
    cam_name_to_idx = {v.stem: i for i, v in enumerate(all_cams)}

    images = []
    camtoworlds = []
    focals = []

    for img_file in image_files:
        cam_name = img_file.stem.split("_")[0] if "_" in img_file.stem else img_file.stem
        idx = cam_name_to_idx[cam_name]

        # Load image
        img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
        images.append(img)

        # Camera pose
        camtoworlds.append(c2w[idx])

        # Focal length (adjusted for downsample)
        focals.append(float(hwf[idx, 2] / downsample))

    images = torch.tensor(np.stack(images), device=device)
    camtoworlds = torch.tensor(np.stack(camtoworlds), device=device)

    # Build intrinsics matrices
    H, W = images.shape[1], images.shape[2]
    K = torch.zeros((len(images), 3, 3), device=device)
    for i, f in enumerate(focals):
        K[i, 0, 0] = f  # fx
        K[i, 1, 1] = f  # fy
        K[i, 0, 2] = W / 2.0  # cx
        K[i, 1, 2] = H / 2.0  # cy
        K[i, 2, 2] = 1.0

    near = float(bounds[:, 0].min())
    far = float(bounds[:, 1].max())

    print(f"Loaded {len(images)} images at {H}x{W}, near={near:.2f}, far={far:.2f}")
    return images, camtoworlds, K, near, far


def load_mono_depths(scene_dir: str, frames_subdir: str = "frames/train",
                     device: str = "cuda"):
    """Load monocular depth maps (from Depth Anything V2) for training cameras.

    Returns:
        depths: (N, H, W) float32 tensor with relative depth in [0, 1], or None
    """
    scene_dir = Path(scene_dir)
    mono_dir = scene_dir / "mono_depth"

    if not mono_dir.exists():
        return None

    frames_dir = scene_dir / frames_subdir
    image_files = sorted(frames_dir.glob("cam*.jpg"))

    depths = []
    for img_file in image_files:
        cam_name = img_file.stem.split("_")[0] if "_" in img_file.stem else img_file.stem
        depth_path = mono_dir / f"{cam_name}_train.npy"

        if not depth_path.exists():
            print(f"  WARNING: No mono depth for {cam_name}")
            return None

        depth = np.load(str(depth_path)).astype(np.float32)
        # Resize to match training image if needed
        sample = np.array(Image.open(img_file))
        target_h, target_w = sample.shape[:2]
        if depth.shape != (target_h, target_w):
            from PIL import Image as PILImage
            depth_pil = PILImage.fromarray(depth, mode='F')
            depth_pil = depth_pil.resize((target_w, target_h), PILImage.BILINEAR)
            depth = np.array(depth_pil)

        depths.append(depth)

    depths = torch.tensor(np.stack(depths), device=device)
    print(f"Loaded {len(depths)} monocular depth maps, shape={depths.shape[1:]},"
          f" range=[{depths.min():.3f}, {depths.max():.3f}]")
    return depths


def read_colmap_depth_bin(path):
    """Read a COLMAP .geometric.bin depth map."""
    with open(path, 'rb') as f:
        data = f.read()
    # Header is ASCII: 'width&height&channels&' then binary float32
    ampersand_count = 0
    for i, b in enumerate(data):
        if b == ord('&'):
            ampersand_count += 1
            if ampersand_count == 3:
                header_end = i + 1
                break
    header = data[:header_end].decode('ascii')
    parts = header.split('&')
    w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
    depth = np.frombuffer(data[header_end:], dtype=np.float32).reshape(h, w)
    return depth


def load_depth_maps(scene_dir: str, frames_subdir: str = "frames/train",
                    downsample: int = 2, device: str = "cuda"):
    """Load COLMAP dense depth maps for training cameras.

    Returns:
        depths: (N, H, W) float32 tensor, 0 where invalid
        depth_masks: (N, H, W) bool tensor, True where depth is valid
    """
    scene_dir = Path(scene_dir)
    depth_dir = scene_dir / "colmap" / "dense" / "stereo" / "depth_maps"

    if not depth_dir.exists():
        return None, None

    frames_dir = scene_dir / frames_subdir
    image_files = sorted(frames_dir.glob("cam*.jpg"))

    depths = []
    masks = []
    for img_file in image_files:
        cam_name = img_file.stem.split("_")[0] if "_" in img_file.stem else img_file.stem
        depth_path = depth_dir / f"{cam_name}.jpg.geometric.bin"

        if not depth_path.exists():
            # No depth for this camera
            sample = np.array(Image.open(img_file))
            h, w = sample.shape[:2]
            depths.append(np.zeros((h, w), dtype=np.float32))
            masks.append(np.zeros((h, w), dtype=bool))
            continue

        depth = read_colmap_depth_bin(str(depth_path))
        # COLMAP depth maps are at the undistorted resolution (same as colmap/images)
        # which may differ from the training image resolution. Resize if needed.
        sample = np.array(Image.open(img_file))
        target_h, target_w = sample.shape[:2]
        if depth.shape != (target_h, target_w):
            from PIL import Image as PILImage
            depth_img = PILImage.fromarray(depth)
            depth_img = depth_img.resize((target_w, target_h), PILImage.NEAREST)
            depth = np.array(depth_img)

        mask = depth > 0
        depths.append(depth)
        masks.append(mask)

    depths = torch.tensor(np.stack(depths), device=device)
    masks = torch.tensor(np.stack(masks), device=device)
    valid_count = masks.sum().item()
    total = masks.numel()
    print(f"Loaded depth maps: {valid_count}/{total} valid pixels ({100*valid_count/total:.1f}%)")
    return depths, masks
