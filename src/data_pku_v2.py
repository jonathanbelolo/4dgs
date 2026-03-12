"""Data loader for PKU-DyMVHumans per_view format with COLMAP calibration.

Loads directly from the per_view/ directory structure with calibration from
colmap_calibrate_pku.py. Compatible with train_4d.py's expected interface.

Directory structure:
    scene_dir/
        calibration.npz          # From colmap_calibrate_pku.py
        canonical_frame.npz      # From canonical_bbox.py
        per_view/
            cam_0/
                images/000000.png ... 000249.png
                pha/000000.png ... 000249.png       # Foreground masks
                semantic/000000.png ... 000249.png  # 5-class semantic labels
            cam_1/
                ...
        smplx_params/
            frame_000000.npz ... frame_000249.npz
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_scene_4d_pku_v2(scene_dir, num_frames=0, frame_stride=1,
                          test_every=0, downsample=1, device="cuda"):
    """Load PKU-DyMVHumans scene in per_view format for 4D training.

    Uses COLMAP calibration (calibration.npz) instead of MVSNet cam_txt files.

    Args:
        scene_dir: Path to scene directory
        num_frames: Max frames to use (0 = all available)
        frame_stride: Temporal subsampling stride
        test_every: Hold out every Nth camera for testing (0 = no holdout)
        downsample: Spatial downsample factor (1 = original resolution)
        device: torch device

    Returns:
        frame_paths: list of T lists, frame_paths[t][c] = path to RGB image
        mask_paths: list of T lists, mask_paths[t][c] = path to mask image
        cam_names: list of C camera name strings
        camtoworlds: (C, 4, 4) float32 tensor on device
        K: (C, 3, 3) float32 tensor on device
        timestamps: (T,) float32 tensor on device, normalized to [0, 1]
        near: float
        far: float
        H: int (image height after downsample)
        W: int (image width after downsample)
    """
    scene_dir = Path(scene_dir)
    per_view = scene_dir / "per_view"

    # ── Load calibration ──────────────────────────────────────────────────────
    calib_path = scene_dir / "calibration.npz"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"calibration.npz not found in {scene_dir}. "
            "Run colmap_calibrate_pku.py first."
        )

    calib = np.load(str(calib_path), allow_pickle=True)
    c2w_all = calib["c2w"]       # (N_cams, 4, 4)
    K_all = calib["K"]           # (N_cams, 3, 3)
    all_cam_names = list(calib["cam_names"])
    registered = calib["registered"]  # (N_cams,) bool

    # ── Filter to registered cameras, optionally hold out test cameras ────────
    reg_indices = np.where(registered)[0]
    train_indices = []
    test_indices = []

    for i, idx in enumerate(reg_indices):
        if test_every > 0 and (i % test_every) == 0:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    cam_indices = train_indices
    cam_names = [all_cam_names[i] for i in cam_indices]
    C = len(cam_names)

    print(f"Loaded {C} training cameras" +
          (f" ({len(test_indices)} held out for testing)" if test_indices else ""))

    # ── Camera matrices ───────────────────────────────────────────────────────
    c2w = np.stack([c2w_all[i] for i in cam_indices])  # (C, 4, 4)
    K = np.stack([K_all[i] for i in cam_indices])       # (C, 3, 3)

    # Apply downsample to intrinsics
    if downsample > 1:
        K[:, 0, :] /= downsample  # fx, cx
        K[:, 1, :] /= downsample  # fy, cy

    camtoworlds = torch.tensor(c2w, dtype=torch.float32, device=device)
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)

    # ── Enumerate frames ──────────────────────────────────────────────────────
    # Use first camera to discover available frames
    sample_images_dir = per_view / cam_names[0] / "images"
    all_frame_files = sorted(sample_images_dir.glob("*.png"))

    if not all_frame_files:
        raise FileNotFoundError(f"No images found in {sample_images_dir}")

    # Apply stride
    frame_files = all_frame_files[::frame_stride]

    # Limit number of frames
    if num_frames > 0:
        frame_files = frame_files[:num_frames]

    T = len(frame_files)
    frame_names = [f.stem for f in frame_files]  # e.g. ["000000", "000001", ...]

    # ── Build frame path index (lazy loading) ─────────────────────────────────
    frame_paths = []  # frame_paths[t][c] = str path
    mask_paths = []   # mask_paths[t][c] = str path

    for frame_name in frame_names:
        frame_images = []
        frame_masks = []
        for cam_name in cam_names:
            img_path = str(per_view / cam_name / "images" / f"{frame_name}.png")
            mask_path = str(per_view / cam_name / "pha" / f"{frame_name}.png")
            frame_images.append(img_path)
            frame_masks.append(mask_path)
        frame_paths.append(frame_images)
        mask_paths.append(frame_masks)

    # ── Image dimensions ──────────────────────────────────────────────────────
    sample_img = Image.open(frame_paths[0][0])
    W_orig, H_orig = sample_img.size
    W = W_orig // downsample
    H = H_orig // downsample

    # ── Timestamps ────────────────────────────────────────────────────────────
    if T > 1:
        timestamps = torch.linspace(0, 1, T, dtype=torch.float32, device=device)
    else:
        timestamps = torch.tensor([0.5], dtype=torch.float32, device=device)

    # ── Near/far from camera positions + canonical frame ──────────────────────
    positions = c2w[:, :3, 3]
    cam_center = positions.mean(axis=0)
    cam_radius = np.linalg.norm(positions - cam_center, axis=1).mean()

    # Load canonical frame for scene extent if available
    canonical_path = scene_dir / "canonical_frame.npz"
    if canonical_path.exists():
        canonical = np.load(str(canonical_path), allow_pickle=True)
        person_height = float(canonical["person_height"])
        near = max(0.1, cam_radius - person_height * 2)
        far = cam_radius + person_height * 2
    else:
        near = cam_radius * 0.1
        far = cam_radius * 3.0

    print(f"Scene: {T} frames x {C} cameras, {W}x{H}px")
    print(f"  near={near:.2f}, far={far:.2f}, cam_radius={cam_radius:.2f}")

    return (frame_paths, mask_paths, cam_names, camtoworlds, K_tensor,
            timestamps, near, far, H, W)


def load_test_cameras(scene_dir, test_every=7, downsample=1, device="cuda"):
    """Load held-out test cameras for evaluation.

    Args:
        scene_dir: Path to scene directory
        test_every: Same value used in load_scene_4d_pku_v2
        downsample: Spatial downsample factor
        device: torch device

    Returns:
        Same format as load_scene_4d_pku_v2 but for test cameras only
    """
    scene_dir = Path(scene_dir)
    per_view = scene_dir / "per_view"
    calib = np.load(str(scene_dir / "calibration.npz"), allow_pickle=True)

    c2w_all = calib["c2w"]
    K_all = calib["K"]
    all_cam_names = list(calib["cam_names"])
    registered = calib["registered"]
    reg_indices = np.where(registered)[0]

    test_indices = [idx for i, idx in enumerate(reg_indices)
                    if test_every > 0 and (i % test_every) == 0]

    if not test_indices:
        return None

    cam_names = [all_cam_names[i] for i in test_indices]
    C = len(cam_names)

    c2w = np.stack([c2w_all[i] for i in test_indices])
    K = np.stack([K_all[i] for i in test_indices])
    if downsample > 1:
        K[:, 0, :] /= downsample
        K[:, 1, :] /= downsample

    camtoworlds = torch.tensor(c2w, dtype=torch.float32, device=device)
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)

    # Enumerate frames
    sample_dir = per_view / cam_names[0] / "images"
    all_frames = sorted(sample_dir.glob("*.png"))
    T = len(all_frames)
    frame_names = [f.stem for f in all_frames]

    frame_paths = []
    mask_paths = []
    for frame_name in frame_names:
        imgs = [str(per_view / cn / "images" / f"{frame_name}.png") for cn in cam_names]
        masks = [str(per_view / cn / "pha" / f"{frame_name}.png") for cn in cam_names]
        frame_paths.append(imgs)
        mask_paths.append(masks)

    sample_img = Image.open(frame_paths[0][0])
    W = sample_img.width // downsample
    H = sample_img.height // downsample
    timestamps = torch.linspace(0, 1, T, dtype=torch.float32, device=device) if T > 1 else torch.tensor([0.5], dtype=torch.float32, device=device)

    print(f"Test set: {T} frames x {C} cameras, {W}x{H}px")
    return (frame_paths, mask_paths, cam_names, camtoworlds, K_tensor,
            timestamps, H, W)


def load_image(path, downsample=1, device="cuda"):
    """Load a single image as a torch tensor.

    Args:
        path: str or Path to image
        downsample: downsample factor
        device: torch device

    Returns:
        (H, W, 3) float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    if downsample > 1:
        W, H = img.size
        img = img.resize((W // downsample, H // downsample), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=device)


def load_mask(path, downsample=1, device="cuda"):
    """Load a foreground mask as a torch tensor.

    Args:
        path: str or Path to mask image
        downsample: downsample factor
        device: torch device

    Returns:
        (H, W) float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("L")
    if downsample > 1:
        W, H = img.size
        img = img.resize((W // downsample, H // downsample), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=device)


def load_semantic(path, downsample=1, device="cuda"):
    """Load a semantic label map as a torch tensor.

    Args:
        path: str or Path to semantic mask (uint8, values 0-4)
        downsample: downsample factor
        device: torch device

    Returns:
        (H, W) long tensor with values 0-4
    """
    img = Image.open(path)
    if downsample > 1:
        W, H = img.size
        img = img.resize((W // downsample, H // downsample), Image.NEAREST)
    arr = np.array(img, dtype=np.int64)
    return torch.tensor(arr, dtype=torch.long, device=device)


def load_smplx_params(scene_dir, frame_idx, device="cuda"):
    """Load SMPL-X parameters for a specific frame.

    Args:
        scene_dir: Path to scene directory
        frame_idx: int frame index
        device: torch device

    Returns:
        dict with torch tensors: vertices (V, 3), joints (J, 3),
        skinning_weights (V, 55), semantic_labels (V,), etc.
        or None if not found
    """
    scene_dir = Path(scene_dir)
    path = scene_dir / "smplx_params" / f"frame_{frame_idx:06d}.npz"
    if not path.exists():
        return None

    data = np.load(str(path), allow_pickle=True)
    result = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype in (np.float32, np.float64):
            result[key] = torch.tensor(arr, dtype=torch.float32, device=device)
        elif arr.dtype in (np.uint8, np.int32, np.int64):
            result[key] = torch.tensor(arr, dtype=torch.long, device=device)
        else:
            result[key] = arr

    return result
