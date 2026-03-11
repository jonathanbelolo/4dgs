"""NeRF Synthetic dataset loader for gsplat-based 3DGS training.

Outputs in OpenCV convention (y-down, z-forward) matching gsplat.
Supports Lanczos downsampling for frequency-matched multi-resolution training.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# OpenGL (NeRF Synthetic native) -> OpenCV (gsplat): negate y and z
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_nerf_synthetic(root: str, scene: str, split: str = "train",
                        resolution: int = 800, device: str = "cuda"):
    """Load NeRF Synthetic scene at target resolution.

    Args:
        root: Path to nerf_synthetic directory (contains scene folders)
        scene: Scene name (e.g. "lego", "chair", "hotdog")
        split: "train", "val", or "test"
        resolution: Target image resolution (square). Lanczos-downsampled from 800px.
        device: torch device

    Returns:
        images: (N, H, W, 3) float32 tensor in [0, 1], white background
        camtoworlds: (N, 4, 4) float32 tensor, OpenCV convention
        K: (N, 3, 3) float32 intrinsics matrices
        near: float (2.0 for synthetic scenes)
        far: float (6.0 for synthetic scenes)
    """
    scene_dir = Path(root) / scene

    with open(scene_dir / f"transforms_{split}.json") as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    focal = 0.5 * resolution / np.tan(0.5 * camera_angle_x)

    images = []
    poses = []

    for frame in meta["frames"]:
        # Load RGBA, Lanczos downsample
        fpath = scene_dir / f"{frame['file_path']}.png"
        img = Image.open(fpath).resize((resolution, resolution), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4)

        # Alpha composite onto white background
        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + (1.0 - alpha)
        images.append(img)

        # Pose: GL -> CV convention
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w_cv = c2w_gl @ _GL_TO_CV
        poses.append(c2w_cv)

    images = torch.from_numpy(np.stack(images)).to(device)
    camtoworlds = torch.from_numpy(np.stack(poses)).to(device)

    # Intrinsics matrix
    N = len(images)
    K = torch.zeros(N, 3, 3, device=device)
    K[:, 0, 0] = focal
    K[:, 1, 1] = focal
    K[:, 0, 2] = resolution / 2.0
    K[:, 1, 2] = resolution / 2.0
    K[:, 2, 2] = 1.0

    # NeRF Synthetic scenes: cameras on sphere radius ~4, object radius ~1
    near, far = 2.0, 6.0

    print(f"[{split}] {N} images at {resolution}x{resolution}, focal={focal:.1f}")
    return images, camtoworlds, K, near, far
