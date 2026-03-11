"""NeRF Synthetic dataset loader."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class NerfSyntheticDataset:
    """Loads NeRF Synthetic (Blender) scenes.

    Each scene has train/val/test splits with transforms.json containing
    camera intrinsics and per-image extrinsics (4x4 camera-to-world matrices).
    """

    def __init__(self, root: str, scene: str, split: str = "train",
                 resolution: int = 800, white_bg: bool = True,
                 device: str = "cuda"):
        self.root = Path(root) / scene
        self.split = split
        self.resolution = resolution
        self.white_bg = white_bg
        self.device = device

        # Load transforms
        with open(self.root / f"transforms_{split}.json") as f:
            meta = json.load(f)

        # Camera intrinsics (shared across all images in a split)
        camera_angle_x = meta["camera_angle_x"]
        self.focal = 0.5 * resolution / np.tan(0.5 * camera_angle_x)

        # Load images and poses
        images = []
        poses = []
        for frame in meta["frames"]:
            # Image
            fpath = self.root / f"{frame['file_path']}.png"
            img = Image.open(fpath).resize((resolution, resolution), Image.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4) RGBA

            if white_bg:
                # Alpha composite onto white background
                alpha = img[..., 3:4]
                img = img[..., :3] * alpha + (1.0 - alpha)
            else:
                img = img[..., :3]

            images.append(img)

            # Pose (4x4 camera-to-world)
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            poses.append(pose)

        self.images = torch.from_numpy(np.stack(images)).to(device)  # (N, H, W, 3)
        self.poses = torch.from_numpy(np.stack(poses)).to(device)    # (N, 4, 4)
        self.n_images = len(images)

        print(f"Loaded {split}: {self.n_images} images at {resolution}x{resolution}, "
              f"focal={self.focal:.1f}")

    def __len__(self):
        return self.n_images

    def get_rays(self, idx: int):
        """Generate all rays for image idx.

        Returns:
            rays_o: (H*W, 3) ray origins
            rays_d: (H*W, 3) ray directions (normalized)
            rgb_gt: (H*W, 3) ground truth colors
        """
        H = W = self.resolution
        pose = self.poses[idx]  # (4, 4) camera-to-world

        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(W, device=self.device, dtype=torch.float32),
            torch.arange(H, device=self.device, dtype=torch.float32),
            indexing="xy"
        )

        # Camera-space directions (OpenGL convention: -Z forward)
        dirs = torch.stack([
            (i - W * 0.5) / self.focal,
            -(j - H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)

        # Transform to world space
        rays_d = (dirs[..., None, :] * pose[:3, :3]).sum(-1)  # (H, W, 3)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_o = pose[:3, 3].expand_as(rays_d)  # (H, W, 3)

        return (
            rays_o.reshape(-1, 3),
            rays_d.reshape(-1, 3),
            self.images[idx].reshape(-1, 3),
        )

    def get_random_rays(self, batch_size: int):
        """Sample random rays across all training images.

        Returns:
            rays_o: (batch_size, 3)
            rays_d: (batch_size, 3)
            rgb_gt: (batch_size, 3)
        """
        H = W = self.resolution

        # Random image indices
        img_idx = torch.randint(0, self.n_images, (batch_size,), device=self.device)
        # Random pixel coordinates
        pix_y = torch.randint(0, H, (batch_size,), device=self.device)
        pix_x = torch.randint(0, W, (batch_size,), device=self.device)

        # Ground truth colors
        rgb_gt = self.images[img_idx, pix_y, pix_x]  # (batch_size, 3)

        # Camera-space directions
        dirs = torch.stack([
            (pix_x.float() - W * 0.5) / self.focal,
            -(pix_y.float() - H * 0.5) / self.focal,
            -torch.ones(batch_size, device=self.device),
        ], dim=-1)  # (batch_size, 3)

        # Per-ray poses
        poses = self.poses[img_idx]  # (batch_size, 4, 4)
        rays_d = (dirs[:, None, :] * poses[:, :3, :3]).sum(-1)  # (batch_size, 3)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_o = poses[:, :3, 3]  # (batch_size, 3)

        return rays_o, rays_d, rgb_gt
