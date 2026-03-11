"""Render orbit videos and side-by-side comparisons from trained models."""

import argparse
import os
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval import load_model
from data import NerfSyntheticDataset
from renderer import render_image


def orbit_pose(theta: float, phi: float, radius: float) -> torch.Tensor:
    """Generate camera-to-world matrix for an orbit camera.

    Args:
        theta: Azimuth angle in radians (0 = front, pi/2 = right).
        phi: Elevation angle in radians (0 = horizontal, pi/2 = top).
        radius: Distance from origin.

    Returns:
        pose: (4, 4) camera-to-world matrix.
    """
    # Camera position on sphere
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi)
    z = radius * np.cos(phi) * np.cos(theta)
    pos = np.array([x, y, z])

    # Look-at: camera points toward origin
    forward = -pos / np.linalg.norm(pos)
    # Up vector: world Y
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Camera-to-world: columns are right, up, -forward, position
    # (OpenGL convention: camera looks along -Z)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = pos

    return torch.from_numpy(pose)


def render_orbit(
    checkpoint_path: str,
    n_frames: int = 120,
    resolution: int = 800,
    elevation_deg: float = 30.0,
    radius: float = 4.0,
    output_dir: str = "orbit_frames",
):
    """Render a 360-degree orbit video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_model(checkpoint_path, device)

    # Load focal length from dataset, scaled to render resolution
    try:
        data = NerfSyntheticDataset(
            config.data_dir, config.scene, "test",
            resolution=config.train_resolution,
            white_bg=config.white_background,
            device="cpu",
        )
        focal = data.focal * resolution / config.train_resolution
    except Exception:
        # Fallback: standard NeRF Synthetic FOV
        focal = 0.5 * resolution / np.tan(0.5 * 0.6911112070083618)

    os.makedirs(output_dir, exist_ok=True)
    phi = np.radians(elevation_deg)

    print(f"Rendering {n_frames} frames at {resolution}×{resolution}...")
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        pose = orbit_pose(theta, phi, radius).to(device)

        with torch.no_grad():
            result = render_image(
                model, pose, resolution, resolution, focal, config,
            )

        img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"{i:04d}.png"))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_frames}")

    print(f"Frames saved to {output_dir}/")
    return output_dir


def frames_to_video(frame_dir: str, output_path: str, fps: int = 30):
    """Stitch PNG frames into an MP4 video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Video saved to {output_path}")


def render_comparison(
    checkpoint_path: str,
    split: str = "test",
    n_views: int = 8,
    resolution: int = 800,
):
    """Render side-by-side comparison: ground truth vs model.

    Saves comparison images with GT on left, rendered on right.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_model(checkpoint_path, device)

    data = NerfSyntheticDataset(
        config.data_dir, config.scene, split,
        resolution=resolution,
        white_bg=config.white_background,
        device=device,
    )

    out_dir = os.path.join(config.output_dir, config.scene, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    indices = np.linspace(0, len(data) - 1, n_views, dtype=int)

    print(f"Rendering {n_views} comparison views...")
    for i, idx in enumerate(indices):
        with torch.no_grad():
            result = render_image(
                model, data.poses[idx],
                resolution, resolution, data.focal, config,
            )

        rendered = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gt = (data.images[idx].cpu().numpy() * 255).astype(np.uint8)

        # Side-by-side: GT | Rendered
        comparison = np.concatenate([gt, rendered], axis=1)
        img = Image.fromarray(comparison)

        # Add labels
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Ground Truth", fill=(255, 0, 0))
        draw.text((resolution + 10, 10), "Rendered", fill=(0, 255, 0))

        img.save(os.path.join(out_dir, f"compare_{idx:04d}.png"))
        print(f"  View {idx}: saved")

    print(f"Comparisons saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render videos and comparisons")
    parser.add_argument("--checkpoint", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    # Orbit video
    orbit = sub.add_parser("orbit", help="Render 360° orbit video")
    orbit.add_argument("--frames", type=int, default=120)
    orbit.add_argument("--resolution", type=int, default=800)
    orbit.add_argument("--elevation", type=float, default=30.0)
    orbit.add_argument("--radius", type=float, default=4.0)
    orbit.add_argument("--fps", type=int, default=30)
    orbit.add_argument("--output", default="orbit.mp4")

    # Comparison
    compare = sub.add_parser("compare", help="Side-by-side GT vs rendered")
    compare.add_argument("--split", default="test")
    compare.add_argument("--views", type=int, default=8)
    compare.add_argument("--resolution", type=int, default=800)

    args = parser.parse_args()

    if args.command == "orbit":
        frame_dir = render_orbit(
            args.checkpoint,
            n_frames=args.frames,
            resolution=args.resolution,
            elevation_deg=args.elevation,
            radius=args.radius,
        )
        frames_to_video(frame_dir, args.output, fps=args.fps)

    elif args.command == "compare":
        render_comparison(
            args.checkpoint,
            split=args.split,
            n_views=args.views,
            resolution=args.resolution,
        )
