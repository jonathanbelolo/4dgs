"""Evaluate a saved svox2 checkpoint on the full 200-view test set."""
import sys
sys.path.insert(0, "/workspace/src-wavelets")

import torch
import svox2
from train_frequency_matched import evaluate_full_svox2, NerfSyntheticDataset

device = "cuda"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "output/fm_poc4/lego/fm/stage3.npz"
img_res = int(sys.argv[2]) if len(sys.argv) > 2 else 800

print(f"Loading checkpoint: {ckpt}")
grid = svox2.SparseGrid.load(ckpt, device=device)
print(f"Grid: {grid.links.shape[0]}^3, {grid.density_data.shape[0]:,} voxels")

test_data = NerfSyntheticDataset(
    "data/nerf_synthetic", "lego", "test",
    resolution=img_res, white_bg=True, device=device,
)
print(f"Loaded {len(test_data)} test images at {img_res}px")

from config import Config
cfg = Config(scene="lego")

metrics = evaluate_full_svox2(
    grid, test_data, cfg, "output/fm_poc4/lego/fm",
    tag=f"full_test_{img_res}px", n_views=200,
)
print(f"\nFull test set (200 views, {img_res}px):")
print(f"  PSNR: {metrics['avg_psnr']:.2f} dB")
print(f"  SSIM: {metrics['avg_ssim']:.4f}")
