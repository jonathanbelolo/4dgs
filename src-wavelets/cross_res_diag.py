"""Cross-resolution diagnostic: evaluate stage 1 model at 100px."""
import torch
import svox2
import numpy as np
from data import NerfSyntheticDataset
from train_frequency_matched import make_svox2_camera, render_svox2_image

device = "cuda:0"

def eval_psnr(grid, dataset, label):
    psnrs = []
    with torch.no_grad():
        for i in range(min(25, dataset.n_images)):
            H = W = dataset.resolution
            pose = dataset.poses[i]
            im = render_svox2_image(grid, pose, dataset.focal, W, device)
            im.clamp_(0.0, 1.0)
            gt = dataset.images[i]
            mse = ((im - gt) ** 2).mean()
            psnr = -10.0 * torch.log10(mse)
            psnrs.append(psnr.item())
    avg = np.mean(psnrs)
    print(f"  {label}: PSNR={avg:.2f} dB (avg {len(psnrs)} views)")
    return avg

# Load stage 1 best checkpoint
print("Loading stage 1 best checkpoint...")
grid1 = svox2.SparseGrid.load(
    "output/fm_poc/lego/fm/best_stage1_200px_264.npz", device=device
)
grid1.opt.near_clip = 0.0
grid1.opt.background_brightness = 1.0

# Load test data at both resolutions
print("Loading test data...")
test_100 = NerfSyntheticDataset(
    "data/nerf_synthetic", "lego", "test", resolution=100, device=device, white_bg=True,
)
test_200 = NerfSyntheticDataset(
    "data/nerf_synthetic", "lego", "test", resolution=200, device=device, white_bg=True,
)

print("\n=== Cross-Resolution Diagnostic ===")
psnr_s1_100 = eval_psnr(grid1, test_100, "Stage 1 model @ 100px")
psnr_s1_200 = eval_psnr(grid1, test_200, "Stage 1 model @ 200px")

# Load stage 0 for reference
print("\nLoading stage 0 checkpoint...")
grid0 = svox2.SparseGrid.load(
    "output/fm_poc/lego/fm/stage0.npz", device=device
)
grid0.opt.near_clip = 0.0
grid0.opt.background_brightness = 1.0
psnr_s0_100 = eval_psnr(grid0, test_100, "Stage 0 model @ 100px")

print(f"\n=== Summary ===")
print(f"Stage 0 @ 100px: {psnr_s0_100:.2f} dB")
print(f"Stage 1 @ 100px: {psnr_s1_100:.2f} dB  (delta: {psnr_s1_100 - psnr_s0_100:+.2f} dB)")
print(f"Stage 1 @ 200px: {psnr_s1_200:.2f} dB")
if psnr_s1_100 >= psnr_s0_100 - 0.5:
    print("PASS: Base quality preserved (within 0.5 dB)")
else:
    print(f"FAIL: Base damaged by {psnr_s0_100 - psnr_s1_100:.2f} dB")
