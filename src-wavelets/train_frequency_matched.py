"""Frequency-matched coarse-to-fine training using svox2 CUDA optimizer.

Uses svox2's SparseGrid for training at each stage. The custom CUDA optimizer
(RMSProp with sparse updates) is ~40x faster than PyTorch Adam for voxel grids.

After each stage, the trained grid is extracted as a dense tensor, wavelet-upsampled
(or trilinear-upsampled) to the next resolution, and used to initialize the next
stage's SparseGrid. Residuals (detail = trained - base) are computed post-hoc
for wavelet analysis.

Four modes:
  fm     - Frequency-matched: image resolution scales with volume resolution
  sp     - Standard progressive: volume upsamples but images stay at full res
  ss     - Single-scale: 520^3 trained from scratch at full resolution
  hybrid - Full-res images + wavelet upsampling (isolates upsampling effect)

Usage:
    python train_frequency_matched.py --mode fm --scene lego
    python train_frequency_matched.py --mode sp --scene lego
    python train_frequency_matched.py --mode ss --scene lego
    python train_frequency_matched.py --mode hybrid --scene lego
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import svox2
except ImportError:
    raise ImportError(
        "svox2 is required for frequency-matched training.\n"
        "Install from: https://github.com/sxyu/svox2\n"
        "This script uses svox2's CUDA kernels for fast voxel grid optimization."
    )

from config import Config
from data import NerfSyntheticDataset
from metrics import psnr, ssim


# --- Default stage schedules ---

# Natural bior4.4 wavelet sizes (each level is exactly the DWT approx of the next):
#   40 -> 72 -> 136 -> 264 -> 520
# These nest perfectly: DWT(264) -> approx=136, DWT(136) -> approx=72, etc.
# Power-of-2 grids (128, 256, 512) do NOT nest -- DWT(256) gives 132, not 128.
#
# Image resolutions are chosen to frequency-match each volume level:
#   volume N over scene 3.0 -> pixel footprint = 3.0/N
#   image res R -> footprint = 3.0*800/(R*1111) = 2.16/R
#   match: R ~ 0.72 * N

# Frequency-matched: image res scales with volume res.
# Max iterations are generous — early stopping on val PSNR handles actual duration.
FM_STAGES = [
    # (image_res, volume_res, max_iterations)
    (100, 136, 200000),   # freq-matched (136 * 0.72 ~ 98)
    (200, 264, 200000),   # freq-matched (264 * 0.72 ~ 190)
    (400, 520, 200000),   # freq-matched (520 * 0.72 ~ 374)
    (800, 520, 200000),   # super-resolution refinement
]

# Standard progressive: full-res images throughout
SP_STAGES = [
    (800, 136, 200000),
    (800, 264, 200000),
    (800, 520, 200000),
    (800, 520, 200000),
]

# Single-scale: 520^3 at full resolution
SS_STAGES = [
    (800, 520, 500000),
]

# Hybrid: full-res images + wavelet upsampling
HYBRID_STAGES = [
    (800, 136, 200000),
    (800, 264, 200000),
    (800, 520, 200000),
    (800, 520, 200000),
]

# OpenGL -> OpenCV camera transform (flip y and z axes)
_GL_TO_CV = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=torch.float32)


# --- Wavelet upsampling ---

def _dwt_sizes_1level(target_size: int, wavelet: str) -> tuple[int, int]:
    """Compute DWT approx and detail sizes via lightweight 1D probe.

    Uses a tiny (target_size x 16 x 16) tensor instead of a full cubic volume.
    Cost: ~target_size * 1 KB, negligible even for target_size=4096.
    """
    import ptwt
    dummy = torch.zeros(1, 1, target_size, 16, 16)
    coeffs = ptwt.wavedec3(dummy, wavelet, level=1)
    approx_size = coeffs[0].shape[2]  # size along first spatial dim
    detail_size = list(coeffs[1].values())[0].shape[2]
    return approx_size, detail_size


def wavelet_upsample(grid: torch.Tensor, target_size: int,
                     wavelet: str = "bior4.4") -> torch.Tensor:
    """Upsample grid from N^3 to target_size^3 via pure wavelet synthesis (W*x).

    The grid IS the approximation coefficients for the target resolution's
    1-level DWT. We apply IDWT with zero detail coefficients to produce the
    smoothest possible upsampling -- a pure linear operator, no interpolation.

    Runs entirely on GPU if the input is on GPU (ptwt supports CUDA tensors).

    Grid size must be a natural wavelet level size. For bior4.4:
    40 -> 72 -> 136 -> 264 -> 520 -> 1032 -> ...

    Args:
        grid: (1, C, N, N, N) dense volume at a natural wavelet size
        target_size: next natural wavelet size (approximately 2N)
        wavelet: wavelet family for synthesis

    Returns:
        (1, C, target_size, target_size, target_size) upsampled volume
    """
    import ptwt

    device = grid.device
    N = grid.shape[-1]
    C = grid.shape[1]

    # Verify sizes are compatible (lightweight probe, not full cubic allocation)
    expected_approx_size, detail_size = _dwt_sizes_1level(target_size, wavelet)

    if N != expected_approx_size:
        raise ValueError(
            f"Grid size {N} does not match the expected DWT approximation "
            f"size {expected_approx_size} for target {target_size} with "
            f"{wavelet}. Use natural wavelet sizes (e.g. 72->136->264->520 "
            f"for bior4.4)."
        )

    # The grid IS the approximation -- keep on same device (GPU if available)
    approx = grid.float()

    # Zero detail coefficients (same device as input)
    zero_details = {
        key: torch.zeros(1, C, detail_size, detail_size, detail_size,
                         device=device, dtype=torch.float32)
        for key in ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]
    }

    # IDWT: pure linear operator W * approx (runs on GPU if input is GPU)
    upsampled = ptwt.waverec3([approx, zero_details], wavelet)

    # Crop to exact target size (IDWT may overshoot by a few voxels)
    upsampled = upsampled[:, :, :target_size, :target_size, :target_size]

    return upsampled


# --- svox2 helpers ---

def create_svox2_grid(resolution: int, scene_bound: float = 1.5,
                      sh_dim: int = 9,
                      device: str = "cuda") -> svox2.SparseGrid:
    """Create a fresh dense svox2 SparseGrid.

    Args:
        resolution: voxel grid resolution (e.g. 136, 264, 520)
        scene_bound: scene fits in [-bound, bound]^3
        sh_dim: SH basis functions per color channel (9 for degree 2)
        device: target device
    """
    grid = svox2.SparseGrid(
        reso=resolution,
        center=[0.0, 0.0, 0.0],
        radius=[scene_bound, scene_bound, scene_bound],
        basis_dim=sh_dim,
        use_z_order=True,
        device=device,
    )
    grid.opt.near_clip = 0.0
    grid.opt.background_brightness = 1.0
    return grid


def dense_to_svox2(svox2_grid: svox2.SparseGrid,
                   dense_volume: torch.Tensor) -> None:
    """Populate svox2 grid data from dense (1, C, R, R, R) tensor.

    Uses the grid's links array to correctly map spatial positions to data
    indices regardless of internal ordering (z-order, raster, etc.).
    Fully vectorized -- no Python loop over channels.

    Args:
        svox2_grid: target SparseGrid (must have same resolution as dense_volume)
        dense_volume: (1, C, R, R, R) dense feature grid
                      Channel 0 = density, channels 1..C-1 = SH coefficients
    """
    device = svox2_grid.density_data.device
    with torch.no_grad():
        flat_links = svox2_grid.links.reshape(-1)
        valid = flat_links >= 0
        indices = flat_links[valid].long()

        C = dense_volume.shape[1]
        # Reshape to (C, R^3) and select valid voxels -> (C, N_valid)
        flat_all = dense_volume[0].reshape(C, -1)[:, valid].to(device)

        # Scatter into svox2 data arrays
        svox2_grid.density_data.data[indices, 0] = flat_all[0]
        n_sh = min(svox2_grid.sh_data.shape[1], C - 1)
        svox2_grid.sh_data.data[indices, :n_sh] = flat_all[1:1 + n_sh].T


def dense_from_svox2(svox2_grid: svox2.SparseGrid) -> torch.Tensor:
    """Extract dense (1, C, R, R, R) tensor from svox2 grid.

    C = 1 (density) + sh_data.shape[1] (SH coefficients).
    Empty voxels (links == -1) are filled with zeros.
    Fully vectorized -- no Python loop over channels.

    Returns:
        (1, C, R, R, R) dense feature grid on the same device as the grid data
    """
    R = svox2_grid.links.shape[0]
    n_sh = svox2_grid.sh_data.shape[1]
    C = 1 + n_sh
    device = svox2_grid.density_data.device

    with torch.no_grad():
        mask_flat = (svox2_grid.links.reshape(-1) >= 0)
        indices = svox2_grid.links.reshape(-1)[mask_flat].long()

        # Scatter directly into flat grid (avoids allocating combined tensor)
        grid_flat = torch.zeros(C, R ** 3, device=device)
        grid_flat[0, mask_flat] = svox2_grid.density_data[indices, 0]
        grid_flat[1:1 + n_sh, mask_flat] = svox2_grid.sh_data[indices, :n_sh].T

    return grid_flat.reshape(1, C, R, R, R)


def make_svox2_camera(pose: torch.Tensor, focal: float,
                      img_res: int, device: str = "cuda") -> svox2.Camera:
    """Convert NeRF OpenGL pose to svox2 Camera (OpenCV convention).

    NeRF Synthetic poses are in OpenGL convention (y-up, z-back).
    svox2.Camera constructs rays assuming OpenCV (y-down, z-forward).
    We convert by flipping y and z axes of the c2w matrix.

    Args:
        pose: (4, 4) camera-to-world matrix (OpenGL convention)
        focal: focal length in pixels (already scaled for img_res)
        img_res: image resolution (square images)
        device: target device
    """
    c2w = (pose.float() @ _GL_TO_CV.to(pose.device)).to(device)
    return svox2.Camera(
        c2w,
        focal, focal,
        img_res * 0.5, img_res * 0.5,
        img_res, img_res,
        ndc_coeffs=(-1.0, -1.0),
    )


def render_svox2_image(svox2_grid: svox2.SparseGrid, pose: torch.Tensor,
                       focal: float, img_res: int,
                       device: str = "cuda") -> torch.Tensor:
    """Render a full image using svox2's CUDA kernel.

    Returns:
        (H, W, 3) tensor in [0, 1]
    """
    cam = make_svox2_camera(pose, focal, img_res, device)
    with torch.no_grad():
        img = svox2_grid.volume_render_image(cam, use_kernel=True)
        img.clamp_(0.0, 1.0)
    return img


# --- Ray generation ---

def get_random_rays_svox2(dataset, batch_size: int):
    """Generate random rays matching svox2's Camera convention exactly.

    Matches svox2's cam2world_ray() in render_util.cuh:
    1. Pixel centers: (ix + 0.5 - cx) / fx  (not integer pixel coords)
    2. Normalized directions: divided by sqrt(x² + y² + 1)
    3. OpenCV c2w is equivalent to our OpenGL c2w with negated y,z dirs
    """
    H = W = dataset.resolution
    device = dataset.device

    img_idx = torch.randint(0, dataset.n_images, (batch_size,), device=device)
    pix_y = torch.randint(0, H, (batch_size,), device=device)
    pix_x = torch.randint(0, W, (batch_size,), device=device)

    rgb_gt = dataset.images[img_idx, pix_y, pix_x]

    # Camera-space directions (OpenGL: y-up, z-back)
    # +0.5 for pixel centers, matching svox2's cam2world_ray CUDA kernel
    x = (pix_x.float() + 0.5 - W * 0.5) / dataset.focal
    y = -(pix_y.float() + 0.5 - H * 0.5) / dataset.focal
    z = -torch.ones(batch_size, device=device)

    # Normalize to unit length (matching svox2's internal normalization)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    poses = dataset.poses[img_idx]
    rays_d = (dirs[:, None, :] * poses[:, :3, :3]).sum(-1)
    rays_o = poses[:, :3, 3]

    return rays_o, rays_d, rgb_gt


# --- Training ---

def train_stage_svox2(
    svox2_grid: svox2.SparseGrid,
    train_data: NerfSyntheticDataset,
    val_data: NerfSyntheticDataset,
    config: Config,
    stage_iters: int,
    lr_sigma: float = 30.0,
    lr_sigma_final: float = 0.05,
    lr_sh: float = 1e-2,
    lr_sh_final: float = 5e-4,
    lr_decay_steps: int = 0,
    global_iter_offset: int = 0,
    out_dir: str = "output",
    stage_name: str = "",
    full_res_val_data: NerfSyntheticDataset | None = None,
    tv_sh_scale: float = 1.0,
    residual_base: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Train one stage using svox2's CUDA optimizer (RMSProp).

    svox2 handles:
    - Fused volume rendering + loss computation (CUDA kernels)
    - Sparse optimizer (RMSProp) that only updates voxels hit by rays
    - In-place TV gradient computation

    Matches svox2's opt.py training recipe:
    - LR delay: density LR starts at init * delay_mult, cosine-warms to init
    - Background sparsification: removes low-density fog voxels periodically
    - TV: lambda_tv=1e-5 on density, lambda_tv_sh=1e-3 on SH (throughout training)

    Args:
        svox2_grid: SparseGrid to train
        train_data: training dataset at this stage's image resolution
        val_data: validation dataset at this stage's image resolution
        config: hyperparameters
        stage_iters: number of training iterations for this stage
        lr_sigma, lr_sigma_final: density LR schedule (exponential decay)
        lr_sh, lr_sh_final: SH LR schedule (exponential decay)
        global_iter_offset: for logging continuity across stages
        out_dir: output directory for checkpoints and images
        stage_name: descriptive name for logging
        full_res_val_data: optional 800px val data for cross-resolution comparison

    Returns:
        dict with stage metrics (best_psnr, best_ssim, etc.)
    """
    device = str(svox2_grid.density_data.device)

    # LR decay: exponential decay per-stage over lr_decay_steps.
    # Matching svox2's opt.py which decays over 153K iterations.
    # Early stopping handles actual duration; this sets the decay timescale.
    if lr_decay_steps <= 0:
        lr_decay_steps = 150000  # fallback

    def lr_fn(step, init, final, max_steps):
        if max_steps <= 1:
            return init
        t = min(step / max_steps, 1.0)
        return init * (final / max(init, 1e-10)) ** t

    # LR delay for density: cosine warm-up (matching svox2's opt.py).
    # Prevents density from oscillating wildly in early iterations.
    # During warm-up, the model learns SH colors first, then carves shape.
    # Fixed at 7500 iters (not a fraction of stage_iters) so early stopping
    # isn't confused by the warm-up dip.
    lr_sigma_delay_steps = 7500
    lr_sigma_delay_mult = 0.01  # start at 1% of lr_sigma

    def lr_delay_factor(step, delay_steps, delay_mult):
        if delay_steps <= 0 or step >= delay_steps:
            return 1.0
        return delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * step / delay_steps
        )

    # TV regularization (base values from svox2's configs).
    # tv_sh_scale allows per-stage scaling: higher at low-res stages where
    # param/data ratio is high and overfitting is the main risk.
    # Scale both density and SH TV with param/data ratio.
    tv_sigma = 1e-5 * tv_sh_scale
    tv_sh = 1e-3 * tv_sh_scale
    tv_sparsity = 0.01  # fraction of voxels for stochastic TV

    # Background sparsification via resample (same reso = sparsify only).
    sparsify_every = 5000  # fixed interval
    sparsify_start = lr_sigma_delay_steps  # after warm-up
    sparsify_thresh = 5.0  # svox2 default

    # Residual mode: freeze base, only train detail.
    # residual_base = (base_density, base_sh) — frozen copies of the upsampled grid.
    # The grid stores (base + residual). Gradients ∂L/∂data = ∂L/∂residual by chain rule.
    # TV is applied to the residual (not the full grid) to encourage sparse detail.
    use_residual = residual_base is not None
    if use_residual:
        base_density, base_sh = residual_base
        print(f"  Residual mode: base frozen, training detail only", flush=True)

    best_psnr = 0.0
    best_ssim = 0.0
    best_iter = 0
    psnr_history = []
    t0 = time.time()

    n_voxels_initial = svox2_grid.density_data.shape[0]

    # Early stopping: stop when val PSNR hasn't improved for `patience` evals.
    # Use multiple val images for a stable signal.
    patience = 20  # number of val checks without improvement before stopping
    n_val_images = min(10, val_data.n_images)  # average over multiple views
    no_improve_count = 0
    best_ckpt_path = os.path.join(out_dir, f"best_{stage_name}.npz")
    min_iters = lr_sigma_delay_steps * 3  # don't stop until well past warm-up

    for iteration in range(stage_iters):
        # LR with exponential decay + density warm-up.
        # Each stage decays independently over lr_decay_steps.
        base_lr_s = lr_fn(iteration, lr_sigma, lr_sigma_final, lr_decay_steps)
        delay = lr_delay_factor(iteration, lr_sigma_delay_steps, lr_sigma_delay_mult)
        cur_lr_s = base_lr_s * delay
        cur_lr_sh = lr_fn(iteration, lr_sh, lr_sh_final, lr_decay_steps)

        # Random ray batch (normalized directions, matching svox2 Camera convention)
        rays_o, rays_d, rgb_gt = get_random_rays_svox2(train_data, config.batch_rays)
        rays = svox2.Rays(rays_o.contiguous(), rays_d.contiguous())

        # Zero gradients (volume_render_fused accumulates)
        if svox2_grid.density_data.grad is not None:
            svox2_grid.density_data.grad.zero_()
        if svox2_grid.sh_data.grad is not None:
            svox2_grid.sh_data.grad.zero_()

        # Fused forward + backward: rendering, MSE gradient, and auxiliary losses
        # all computed in one CUDA kernel. Populates .grad directly (no autograd).
        rgb_pred = svox2_grid.volume_render_fused(
            rays, rgb_gt,
            beta_loss=0.0,
            sparsity_loss=0.0,
            randomize=True,
        )

        # MSE for logging only (gradients already computed by fused kernel)
        mse = F.mse_loss(rgb_gt, rgb_pred)

        # TV regularization: modifies gradients in-place before optimizer step.
        # In residual mode, TV is applied to the residual (data - base) to
        # encourage sparse detail. We temporarily swap in the residual,
        # compute TV grad, then swap back.
        if use_residual and (tv_sigma > 0 or tv_sh > 0):
            with torch.no_grad():
                # Save current data (base + residual)
                orig_density = svox2_grid.density_data.data.clone()
                orig_sh = svox2_grid.sh_data.data.clone()
                # Swap in residual only
                svox2_grid.density_data.data.sub_(base_density)
                svox2_grid.sh_data.data.sub_(base_sh)
            # TV on residual
            if tv_sigma > 0:
                svox2_grid.inplace_tv_grad(
                    svox2_grid.density_data.grad,
                    scaling=tv_sigma,
                    sparse_frac=tv_sparsity,
                    contiguous=True,
                )
            if tv_sh > 0:
                svox2_grid.inplace_tv_color_grad(
                    svox2_grid.sh_data.grad,
                    scaling=tv_sh,
                    sparse_frac=tv_sparsity,
                    contiguous=True,
                )
            with torch.no_grad():
                # Restore data = base + residual
                svox2_grid.density_data.data.copy_(orig_density)
                svox2_grid.sh_data.data.copy_(orig_sh)
        else:
            # Standard mode: TV on full grid
            if tv_sigma > 0:
                svox2_grid.inplace_tv_grad(
                    svox2_grid.density_data.grad,
                    scaling=tv_sigma,
                    sparse_frac=tv_sparsity,
                    contiguous=True,
                )
            if tv_sh > 0:
                svox2_grid.inplace_tv_color_grad(
                    svox2_grid.sh_data.grad,
                    scaling=tv_sh,
                    sparse_frac=tv_sparsity,
                    contiguous=True,
                )

        # Custom CUDA optimizer step (RMSProp with sparse updates)
        svox2_grid.optim_density_step(cur_lr_s, beta=0.95, optim="rmsprop")
        svox2_grid.optim_sh_step(cur_lr_sh, beta=0.95, optim="rmsprop")

        # Sparsification: skip in residual mode (resample changes grid layout,
        # invalidating the base tensor alignment).
        if (not use_residual
                and iteration >= sparsify_start
                and iteration % sparsify_every == 0
                and iteration > 0):
            n_before = svox2_grid.density_data.shape[0]
            reso = svox2_grid.links.shape[0]
            svox2_grid.resample(
                reso=reso, sigma_thresh=sparsify_thresh,
                weight_thresh=0.0, dilate=1, use_z_order=False,
            )
            n_after = svox2_grid.density_data.shape[0]
            if n_after < n_before:
                print(f"  [sparsify] {n_before:,} -> {n_after:,} voxels "
                      f"({100*n_after/n_before:.1f}% kept)")

        global_iter = global_iter_offset + iteration

        # Logging
        if (iteration + 1) % config.log_every == 0:
            elapsed = time.time() - t0
            train_psnr = -10.0 * math.log10(max(mse.item(), 1e-10))
            print(f"  [{global_iter+1:6d}] "
                  f"mse={mse.item():.6f} psnr={train_psnr:.2f} "
                  f"lr_s={cur_lr_s:.2e} lr_sh={cur_lr_sh:.2e} "
                  f"elapsed={elapsed:.0f}s")

        # Validation (average over multiple views for stable signal)
        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            val_psnrs = []
            val_ssims = []
            for vi in range(n_val_images):
                img_pred_v = render_svox2_image(
                    svox2_grid, val_data.poses[vi], val_data.focal,
                    val_data.resolution, device,
                )
                val_psnrs.append(psnr(img_pred_v, val_data.images[vi]))
                val_ssims.append(ssim(img_pred_v, val_data.images[vi]).item())

            val_psnr = float(np.mean(val_psnrs))
            val_ssim_val = float(np.mean(val_ssims))
            psnr_history.append((global_iter + 1, val_psnr))
            print(f"  -> val PSNR={val_psnr:.2f} dB, SSIM={val_ssim_val:.4f} "
                  f"(avg {n_val_images} views at {val_data.resolution}px)")

            # Also validate at full resolution if available
            if full_res_val_data is not None:
                img_full = render_svox2_image(
                    svox2_grid, full_res_val_data.poses[0],
                    full_res_val_data.focal,
                    full_res_val_data.resolution, device,
                )
                full_psnr = psnr(img_full, full_res_val_data.images[0])
                print(f"       full-res PSNR={full_psnr:.2f} dB (at 800px)")

            # Save validation image (first view)
            img_pred = render_svox2_image(
                svox2_grid, val_data.poses[0], val_data.focal,
                val_data.resolution, device,
            )
            img = (img_pred.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_{global_iter+1:06d}.png")
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_ssim = val_ssim_val
                best_iter = iteration + 1
                no_improve_count = 0
                svox2_grid.save(best_ckpt_path)
            else:
                no_improve_count += 1

            # Early stopping check (only after warm-up)
            if (iteration + 1 >= min_iters
                    and no_improve_count >= patience):
                print(f"  Early stopping at iter {iteration+1}: "
                      f"no improvement for {patience} evals "
                      f"(best={best_psnr:.2f} dB at iter {best_iter})")
                break

    elapsed = time.time() - t0
    actual_iters = iteration + 1

    print(f"  Stage done. Best PSNR={best_psnr:.2f} dB "
          f"(iter {best_iter}/{actual_iters}, {elapsed:.0f}s)\n")

    return {
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_iter": best_iter,
        "actual_iters": actual_iters,
        "psnr_history": psnr_history,
        "elapsed": elapsed,
    }


def evaluate_full_svox2(
    svox2_grid: svox2.SparseGrid,
    test_data: NerfSyntheticDataset,
    config: Config,
    out_dir: str,
    tag: str = "",
    n_views: int = 200,
) -> dict:
    """Evaluate on the full test set using svox2 rendering.

    Returns:
        dict with avg_psnr, avg_ssim, per_view results
    """
    device = str(svox2_grid.density_data.device)
    n_views = min(n_views, len(test_data))

    psnr_vals = []
    ssim_vals = []

    render_dir = os.path.join(out_dir, f"renders_{tag}")
    os.makedirs(render_dir, exist_ok=True)

    for idx in range(n_views):
        img_pred = render_svox2_image(
            svox2_grid, test_data.poses[idx], test_data.focal,
            test_data.resolution, device,
        )

        val_p = psnr(img_pred, test_data.images[idx])
        val_s = ssim(img_pred, test_data.images[idx]).item()
        psnr_vals.append(val_p)
        ssim_vals.append(val_s)

        img = (img_pred.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(
            os.path.join(render_dir, f"test_{idx:03d}.png")
        )

        if (idx + 1) % 20 == 0:
            print(f"  Rendered {idx+1}/{n_views}...")

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    print(f"  {tag}: avg PSNR={avg_psnr:.2f} dB, avg SSIM={avg_ssim:.4f} "
          f"({n_views} views at {test_data.resolution}px)")

    return {
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "psnr_per_view": psnr_vals,
        "ssim_per_view": ssim_vals,
    }


# --- Main pipeline ---

def get_stages(mode: str, config: Config):
    """Return stage schedule for the given mode."""
    if config.fm_stages is not None:
        return config.fm_stages
    return {"fm": FM_STAGES, "sp": SP_STAGES, "ss": SS_STAGES,
            "hybrid": HYBRID_STAGES}[mode]


def train_single_stage(
    config: Config,
    mode: str,
    stage_idx: int,
    resume_from: str | None = None,
):
    """Train a single stage. Saves checkpoint when done.

    Usage:
        # Stage 0: from scratch
        python train_frequency_matched.py --mode fm --stage 0

        # Stage 1: resume from stage 0 checkpoint
        python train_frequency_matched.py --mode fm --stage 1 --resume output/fm_poc/lego/fm/stage0.npz

    Args:
        config: hyperparameters
        mode: "fm", "sp", "ss", "hybrid"
        stage_idx: which stage to train (0, 1, 2, 3)
        resume_from: path to previous stage's .npz checkpoint (svox2 native)
    """
    device = config.device
    stages = get_stages(mode, config)
    total_iters = sum(s[2] for s in stages)

    if stage_idx >= len(stages):
        raise ValueError(f"Stage {stage_idx} out of range (max {len(stages)-1})")

    img_res, vol_res, stage_iters = stages[stage_idx]
    sh_dim = (config.num_channels - 1) // 3
    num_channels = config.num_channels
    use_wavelet = mode in ("fm", "hybrid")
    out_dir = os.path.join(config.output_dir, config.scene, mode)
    os.makedirs(out_dir, exist_ok=True)

    global_iter_offset = sum(s[2] for s in stages[:stage_idx])

    print(f"{'='*60}")
    print(f"Stage {stage_idx}: {img_res}px images -> {vol_res}^3 volume")
    print(f"  Mode: {mode}, Iterations: {stage_iters}")
    print(f"  SH degree {int(sh_dim**0.5)}: {num_channels} channels "
          f"({vol_res**3 * num_channels / 1e6:.1f}M params)")
    print(f"  Resume: {resume_from or 'from scratch'}")
    print(f"  Upsampling: {'wavelet' if use_wavelet else 'trilinear'}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}", flush=True)

    # Load datasets
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=img_res, white_bg=config.white_background, device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=img_res, white_bg=config.white_background, device=device,
    )
    full_res_val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=800, white_bg=config.white_background, device=device,
    )

    # --- Create or resume grid ---
    base_grid_dense = None  # for residual analysis

    if resume_from is not None:
        # Load previous stage's grid and upsample
        prev_vol_res = stages[stage_idx - 1][1] if stage_idx > 0 else vol_res

        if resume_from.endswith(".npz"):
            print(f"  Loading svox2 checkpoint: {resume_from}", flush=True)
            prev_grid = svox2.SparseGrid.load(resume_from, device=device)
            prev_vol_res = prev_grid.links.shape[0]
            print(f"  Loaded {prev_vol_res}^3 grid", flush=True)
            dense = dense_from_svox2(prev_grid)
            del prev_grid
        else:
            print(f"  Loading dense checkpoint: {resume_from}", flush=True)
            ckpt = torch.load(resume_from, map_location=device)
            dense = ckpt["merged_grid"].to(device)
            prev_vol_res = dense.shape[-1]
            del ckpt

        if prev_vol_res != vol_res:
            if use_wavelet:
                print(f"  Wavelet upsample {prev_vol_res}^3 -> {vol_res}^3...",
                      flush=True)
                base = wavelet_upsample(dense, vol_res, config.fm_wavelet)
            else:
                print(f"  Trilinear upsample {prev_vol_res}^3 -> {vol_res}^3...",
                      flush=True)
                base = F.interpolate(
                    dense, size=vol_res, mode="trilinear", align_corners=True,
                )
            base_grid_dense = base.cpu()
        else:
            # Same resolution (refinement stage)
            base = dense
            base_grid_dense = base.cpu()

        del dense
        torch.cuda.empty_cache()

        svox2_grid = create_svox2_grid(
            vol_res, config.scene_bound, sh_dim=sh_dim, device=device,
        )
        dense_to_svox2(svox2_grid, base.to(device))
        del base
        torch.cuda.empty_cache()

        # Freeze base: clone the grid data as the frozen reference.
        # The grid stores (base + residual). Initially residual = 0.
        frozen_base_density = svox2_grid.density_data.data.clone()
        frozen_base_sh = svox2_grid.sh_data.data.clone()

        n_voxels = svox2_grid.density_data.shape[0]
        print(f"  Grid ready: {vol_res}^3, {n_voxels:,} voxels", flush=True)

        # Low LRs for residual training: the residual should be small
        # refinements on the base, not a full rewrite. High LR lets the
        # residual overwrite the base signal, damaging lower-resolution quality.
        lr_s, lr_s_f = 0.5, 0.005
        lr_sh, lr_sh_f = 1e-3, 1e-4

    else:
        # Stage 0: from scratch
        svox2_grid = create_svox2_grid(
            vol_res, config.scene_bound, sh_dim=sh_dim, device=device,
        )
        svox2_grid.density_data.data[:] = 0.1
        svox2_grid.sh_data.data[:] = 0.0

        n_voxels = svox2_grid.density_data.shape[0]
        print(f"  Created fresh {vol_res}^3 grid: {n_voxels:,} voxels",
              flush=True)

        lr_s, lr_s_f = 30.0, 0.05
        lr_sh, lr_sh_f = 1e-2, 5e-4

    # --- Compute TV scale ---
    # Principle: TV on SH should scale with param/data ratio so that
    # regularization pressure is constant regardless of resolution.
    # Reference: 800px images, 520³ grid, 5% occupancy → ratio ~0.5
    # At lower res, ratio is higher → need proportionally stronger TV.
    n_data = train_data.n_images * img_res ** 2 * 3
    n_params = n_voxels * num_channels
    param_data_ratio = n_params / n_data
    ref_ratio = (520**3 * 0.05 * 28) / (100 * 800**2 * 3)
    # Use sqrt of ratio: full linear scaling over-regularizes,
    # sqrt balances overfitting vs underfitting.
    tv_sh_scale = max(1.0, (param_data_ratio / ref_ratio) ** 0.5)
    print(f"  Param/data ratio: {param_data_ratio:.1f} "
          f"(ref={ref_ratio:.2f}, TV scale={tv_sh_scale:.1f}x)", flush=True)

    # --- Train ---
    stage_name = f"stage{stage_idx}_{img_res}px_{vol_res}"
    # Pass frozen base for explicit residual training (stages > 0)
    res_base = None
    if resume_from is not None:
        res_base = (frozen_base_density, frozen_base_sh)
    metrics = train_stage_svox2(
        svox2_grid, train_data, val_data, config,
        stage_iters=stage_iters,
        lr_sigma=lr_s, lr_sigma_final=lr_s_f,
        lr_sh=lr_sh, lr_sh_final=lr_sh_f,
        lr_decay_steps=150000,  # per-stage, matching svox2's 153K
        global_iter_offset=global_iter_offset,
        out_dir=out_dir,
        stage_name=stage_name,
        full_res_val_data=full_res_val_data,
        tv_sh_scale=tv_sh_scale,
        residual_base=res_base,
    )

    # --- Reload best checkpoint if early stopped ---
    best_ckpt_path = os.path.join(out_dir, f"best_{stage_name}.npz")
    if os.path.exists(best_ckpt_path):
        best_iter = metrics.get("best_iter", 0)
        actual_iters = metrics.get("actual_iters", 0)
        if actual_iters > best_iter:
            svox2_grid = svox2.SparseGrid.load(best_ckpt_path, device=device)
            svox2_grid.opt.near_clip = 0.0
            svox2_grid.opt.background_brightness = 1.0
            print(f"  Restored best checkpoint from iter {best_iter}")

    # --- Save checkpoints ---
    # svox2 native (for resuming next stage)
    npz_path = os.path.join(out_dir, f"stage{stage_idx}.npz")
    svox2_grid.save(npz_path)
    print(f"  Saved svox2 checkpoint: {npz_path}", flush=True)

    # Dense grid + residual (for wavelet analysis)
    with torch.no_grad():
        trained_dense = dense_from_svox2(svox2_grid).cpu()

    analysis = {
        "stage": stage_idx,
        "mode": mode,
        "img_res": img_res,
        "vol_res": vol_res,
        "config": config,
        "merged_grid": trained_dense,
        "best_psnr": metrics["best_psnr"],
        "best_ssim": metrics["best_ssim"],
    }
    if base_grid_dense is not None:
        detail = trained_dense - base_grid_dense
        analysis["base_grid"] = base_grid_dense
        analysis["detail_grid"] = detail
        sparsity = (detail.abs() < 1e-3).float().mean().item()
        detail_energy = detail.abs().mean().item()
        base_energy = base_grid_dense.abs().mean().item()
        print(f"  Detail sparsity: {100*sparsity:.1f}% near zero", flush=True)
        print(f"  Detail/base energy ratio: {detail_energy/max(base_energy, 1e-10):.4f}",
              flush=True)

    pt_path = os.path.join(out_dir, f"stage{stage_idx}_analysis.pt")
    torch.save(analysis, pt_path)
    print(f"  Saved analysis checkpoint: {pt_path}", flush=True)

    # --- Full test set eval ---
    test_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "test",
        resolution=img_res, white_bg=config.white_background, device=device,
    )
    test_metrics = evaluate_full_svox2(
        svox2_grid, test_data, config, out_dir,
        tag=f"stage{stage_idx}_test_{img_res}px", n_views=50,
    )

    print(f"\n{'='*60}")
    print(f"Stage {stage_idx} complete")
    print(f"  Best val PSNR:  {metrics['best_psnr']:.2f} dB "
          f"(at {img_res}px)")
    print(f"  Test avg PSNR:  {test_metrics['avg_psnr']:.2f} dB "
          f"(at {img_res}px, 50 views)")
    print(f"  Time: {metrics['elapsed']:.0f}s")
    print(f"  Resume next stage with: --resume {npz_path}")
    print(f"{'='*60}", flush=True)

    return svox2_grid, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Frequency-matched coarse-to-fine training (svox2)"
    )
    parser.add_argument("--mode", required=True,
                        choices=["fm", "sp", "ss", "hybrid"])
    parser.add_argument("--stage", type=int, required=True,
                        help="Stage index to train (0, 1, 2, 3)")
    parser.add_argument("--resume", default=None,
                        help="Previous stage checkpoint (.npz or .pt)")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--data_dir", default="data/nerf_synthetic")
    parser.add_argument("--output_dir", default="output/fm_poc")
    parser.add_argument("--num_channels", type=int, default=28)
    parser.add_argument("--batch_rays", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=None,
                        help="Override stage iteration count")
    args = parser.parse_args()

    config = Config(
        scene=args.scene,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_channels=args.num_channels,
        batch_rays=args.batch_rays,
        device=args.device,
        val_every=args.val_every,
        log_every=args.log_every,
    )

    # Allow overriding iteration count for a single stage
    if args.iters is not None:
        stages = get_stages(args.mode, config)
        img_res, vol_res, _ = stages[args.stage]
        if args.mode == "fm":
            FM_STAGES[args.stage] = (img_res, vol_res, args.iters)
        elif args.mode == "sp":
            SP_STAGES[args.stage] = (img_res, vol_res, args.iters)
        elif args.mode == "ss":
            SS_STAGES[args.stage] = (img_res, vol_res, args.iters)
        elif args.mode == "hybrid":
            HYBRID_STAGES[args.stage] = (img_res, vol_res, args.iters)

    train_single_stage(config, args.mode, args.stage, args.resume)
