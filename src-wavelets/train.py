"""Training loop for wavelet volume rendering.

Supports three modes:
- Dense: WaveletVolume for decomp_levels <= 4 (up to 512³)
- Tiled: TiledWaveletVolume for decomp_levels > 4 (1024³+)
- Direct: DirectGridVolume (Plenoxels-style) → DWT → TiledWaveletVolume
  Train a direct voxel grid for fast convergence, then convert to wavelet
  coefficients via forward DWT. Optionally continue with tiled fine levels.
"""

import argparse
import os
import time

import torch
torch.backends.cudnn.benchmark = True
import ptwt
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from wavelet_volume import WaveletVolume, SUBBAND_KEYS
from tiled_wavelet_volume import TiledWaveletVolume
from direct_grid_volume import DirectGridVolume
from renderer import render_rays, render_image
from metrics import psnr, ssim
from occupancy import estimate_occupancy, occupancy_to_tile_mask, compute_occupancy_stats


def get_active_level(iteration: int, config: Config) -> int:
    """Determine the maximum active detail level for progressive training.

    Returns -1 if no detail level is active yet (approx only).
    """
    active = -1
    for i, start in enumerate(config.progressive_starts):
        if iteration >= start:
            active = i
    return active


def train_dense(config: Config):
    """Train with dense WaveletVolume (up to 512³)."""
    device = torch.device(config.device)

    # Data
    print(f"Loading {config.scene}...")
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    # Model
    model = WaveletVolume(
        base_resolution=config.base_resolution,
        decomp_levels=config.decomp_levels,
        num_channels=config.num_channels,
        wavelet=config.wavelet,
        scene_bound=config.scene_bound,
    ).to(device)

    total_params = model.total_params()
    param_mb = total_params * 4 / 1024**2
    print(f"Model: {config.base_resolution}³ → {config.target_resolution}³, "
          f"{total_params / 1e6:.1f}M params ({param_mb:.0f} MB fp32)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=config.lr_min,
    )

    # AMP
    amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else None
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Output
    out_dir = os.path.join(config.output_dir, config.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Training loop
    print(f"\nTraining for {config.iterations} iterations...")
    print(f"Progressive schedule: {config.progressive_starts}")
    print(f"Target: {config.target_resolution}³ × {config.num_channels}ch "
          f"({config.wavelet} wavelets)")
    if config.use_amp:
        print(f"Mixed precision: {config.amp_dtype}")
    print()

    best_psnr = 0.0
    t0 = time.time()

    for iteration in range(config.iterations):
        model.train()
        active_level = get_active_level(iteration, config)
        rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

        # Forward pass (optionally with AMP)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=config.use_amp):
            coarse_level = max(-1, active_level - 1)
            volume_fine, volume_coarse = model.reconstruct_pair(active_level, coarse_level)

            result = render_rays(
                model, rays_o, rays_d, config,
                volume_fine=volume_fine,
                volume_coarse=volume_coarse,
                perturb=True,
            )

            l1_loss = torch.nn.functional.l1_loss(result["rgb"], rgb_gt)
            l1_coarse = torch.nn.functional.l1_loss(result["rgb_coarse"], rgb_gt)
            sparse_loss = result["density_fine"].mean()
            loss = l1_loss + 0.1 * l1_coarse + config.lambda_sparse * sparse_loss

        # Backprop
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Logging
        if iteration % config.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            if active_level >= 0:
                res = config.base_resolution * (2 ** (active_level + 1))
            else:
                res = config.base_resolution
            print(f"[{iteration:6d}/{config.iterations}] "
                  f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"lr={lr:.2e} level={active_level} ({res}³) "
                  f"elapsed={elapsed:.0f}s")

        # Validation
        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            model.eval()
            val_idx = 0
            with torch.no_grad():
                result = render_image(
                    model, val_data.poses[val_idx],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config,
                )

            val_psnr = psnr(result["rgb"], val_data.images[val_idx])
            val_ssim = ssim(result["rgb"], val_data.images[val_idx]).item()
            print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_{iteration + 1:06d}.png")
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "psnr": val_psnr,
                }, os.path.join(out_dir, "best.pt"))
                print(f"  → saved best model (PSNR={best_psnr:.2f})")

        # Save checkpoint
        if (iteration + 1) % config.save_every == 0:
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, os.path.join(out_dir, f"ckpt_{iteration + 1:06d}.pt"))

    # Final save
    torch.save({
        "iteration": config.iterations,
        "model_state_dict": model.state_dict(),
        "config": config,
        "psnr": best_psnr,
    }, os.path.join(out_dir, "final.pt"))

    total_time = time.time() - t0
    print(f"\nDone. Best PSNR={best_psnr:.2f} dB. Time={total_time / 60:.1f} min")
    print(f"Model size: {model.total_params() / 1e6:.1f}M params, "
          f"{model.total_params() * 4 / 1024**2:.0f} MB (fp32)")

    return model


def train_tiled(config: Config):
    """Train with TiledWaveletVolume (1024³+).

    Uses the same TiledWaveletVolume throughout to avoid coefficient size
    mismatches between models (bior4.4 produces different intermediate sizes
    when decomposing from different starting resolutions).

    Phase 1: Train base levels (dense) with empty sparse levels for occupancy.
    Phase 2: Allocate sparse fine levels from occupancy, train progressively.
    """
    device = torch.device(config.device)

    # Data
    print(f"Loading {config.scene}...")
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    out_dir = os.path.join(config.output_dir, config.scene)
    os.makedirs(out_dir, exist_ok=True)

    # --- Create model (used for both phases) ---
    # Phase 1 starts with empty sparse levels (all-unoccupied).
    empty_masks = {}
    for i in range(config.base_level + 1, config.decomp_levels):
        empty_masks[i] = torch.zeros(1, 1, 1, dtype=torch.bool)

    model = TiledWaveletVolume(
        base_resolution=config.base_resolution,
        decomp_levels=config.decomp_levels,
        num_channels=config.num_channels,
        channels_per_level=config.channels_per_level,
        wavelet=config.wavelet,
        scene_bound=config.scene_bound,
        tile_size=config.tile_size,
        base_level=config.base_level,
        occupancy_masks=empty_masks,
    ).to(device)

    base_res = model.level_sizes[config.base_level + 2]  # resolution after base_level IDWT
    base_levels = config.base_level + 1

    # Progressive schedule for base levels
    base_starts = [
        int(i * config.coarse_pretrain_iters / (base_levels + 1))
        for i in range(base_levels)
    ]

    total_params = model.total_params()
    param_mb = total_params * 4 / 1024**2
    print(f"=== Phase 1: Base pretrain (~{base_res}³, "
          f"{config.coarse_pretrain_iters} iters) ===")
    print(f"Model: {config.base_resolution}³ → {config.target_resolution}³, "
          f"{total_params / 1e6:.1f}M params ({param_mb:.0f} MB fp32)")
    print(f"Progressive schedule: {base_starts}\n")

    # --- Phase 1: Train base levels ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.coarse_pretrain_iters, eta_min=config.lr_min,
    )

    best_psnr = 0.0
    t0 = time.time()

    for iteration in range(config.coarse_pretrain_iters):
        model.train()

        # Progressive base level activation
        active_level = -1
        for i, start in enumerate(base_starts):
            if iteration >= start:
                active_level = i

        rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

        # Reconstruct base volume at current progressive level
        coarse_level = max(-1, active_level - 1)
        volume_fine = model.reconstruct_base(max_level=active_level)
        volume_coarse = model.reconstruct_base(max_level=coarse_level)

        result = render_rays(
            model, rays_o, rays_d, config,
            volume_fine=volume_fine,
            volume_coarse=volume_coarse,
            perturb=True,
        )

        l1_loss = torch.nn.functional.l1_loss(result["rgb"], rgb_gt)
        l1_coarse = torch.nn.functional.l1_loss(result["rgb_coarse"], rgb_gt)
        sparse_loss = result["density_fine"].mean()
        loss = l1_loss + 0.1 * l1_coarse + config.lambda_sparse * sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if iteration % config.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            if active_level >= 0:
                res = model.level_sizes[active_level + 2]
            else:
                res = model.level_sizes[0]
            print(f"[{iteration:6d}/{config.coarse_pretrain_iters}] "
                  f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"lr={lr:.2e} level={active_level} (~{res}³) "
                  f"elapsed={elapsed:.0f}s")

        # Validation
        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            model.eval()
            val_idx = 0
            with torch.no_grad():
                result = render_image(
                    model, val_data.poses[val_idx],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config,
                    max_level=active_level,
                )

            val_psnr = psnr(result["rgb"], val_data.images[val_idx])
            val_ssim = ssim(result["rgb"], val_data.images[val_idx]).item()
            print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_{iteration + 1:06d}.png")
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "psnr": val_psnr,
                }, os.path.join(out_dir, "best.pt"))
                print(f"  → saved best model (PSNR={best_psnr:.2f})")

        # Save checkpoint
        if (iteration + 1) % config.save_every == 0:
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, os.path.join(out_dir, f"ckpt_{iteration + 1:06d}.pt"))

    phase1_time = time.time() - t0
    print(f"\nPhase 1 done. Best PSNR={best_psnr:.2f} dB. Time={phase1_time / 60:.1f} min")

    # --- Estimate occupancy from trained base levels ---
    print("\n=== Estimating spatial occupancy ===")
    model.eval()
    occupancy = estimate_occupancy(
        model,
        grid_resolution=config.occupancy_grid_resolution,
        density_threshold=config.occupancy_threshold,
        dilate_kernel=config.occupancy_dilate,
    )
    occ_pct = 100.0 * occupancy.float().mean().item()
    print(f"Occupancy: {occ_pct:.1f}% of volume is occupied")

    # Print occupancy stats
    stats = compute_occupancy_stats(
        occupancy, model.level_sizes[1:], config.tile_size, config.num_channels,
    )
    for s in stats:
        print(f"  Level {s['level']}: {s['detail_size']}³, "
              f"{s['occupancy_pct']:.1f}% occupied, "
              f"{s['sparse_params_M']:.0f}M params (vs {s['dense_params_M']:.0f}M dense)")

    # Build occupancy masks and replace sparse levels
    occupancy_masks = {}
    for i in range(config.base_level + 1, config.decomp_levels):
        D = model.level_sizes[i + 1]
        tiles_per_axis = max(1, (D + config.tile_size - 1) // config.tile_size)
        occupancy_masks[i] = occupancy_to_tile_mask(occupancy, tiles_per_axis)

    model.set_sparse_levels(occupancy_masks)
    model = model.to(device)
    torch.cuda.empty_cache()

    # Print memory summary
    mem = model.memory_summary()
    print(f"\nTiled model: {mem['total_params_M']:.0f}M params, "
          f"{mem['total_mb']:.0f} MB")
    print(f"  Approx: {mem['approx_mb']:.0f} MB")
    print(f"  Dense details: {mem['dense_mb']:.0f} MB")
    print(f"  Sparse details: {mem['sparse_mb']:.0f} MB")
    for sl in mem['sparse_levels']:
        print(f"    Level {sl['level']}: {sl['occupied']}/{sl['total']} tiles, "
              f"{sl['memory_mb']:.0f} MB")

    # --- Phase 2: Tiled training with sparse fine levels ---
    print(f"\n=== Phase 2: Tiled training ({config.target_resolution}³) ===\n")

    # New optimizer includes sparse level parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    remaining_iters = config.iterations - config.coarse_pretrain_iters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_iters, eta_min=config.lr_min,
    )

    # Progressive schedule for fine levels only
    fine_levels = config.decomp_levels - config.base_level - 1
    fine_starts = [
        int(i * remaining_iters / (fine_levels + 1))
        for i in range(fine_levels)
    ]
    print(f"Fine level activation schedule: {fine_starts}")
    print(f"Training for {remaining_iters} iterations...\n")

    best_psnr = 0.0
    t0 = time.time()

    for iteration in range(remaining_iters):
        model.train()

        # Progressive fine level activation: freeze inactive sparse levels
        for lvl_idx, sparse_level in enumerate(model.sparse_details):
            active = iteration >= fine_starts[lvl_idx]
            for p in sparse_level.parameters():
                p.requires_grad_(active)

        rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

        # Reconstruct base volume (shared across all ray queries)
        base_volume = model.reconstruct_base()

        result = render_rays(
            model, rays_o, rays_d, config,
            base_volume=base_volume,
            perturb=True,
        )

        l1_loss = torch.nn.functional.l1_loss(result["rgb"], rgb_gt)
        l1_coarse = torch.nn.functional.l1_loss(result["rgb_coarse"], rgb_gt)
        sparse_loss = result["density_fine"].mean()
        loss = l1_loss + 0.1 * l1_coarse + config.lambda_sparse * sparse_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        global_iter = config.coarse_pretrain_iters + iteration
        if iteration % config.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{global_iter:6d}/{config.iterations}] "
                  f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"lr={lr:.2e} elapsed={elapsed:.0f}s")

        # Validation
        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            model.eval()
            val_idx = 0
            with torch.no_grad():
                result = render_image(
                    model, val_data.poses[val_idx],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config,
                )

            val_psnr = psnr(result["rgb"], val_data.images[val_idx])
            val_ssim = ssim(result["rgb"], val_data.images[val_idx]).item()
            print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_{global_iter + 1:06d}.png")
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "iteration": global_iter,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "psnr": val_psnr,
                }, os.path.join(out_dir, "best.pt"))
                print(f"  → saved best model (PSNR={best_psnr:.2f})")

        # Save checkpoint
        if (iteration + 1) % config.save_every == 0:
            torch.save({
                "iteration": global_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, os.path.join(out_dir, f"ckpt_{global_iter + 1:06d}.pt"))

    # Final save
    torch.save({
        "iteration": config.iterations,
        "model_state_dict": model.state_dict(),
        "config": config,
        "psnr": best_psnr,
    }, os.path.join(out_dir, "final.pt"))

    total_time = time.time() - t0
    print(f"\nDone. Best PSNR={best_psnr:.2f} dB. Time={total_time / 60:.1f} min")

    mem = model.memory_summary()
    print(f"Model size: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")


def _plenoxels_lr(lr_init, lr_final, step, max_steps, delay_steps=0, delay_mult=0.01):
    """Plenoxels exponential LR decay with optional warmup delay."""
    if delay_steps > 0 and step < delay_steps:
        delay_rate = delay_mult + (1 - delay_mult) * (step / delay_steps)
    else:
        delay_rate = 1.0
    t = min(step / max(max_steps, 1), 1.0)
    log_lerp = lr_init * (lr_final / lr_init) ** t
    return delay_rate * log_lerp


def train_direct(config: Config):
    """Train a direct voxel grid with coarse-to-fine upsampling.

    Plenoxels-faithful optimization:
    - ReLU density activation, init sigma=0.1
    - RMSProp with separate density/SH LR groups
    - Density LR 30→0.05 with 15K warmup, SH LR 0.01→5e-6
    - TV regularization early only (density 1e-5, SH 1e-3)
    - Weight-based voxel pruning
    """
    device = torch.device(config.device)

    # Data
    print(f"Loading {config.scene}...")
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    out_dir = os.path.join(config.output_dir, config.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Create model at initial resolution
    initial_res = config.direct_res_schedule[0][0]
    model = DirectGridVolume(
        resolution=initial_res,
        num_channels=config.num_channels,
        scene_bound=config.scene_bound,
    ).to(device)

    total_stage_iters = sum(iters for _, iters in config.direct_res_schedule)

    # Plenoxels LR config
    lr_sigma_init = 3e1
    lr_sigma_final = 5e-2
    lr_sh_init = 1e-2
    lr_sh_final = 5e-6
    lr_sigma_delay_steps = 15000
    lr_sigma_delay_mult = 0.01

    # TV config (Plenoxels synthetic: early only)
    tv_sigma_weight = 1e-5
    tv_sh_weight = 1e-3
    tv_early_only_iters = int(total_stage_iters * 0.32)  # ~first 1/3

    # Pruning config
    prune_threshold = 0.01
    prune_every = 5000

    print(f"\n=== Direct Grid Training (Plenoxels-style) ===")
    print(f"Schedule: {config.direct_res_schedule}")
    print(f"Total iterations: {total_stage_iters}")
    print(f"LR sigma: {lr_sigma_init}→{lr_sigma_final} (delay {lr_sigma_delay_steps})")
    print(f"LR SH: {lr_sh_init}→{lr_sh_final}")
    print(f"TV early only: {tv_early_only_iters} iters\n")

    best_psnr = 0.0
    global_iter = 0
    t0 = time.time()

    for stage_idx, (res, stage_iters) in enumerate(config.direct_res_schedule):
        # Upsample if not first stage
        if stage_idx > 0:
            print(f"\n--- Upsampling {model.resolution}³ → {res}³ ---")
            model.upsample(res)
            model = model.to(device)
            torch.cuda.empty_cache()

        param_mb = model.total_params() * 4 / 1024**2
        print(f"Stage {stage_idx}: {res}³, {stage_iters} iters, "
              f"{model.total_params()/1e6:.1f}M params ({param_mb:.0f} MB fp32)")

        # RMSProp with separate param groups (Plenoxels uses RMSProp beta=0.95)
        optimizer = torch.optim.RMSprop(
            [
                {"params": [model.density_grid], "lr": lr_sigma_init},
                {"params": [model.sh_grid], "lr": lr_sh_init},
            ],
            alpha=0.95,  # beta / smoothing constant
            eps=1e-8,
        )

        for iteration in range(stage_iters):
            # Manual LR scheduling (Plenoxels exponential decay)
            current_lr_sigma = _plenoxels_lr(
                lr_sigma_init, lr_sigma_final,
                global_iter, total_stage_iters,
                delay_steps=lr_sigma_delay_steps,
                delay_mult=lr_sigma_delay_mult,
            )
            current_lr_sh = _plenoxels_lr(
                lr_sh_init, lr_sh_final,
                global_iter, total_stage_iters,
            )
            optimizer.param_groups[0]["lr"] = current_lr_sigma
            optimizer.param_groups[1]["lr"] = current_lr_sh

            model.train()
            rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

            result = render_rays(
                model, rays_o, rays_d, config,
                perturb=True,
            )

            l1_loss = torch.nn.functional.l1_loss(result["rgb"], rgb_gt)
            l1_coarse = torch.nn.functional.l1_loss(result["rgb_coarse"], rgb_gt)

            loss = l1_loss + 0.1 * l1_coarse

            # TV regularization (early only, matching Plenoxels synthetic config)
            if global_iter < tv_early_only_iters:
                tv_sigma = model.tv_loss()
                tv_sh = model.tv_loss_sh()
                loss = loss + tv_sigma_weight * tv_sigma + tv_sh_weight * tv_sh

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iter += 1

            # Logging
            if global_iter % config.log_every == 0:
                elapsed = time.time() - t0
                print(f"[{global_iter:6d}/{total_stage_iters}] "
                      f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                      f"lr_s={current_lr_sigma:.2e} lr_sh={current_lr_sh:.2e} "
                      f"res={res}³ elapsed={elapsed:.0f}s")

            # Pruning: zero out voxels with low rendering weight
            # (simplified: prune by density threshold since we don't
            # track per-voxel max weight across rays)
            if global_iter % prune_every == 0 and global_iter > lr_sigma_delay_steps:
                with torch.no_grad():
                    density_vals = model.density_grid.data[0, 0]
                    low_density = density_vals < prune_threshold
                    n_pruned = low_density.sum().item()
                    n_total = density_vals.numel()
                    model.density_grid.data[0, 0][low_density] = 0.0
                    print(f"  Pruned {n_pruned}/{n_total} voxels "
                          f"({100*n_pruned/n_total:.1f}%)")

            # Validation
            if global_iter % config.val_every == 0 or global_iter == 1:
                model.eval()
                with torch.no_grad():
                    result = render_image(
                        model, val_data.poses[0],
                        config.train_resolution, config.train_resolution,
                        val_data.focal, config,
                    )

                val_psnr = psnr(result["rgb"], val_data.images[0])
                val_ssim = ssim(result["rgb"], val_data.images[0]).item()
                print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

                img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                Image.fromarray(img).save(
                    os.path.join(out_dir, f"val_{global_iter:06d}.png")
                )

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save({
                        "iteration": global_iter,
                        "model_state_dict": model.state_dict(),
                        "resolution": model.resolution,
                        "config": config,
                        "psnr": val_psnr,
                    }, os.path.join(out_dir, "best_direct.pt"))
                    print(f"  → saved best model (PSNR={best_psnr:.2f})")

            # Checkpoint (model weights only — no optimizer state to save disk)
            if global_iter % config.save_every == 0:
                ckpt_path = os.path.join(out_dir, "direct_ckpt_latest.pt")
                torch.save({
                    "iteration": global_iter,
                    "model_state_dict": model.state_dict(),
                    "resolution": model.resolution,
                    "config": config,
                }, ckpt_path)

    # Final save
    torch.save({
        "iteration": global_iter,
        "model_state_dict": model.state_dict(),
        "resolution": model.resolution,
        "config": config,
        "psnr": best_psnr,
    }, os.path.join(out_dir, "direct_final.pt"))

    total_time = time.time() - t0
    print(f"\nDirect training done. Best PSNR={best_psnr:.2f} dB. "
          f"Time={total_time / 60:.1f} min")

    return model


def convert_direct_to_wavelet(
    direct_model: DirectGridVolume,
    config: Config,
) -> TiledWaveletVolume:
    """Convert a trained direct grid to TiledWaveletVolume via forward DWT.

    The DWT is lossless — the wavelet coefficients exactly represent the
    trained grid. Verified via round-trip reconstruction.
    """
    device = next(direct_model.parameters()).device

    # Create TiledWaveletVolume with empty sparse levels
    empty_masks = {}
    for i in range(config.base_level + 1, config.decomp_levels):
        empty_masks[i] = torch.zeros(1, 1, 1, dtype=torch.bool)

    tiled_model = TiledWaveletVolume(
        base_resolution=config.base_resolution,
        decomp_levels=config.decomp_levels,
        num_channels=config.num_channels,
        channels_per_level=config.channels_per_level,
        wavelet=config.wavelet,
        scene_bound=config.scene_bound,
        tile_size=config.tile_size,
        base_level=config.base_level,
        occupancy_masks=empty_masks,
    ).to(device)

    # Verify direct grid resolution matches expected base volume resolution
    base_vol_res = tiled_model.level_sizes[config.base_level + 2]
    assert direct_model.resolution == base_vol_res, (
        f"Direct grid resolution {direct_model.resolution} != expected "
        f"base volume resolution {base_vol_res}. The direct grid must be "
        f"trained to exactly {base_vol_res}³ for lossless DWT conversion."
    )

    grid = direct_model.grid.data  # (1, C, R, R, R)
    dwt_levels = config.base_level + 1

    print(f"Running forward DWT ({dwt_levels} levels, {config.wavelet})...")
    with torch.no_grad():
        coeffs = ptwt.wavedec3(grid, wavelet=config.wavelet, level=dwt_levels)

        # Verify and copy coefficients
        approx = coeffs[0]
        print(f"  Approx: {approx.shape[2]}³ "
              f"(expected {tiled_model.approx.shape[2]}³)")
        assert approx.shape == tiled_model.approx.shape, (
            f"Approx shape mismatch: DWT={approx.shape} vs "
            f"model={tiled_model.approx.shape}"
        )
        C_approx = tiled_model.approx.shape[1]
        tiled_model.approx.data.copy_(approx[:, :C_approx])

        for i in range(dwt_levels):
            detail_dict = coeffs[i + 1]
            detail_size = next(iter(detail_dict.values())).shape[2]
            expected_size = tiled_model.dense_details[i].shape[2]
            print(f"  Detail {i}: {detail_size}³ (expected {expected_size}³)")
            assert detail_size == expected_size, (
                f"Detail {i} size mismatch: DWT={detail_size} vs "
                f"model={expected_size}"
            )

            stacked = torch.stack(
                [detail_dict[key].squeeze(0) for key in SUBBAND_KEYS]
            )  # (7, C, D, D, D)
            C_model = tiled_model.dense_details[i].shape[1]
            tiled_model.dense_details[i].data.copy_(stacked[:, :C_model])

        # Verify round-trip reconstruction
        tiled_vol = tiled_model.reconstruct_base()
        max_error = (tiled_vol - grid).abs().max().item()
        mean_error = (tiled_vol - grid).abs().mean().item()
        print(f"  Round-trip max error: {max_error:.2e}, mean: {mean_error:.2e}")
        assert max_error < 1e-3, (
            f"DWT round-trip error too large: {max_error:.2e}"
        )

    print("DWT conversion successful.")
    return tiled_model


def train_direct_then_tiled(config: Config):
    """Full pipeline: direct grid → DWT → tiled fine levels.

    Stage 1: Train a direct voxel grid (Plenoxels-style) at the base
             volume resolution for fast convergence.
    Stage 2: Convert the converged grid to wavelet coefficients via
             forward DWT (lossless).
    Stage 3: Estimate occupancy, allocate sparse fine levels, freeze
             the base, and train fine details through tiled IDWT.
    """
    device = torch.device(config.device)
    from tiled_wavelet_volume import _get_level_sizes

    # Compute the correct base volume resolution from wavelet decomposition
    target_res = config.base_resolution * (2 ** config.decomp_levels)
    level_sizes = _get_level_sizes(target_res, config.decomp_levels, config.wavelet)
    base_vol_res = level_sizes[config.base_level + 2]

    # Adjust direct_res_schedule: replace final resolution with exact value
    # to ensure DWT compatibility (bior4.4 produces non-power-of-2 sizes)
    schedule = list(config.direct_res_schedule)
    schedule[-1] = (base_vol_res, schedule[-1][1])
    # Remove any intermediate stages that now exceed base_vol_res
    schedule = [(r, i) for r, i in schedule if r <= base_vol_res]
    config.direct_res_schedule = schedule

    print(f"Target: {target_res}³ ({config.decomp_levels} decomp levels)")
    print(f"Base volume resolution: {base_vol_res}³ "
          f"(base_level={config.base_level})")
    print(f"Direct grid schedule: {schedule}")
    print(f"Level sizes: {level_sizes}\n")

    # --- Stage 1: Direct grid training ---
    direct_model = train_direct(config)

    # --- Stage 2: DWT conversion ---
    print(f"\n=== Stage 2: DWT Conversion ===")
    tiled_model = convert_direct_to_wavelet(direct_model, config)
    del direct_model
    torch.cuda.empty_cache()

    # --- Stage 3: Fine-level training ---
    print(f"\n=== Stage 3: Fine-Level Training ===")

    print(f"Loading {config.scene}...")
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    out_dir = os.path.join(config.output_dir, config.scene)

    # Estimate occupancy from the converted base volume
    print("\nEstimating spatial occupancy...")
    tiled_model.eval()
    occupancy = estimate_occupancy(
        tiled_model,
        grid_resolution=config.occupancy_grid_resolution,
        density_threshold=config.occupancy_threshold,
        dilate_kernel=config.occupancy_dilate,
    )
    occ_pct = 100.0 * occupancy.float().mean().item()
    print(f"Occupancy: {occ_pct:.1f}% of volume is occupied")

    stats = compute_occupancy_stats(
        occupancy, tiled_model.level_sizes[1:],
        config.tile_size, config.num_channels,
    )
    for s in stats:
        print(f"  Level {s['level']}: {s['detail_size']}³, "
              f"{s['occupancy_pct']:.1f}% occupied, "
              f"{s['sparse_params_M']:.0f}M params "
              f"(vs {s['dense_params_M']:.0f}M dense)")

    # Build occupancy masks and allocate sparse levels
    occupancy_masks = {}
    for i in range(config.base_level + 1, config.decomp_levels):
        D = tiled_model.level_sizes[i + 1]
        tiles_per_axis = max(1, (D + config.tile_size - 1) // config.tile_size)
        occupancy_masks[i] = occupancy_to_tile_mask(occupancy, tiles_per_axis)

    tiled_model.set_sparse_levels(occupancy_masks)
    tiled_model = tiled_model.to(device)
    torch.cuda.empty_cache()

    mem = tiled_model.memory_summary()
    print(f"\nTiled model: {mem['total_params_M']:.0f}M params, "
          f"{mem['total_mb']:.0f} MB")
    print(f"  Approx: {mem['approx_mb']:.0f} MB")
    print(f"  Dense details: {mem['dense_mb']:.0f} MB")
    print(f"  Sparse details: {mem['sparse_mb']:.0f} MB")
    for sl in mem['sparse_levels']:
        print(f"    Level {sl['level']}: {sl['occupied']}/{sl['total']} tiles, "
              f"{sl['memory_mb']:.0f} MB")

    # Freeze base levels — they're already converged from direct grid
    tiled_model.approx.requires_grad_(False)
    for d in tiled_model.dense_details:
        d.requires_grad_(False)
    frozen_params = (tiled_model.approx.numel()
                     + sum(d.numel() for d in tiled_model.dense_details))
    trainable_params = sum(
        p.numel() for p in tiled_model.parameters() if p.requires_grad
    )
    print(f"\nFrozen: {frozen_params/1e6:.0f}M params (base levels)")
    print(f"Trainable: {trainable_params/1e6:.0f}M params (sparse fine levels)")

    # Fine-level training
    fine_levels = config.decomp_levels - config.base_level - 1
    fine_iters = config.iterations

    # Only optimize trainable parameters (sparse fine levels)
    trainable = [p for p in tiled_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fine_iters, eta_min=config.lr_min,
    )

    # Progressive activation schedule for fine levels
    fine_starts = [
        int(i * fine_iters / (fine_levels + 1))
        for i in range(fine_levels)
    ]
    print(f"\nFine level activation schedule: {fine_starts}")
    print(f"Training for {fine_iters} iterations...\n")

    best_psnr = 0.0
    t0 = time.time()

    for iteration in range(fine_iters):
        tiled_model.train()

        # Progressive fine level activation
        for lvl_idx, sparse_level in enumerate(tiled_model.sparse_details):
            active = iteration >= fine_starts[lvl_idx]
            for p in sparse_level.parameters():
                p.requires_grad_(active)

        rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

        # Reconstruct base volume (frozen, no grad needed)
        base_volume = tiled_model.reconstruct_base()

        result = render_rays(
            tiled_model, rays_o, rays_d, config,
            base_volume=base_volume,
            perturb=True,
        )

        l1_loss = torch.nn.functional.l1_loss(result["rgb"], rgb_gt)
        l1_coarse = torch.nn.functional.l1_loss(result["rgb_coarse"], rgb_gt)
        sparse_loss = result["density_fine"].mean()
        loss = l1_loss + 0.1 * l1_coarse + config.lambda_sparse * sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if (iteration + 1) % config.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{iteration + 1:6d}/{fine_iters}] "
                  f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"lr={lr:.2e} elapsed={elapsed:.0f}s")

        # Validation
        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            tiled_model.eval()
            with torch.no_grad():
                result = render_image(
                    tiled_model, val_data.poses[0],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config,
                )

            val_psnr = psnr(result["rgb"], val_data.images[0])
            val_ssim = ssim(result["rgb"], val_data.images[0]).item()
            print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_fine_{iteration + 1:06d}.png")
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": tiled_model.state_dict(),
                    "config": config,
                    "psnr": val_psnr,
                }, os.path.join(out_dir, "best.pt"))
                print(f"  → saved best model (PSNR={best_psnr:.2f})")

        # Checkpoint
        if (iteration + 1) % config.save_every == 0:
            torch.save({
                "iteration": iteration,
                "model_state_dict": tiled_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, os.path.join(out_dir, f"ckpt_fine_{iteration + 1:06d}.pt"))

    # Final save
    torch.save({
        "iteration": fine_iters,
        "model_state_dict": tiled_model.state_dict(),
        "config": config,
        "psnr": best_psnr,
    }, os.path.join(out_dir, "final.pt"))

    total_time = time.time() - t0
    print(f"\nFine-level training done. Best PSNR={best_psnr:.2f} dB. "
          f"Time={total_time / 60:.1f} min")
    mem = tiled_model.memory_summary()
    print(f"Model size: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")


def train(config: Config):
    """Route to appropriate training function based on config."""
    if config.training_mode == "direct":
        train_direct(config)
    elif config.training_mode == "direct_then_tiled":
        train_direct_then_tiled(config)
    elif config.decomp_levels <= 4:
        train_dense(config)
    else:
        train_tiled(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--data_dir", default="data/nerf_synthetic")
    parser.add_argument("--decomp_levels", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_rays", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wavelet", default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--base_level", type=int, default=None)
    parser.add_argument("--channels_per_level", type=str, default=None,
                        help="Comma-separated channel counts, e.g. '28,28,28,16,8,4'")
    parser.add_argument("--mode", default=None,
                        choices=["wavelet", "direct", "direct_then_tiled"],
                        help="Training mode")
    parser.add_argument("--direct_lr", type=float, default=None)
    parser.add_argument("--lambda_tv", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    args = parser.parse_args()

    config = Config(scene=args.scene, data_dir=args.data_dir)
    if args.decomp_levels is not None:
        config.decomp_levels = args.decomp_levels
        config.progressive_starts = []
        config.__post_init__()
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.batch_rays is not None:
        config.batch_rays = args.batch_rays
    if args.lr is not None:
        config.lr = args.lr
    if args.wavelet is not None:
        config.wavelet = args.wavelet
    if args.use_amp:
        config.use_amp = True
    if args.tile_size is not None:
        config.tile_size = args.tile_size
    if args.base_level is not None:
        config.base_level = args.base_level
    if args.channels_per_level is not None:
        config.channels_per_level = [int(x) for x in args.channels_per_level.split(",")]
    if args.mode is not None:
        config.training_mode = args.mode
    if args.direct_lr is not None:
        config.direct_lr = args.direct_lr
    if args.lambda_tv is not None:
        config.lambda_tv = args.lambda_tv
    if args.lambda_sparse is not None:
        config.lambda_sparse = args.lambda_sparse

    train(config)
