"""Plenoxels-faithful direct grid training.

Strictly follows svox2 (Plenoxels) hyperparameters for NeRF Synthetic:
- RMSProp (beta=0.95), continuous across upsample
- MSE loss
- Density LR 30→0.05 with 15K warmup, SH LR 0.01→5e-6
- TV early only (first 38,400 steps): density 1e-5, SH 1e-3
- Density-threshold pruning
- Resolution: 256³ → 518³ (518 = exact bior4.4 DWT base volume size)
"""

import os
import time

import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from direct_grid_volume import DirectGridVolume
from metrics import psnr, ssim
from renderer import render_rays, render_image


# --- Plenoxels hyperparameters (NeRF Synthetic) ---
LR_SIGMA_INIT = 3e1
LR_SIGMA_FINAL = 5e-2
LR_SH_INIT = 1e-2
LR_SH_FINAL = 5e-6
LR_SIGMA_DELAY_STEPS = 15000
LR_SIGMA_DELAY_MULT = 0.01

TV_SIGMA_WEIGHT = 1e-5
TV_SH_WEIGHT = 1e-3
TV_EARLY_ONLY_STEPS = 38400

UPSAMPLE_STEP = 38400
PRUNE_THRESHOLD = 0.01
PRUNE_EVERY = 5000

RES_INITIAL = 256
RES_FINAL = 518  # bior4.4 DWT-compatible for base_level=3, decomp=6
TOTAL_ITERS = 250000
BATCH_RAYS = 8192


def plenoxels_lr(lr_init, lr_final, step, max_steps,
                 delay_steps=0, delay_mult=0.01):
    """Plenoxels exponential LR decay with optional warmup."""
    if delay_steps > 0 and step < delay_steps:
        delay_rate = delay_mult + (1 - delay_mult) * (step / delay_steps)
    else:
        delay_rate = 1.0
    t = min(step / max(max_steps, 1), 1.0)
    log_lerp = lr_init * (lr_final / lr_init) ** t
    return delay_rate * log_lerp


def main():
    config = Config(training_mode='direct')
    device = torch.device('cuda')

    # Data
    print(f'Loading {config.scene}...')
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, 'train',
        resolution=config.train_resolution,
        white_bg=config.white_background, device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, 'val',
        resolution=config.train_resolution,
        white_bg=config.white_background, device=device,
    )
    out_dir = os.path.join(config.output_dir, config.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Model
    model = DirectGridVolume(
        resolution=RES_INITIAL,
        num_channels=config.num_channels,
        scene_bound=config.scene_bound,
    ).to(device)

    print(f'\n=== Plenoxels-Faithful Training ===')
    print(f'  {RES_INITIAL}³ → {RES_FINAL}³ (upsample at {UPSAMPLE_STEP})')
    print(f'  {TOTAL_ITERS} iterations, MSE loss')
    print(f'  LR sigma: {LR_SIGMA_INIT}→{LR_SIGMA_FINAL} '
          f'(delay {LR_SIGMA_DELAY_STEPS})')
    print(f'  LR SH: {LR_SH_INIT}→{LR_SH_FINAL}')
    print(f'  TV early only: {TV_EARLY_ONLY_STEPS} steps')
    print(f'  Params: {model.total_params()/1e6:.1f}M '
          f'({model.total_params()*4/1024**2:.0f} MB)\n')

    # Single continuous optimizer (Plenoxels does NOT restart at upsample)
    optimizer = torch.optim.RMSprop(
        [
            {'params': [model.density_grid], 'lr': LR_SIGMA_INIT},
            {'params': [model.sh_grid], 'lr': LR_SH_INIT},
        ],
        alpha=0.95,
        eps=1e-8,
    )

    best_psnr = 0.0
    t0 = time.time()
    upsampled = False

    for step in range(1, TOTAL_ITERS + 1):
        # Upsample once
        if step == UPSAMPLE_STEP and not upsampled:
            print(f'\n--- Upsampling {model.resolution}³ → {RES_FINAL}³ ---')
            model.upsample(RES_FINAL)
            model = model.to(device)
            torch.cuda.empty_cache()
            # Recreate optimizer for new parameter tensors
            # (can't keep RMSProp state across tensor resize)
            optimizer = torch.optim.RMSprop(
                [
                    {'params': [model.density_grid], 'lr': LR_SIGMA_INIT},
                    {'params': [model.sh_grid], 'lr': LR_SH_INIT},
                ],
                alpha=0.95,
                eps=1e-8,
            )
            upsampled = True
            print(f'  Params: {model.total_params()/1e6:.1f}M '
                  f'({model.total_params()*4/1024**2:.0f} MB)\n')

        # LR schedule (continuous across upsample)
        lr_sigma = plenoxels_lr(
            LR_SIGMA_INIT, LR_SIGMA_FINAL, step, TOTAL_ITERS,
            LR_SIGMA_DELAY_STEPS, LR_SIGMA_DELAY_MULT)
        lr_sh = plenoxels_lr(
            LR_SH_INIT, LR_SH_FINAL, step, TOTAL_ITERS)
        optimizer.param_groups[0]['lr'] = lr_sigma
        optimizer.param_groups[1]['lr'] = lr_sh

        # Forward
        model.train()
        rays_o, rays_d, rgb_gt = train_data.get_random_rays(BATCH_RAYS)
        result = render_rays(model, rays_o, rays_d, config, perturb=True)

        # MSE loss (Plenoxels uses MSE, not L1)
        mse_fine = F.mse_loss(result['rgb'], rgb_gt)
        mse_coarse = F.mse_loss(result['rgb_coarse'], rgb_gt)
        loss = mse_fine + 0.1 * mse_coarse

        # TV (early only)
        if step <= TV_EARLY_ONLY_STEPS:
            loss = loss + TV_SIGMA_WEIGHT * model.tv_loss()
            loss = loss + TV_SH_WEIGHT * model.tv_loss_sh()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Pruning
        if step % PRUNE_EVERY == 0 and step > LR_SIGMA_DELAY_STEPS:
            with torch.no_grad():
                d = model.density_grid.data[0, 0]
                mask = d < PRUNE_THRESHOLD
                n = mask.sum().item()
                d[mask] = 0.0
                print(f'  Pruned {n}/{d.numel()} '
                      f'({100*n/d.numel():.1f}%)')

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'[{step:7d}/{TOTAL_ITERS}] '
                  f'loss={loss.item():.5f} mse={mse_fine.item():.5f} '
                  f'lr_s={lr_sigma:.2e} lr_sh={lr_sh:.2e} '
                  f'res={model.resolution}³ elapsed={elapsed:.0f}s')

        # Validation
        if step % 5000 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                result = render_image(
                    model, val_data.poses[0],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config)
            val_psnr = psnr(result['rgb'], val_data.images[0])
            val_ssim = ssim(result['rgb'], val_data.images[0]).item()
            print(f'  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}')

            img = (result['rgb'].clamp(0, 1).cpu().numpy() * 255
                   ).astype('uint8')
            Image.fromarray(img).save(
                os.path.join(out_dir, f'val_{step:06d}.png'))

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'iteration': step,
                    'model_state_dict': model.state_dict(),
                    'resolution': model.resolution,
                    'config': config,
                    'psnr': val_psnr,
                }, os.path.join(out_dir, 'best_direct.pt'))
                print(f'  → saved best (PSNR={best_psnr:.2f})')

    # Final
    torch.save({
        'iteration': TOTAL_ITERS,
        'model_state_dict': model.state_dict(),
        'resolution': model.resolution,
        'config': config,
        'psnr': best_psnr,
    }, os.path.join(out_dir, 'direct_final.pt'))

    print(f'\nDone. Best PSNR={best_psnr:.2f} dB. '
          f'Time={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
