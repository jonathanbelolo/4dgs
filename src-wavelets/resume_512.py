"""Resume 512³ training from best_direct.pt checkpoint."""
import os, time, torch
from direct_grid_volume import DirectGridVolume
from data import NerfSyntheticDataset
from renderer import render_rays, render_image
from metrics import psnr, ssim
from config import Config
from PIL import Image

# Match the train_direct Plenoxels LR schedule
def plenoxels_lr(lr_init, lr_final, step, max_steps, delay_steps=0, delay_mult=0.01):
    if delay_steps > 0 and step < delay_steps:
        delay_rate = delay_mult + (1 - delay_mult) * (step / delay_steps)
    else:
        delay_rate = 1.0
    t = min(step / max(max_steps, 1), 1.0)
    log_lerp = lr_init * (lr_final / lr_init) ** t
    return delay_rate * log_lerp

config = Config(training_mode='direct')
device = torch.device('cuda')
total_iters = 120000

# Load checkpoint
ckpt = torch.load('output/lego/best_direct.pt', map_location='cpu', weights_only=False)
model = DirectGridVolume(resolution=512, num_channels=28, scene_bound=1.5).to(device)
model.load_state_dict(ckpt['model_state_dict'])
start_iter = ckpt['iteration']
best_psnr = ckpt['psnr']
print(f'Resumed from iter {start_iter}, PSNR={best_psnr:.2f} dB, res=512³')

# Data
train_data = NerfSyntheticDataset('data/nerf_synthetic', 'lego', 'train',
    resolution=800, white_bg=True, device=device)
val_data = NerfSyntheticDataset('data/nerf_synthetic', 'lego', 'val',
    resolution=800, white_bg=True, device=device)

out_dir = 'output/lego'

# Plenoxels LR config
lr_sigma_init, lr_sigma_final = 3e1, 5e-2
lr_sh_init, lr_sh_final = 1e-2, 5e-6
lr_sigma_delay_steps = 15000
lr_sigma_delay_mult = 0.01

optimizer = torch.optim.RMSprop([
    {'params': [model.density_grid], 'lr': lr_sigma_init},
    {'params': [model.sh_grid], 'lr': lr_sh_init},
], alpha=0.95, eps=1e-8)

t0 = time.time()
global_iter = start_iter

for iteration in range(total_iters - start_iter):
    current_lr_sigma = plenoxels_lr(lr_sigma_init, lr_sigma_final,
        global_iter, total_iters, lr_sigma_delay_steps, lr_sigma_delay_mult)
    current_lr_sh = plenoxels_lr(lr_sh_init, lr_sh_final, global_iter, total_iters)
    optimizer.param_groups[0]['lr'] = current_lr_sigma
    optimizer.param_groups[1]['lr'] = current_lr_sh

    model.train()
    rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)
    result = render_rays(model, rays_o, rays_d, config, perturb=True)

    l1_loss = torch.nn.functional.l1_loss(result['rgb'], rgb_gt)
    l1_coarse = torch.nn.functional.l1_loss(result['rgb_coarse'], rgb_gt)
    loss = l1_loss + 0.1 * l1_coarse

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    global_iter += 1

    if global_iter % 100 == 0:
        elapsed = time.time() - t0
        print(f'[{global_iter:6d}/{total_iters}] loss={loss.item():.4f} l1={l1_loss.item():.4f} '
              f'lr_s={current_lr_sigma:.2e} lr_sh={current_lr_sh:.2e} elapsed={elapsed:.0f}s')

    if global_iter % 5000 == 0:
        with torch.no_grad():
            density_vals = model.density_grid.data[0, 0]
            low = density_vals < 0.01
            n_pruned = low.sum().item()
            model.density_grid.data[0, 0][low] = 0.0
            print(f'  Pruned {n_pruned}/{density_vals.numel()} ({100*n_pruned/density_vals.numel():.1f}%)')

    if global_iter % 5000 == 0 or global_iter == start_iter + 1:
        model.eval()
        with torch.no_grad():
            result = render_image(model, val_data.poses[0], 800, 800, val_data.focal, config)
        val_psnr = psnr(result['rgb'], val_data.images[0])
        val_ssim = ssim(result['rgb'], val_data.images[0]).item()
        print(f'  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}')
        img = (result['rgb'].clamp(0,1).cpu().numpy() * 255).astype('uint8')
        Image.fromarray(img).save(os.path.join(out_dir, f'val_{global_iter:06d}.png'))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'iteration': global_iter, 'model_state_dict': model.state_dict(),
                'resolution': 512, 'config': config, 'psnr': val_psnr},
                os.path.join(out_dir, 'best_direct.pt'))
            print(f'  → saved best (PSNR={best_psnr:.2f})')

torch.save({'iteration': global_iter, 'model_state_dict': model.state_dict(),
    'resolution': 512, 'config': config, 'psnr': best_psnr},
    os.path.join(out_dir, 'direct_final.pt'))
print(f'Done. Best PSNR={best_psnr:.2f} dB. Time={( time.time()-t0)/60:.1f} min')
