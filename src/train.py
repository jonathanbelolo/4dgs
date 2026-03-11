"""Baseline 3DGS training on a single timestep using gsplat."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import gsplat
from data import load_scene


# ─── Gaussian Parameters ───────────────────────────────────────────────────────

class GaussianModel:
    """Stores and manages 3D Gaussian parameters."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def init_from_points(self, points: Tensor, colors: Tensor = None):
        """Initialize Gaussians from a point cloud."""
        n = points.shape[0]
        self.means = points.clone().detach().requires_grad_(True)
        self.scales = torch.full((n, 3), -3.0, device=self.device).requires_grad_(True)
        self.quats = torch.zeros(n, 4, device=self.device)
        self.quats[:, 0] = 1.0
        self.quats = self.quats.requires_grad_(True)
        self.opacities = torch.logit(
            torch.full((n,), 0.1, device=self.device)
        ).requires_grad_(True)
        self.sh0 = torch.zeros(n, 1, 3, device=self.device).requires_grad_(True)
        self.shN = torch.zeros(n, 15, 3, device=self.device).requires_grad_(True)

        if colors is not None:
            C0 = 0.28209479177387814
            self.sh0 = ((colors - 0.5) / C0).unsqueeze(1).clone().detach().requires_grad_(True)

    @property
    def num_gaussians(self):
        return self.means.shape[0]


# ─── Optimizer Helper ──────────────────────────────────────────────────────────

def make_optimizer(model, lr_means, lr_scales=0.005, lr_quats=0.001,
                   lr_opacities=0.05, lr_sh0=0.0025, lr_shN=0.000125):
    return torch.optim.Adam([
        {"params": [model.means], "lr": lr_means, "name": "means"},
        {"params": [model.scales], "lr": lr_scales, "name": "scales"},
        {"params": [model.quats], "lr": lr_quats, "name": "quats"},
        {"params": [model.opacities], "lr": lr_opacities, "name": "opacities"},
        {"params": [model.sh0], "lr": lr_sh0, "name": "sh0"},
        {"params": [model.shN], "lr": lr_shN, "name": "shN"},
    ])


# ─── Rendering ─────────────────────────────────────────────────────────────────

def render(model, camtoworld, K, W, H, near=0.01, far=1000.0, sh_degree=3,
           render_depth=False, antialiased=False, absgrad=False):
    """Render a single view using gsplat."""
    viewmat = torch.linalg.inv(camtoworld)

    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities)
    sh_coeffs = torch.cat([model.sh0, model.shN], dim=1)

    render_mode = "RGB+ED" if render_depth else "RGB"
    rasterize_mode = "antialiased" if antialiased else "classic"

    renders, alphas, info = gsplat.rasterization(
        means=model.means,
        quats=model.quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        near_plane=near,
        far_plane=far,
        sh_degree=sh_degree,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
        absgrad=absgrad,
    )

    if render_depth:
        # Last channel is expected depth
        rgb = renders[0, :, :, :3]
        depth = renders[0, :, :, 3]
        return rgb, depth, alphas[0], info

    return renders[0], alphas[0], info


# ─── Loss ──────────────────────────────────────────────────────────────────────

def ssim_loss(pred: Tensor, target: Tensor, window_size: int = 11):
    """SSIM loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)

    kernel_1d = torch.exp(-torch.arange(window_size, device=pred.device, dtype=pred.dtype)
                          .sub(window_size // 2).pow(2) / (2 * 1.5 ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(3, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=3)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# ─── Init from scene ───────────────────────────────────────────────────────────

def init_gaussians_from_sfm(sfm_dir, device="cuda"):
    """Initialize Gaussians from COLMAP point cloud (dense preferred, sparse fallback)."""
    sfm_dir = Path(sfm_dir)
    # Prefer dense stereo points over sparse triangulation
    if (sfm_dir / "dense_points3d.npy").exists():
        points = np.load(str(sfm_dir / "dense_points3d.npy"))
        colors = np.load(str(sfm_dir / "dense_colors3d.npy"))
        print(f"Loaded {len(points)} DENSE SfM points from {sfm_dir}")
    elif (sfm_dir / "points3d.npy").exists():
        points = np.load(str(sfm_dir / "points3d.npy"))
        colors = np.load(str(sfm_dir / "colors3d.npy"))
        print(f"Loaded {len(points)} sparse SfM points from {sfm_dir}")
    else:
        return None, None
    return torch.tensor(points, device=device), torch.tensor(colors, device=device)


def init_gaussians_from_cameras(camtoworlds, near, far, num_points, device="cuda"):
    """Initialize Gaussians randomly in the view frustum (fallback)."""
    cam_positions = camtoworlds[:, :3, 3]
    center = cam_positions.mean(dim=0)
    dists = (cam_positions - center).norm(dim=1)
    extent = dists.max().item()

    points = center[None] + torch.randn(num_points, 3, device=device) * extent * 0.5
    colors = torch.rand(num_points, 3, device=device) * 0.5 + 0.25

    return points, colors


# ─── Densification ─────────────────────────────────────────────────────────────

def densify_and_prune(model, grad_accum, grad_count, grad_threshold=0.0002,
                      min_opacity=0.005, max_scale=0.05, scene_extent=1.0,
                      max_gaussians=3_000_000):
    """Densify Gaussians with high gradients, prune low-opacity ones."""
    avg_grad = grad_accum / grad_count.clamp(min=1)
    avg_grad[avg_grad.isnan()] = 0.0

    scales = torch.exp(model.scales)
    max_scale_per_gauss = scales.max(dim=1).values
    high_grad = avg_grad.squeeze() > grad_threshold

    # Clone small Gaussians
    clone_mask = high_grad & (max_scale_per_gauss < max_scale * scene_extent)
    n_clone = clone_mask.sum().item()

    # Split large Gaussians
    split_mask = high_grad & (max_scale_per_gauss >= max_scale * scene_extent)
    n_split = split_mask.sum().item()

    # Cap growth if approaching max
    total_after = model.num_gaussians + n_clone + n_split
    if total_after > max_gaussians:
        n_clone = 0
        n_split = 0
        clone_mask[:] = False
        split_mask[:] = False

    new_params = {k: [] for k in ["means", "scales", "quats", "opacities", "sh0", "shN"]}

    if n_clone > 0:
        for attr in new_params:
            new_params[attr].append(getattr(model, attr)[clone_mask].detach().clone())

    if n_split > 0:
        split_scales = model.scales[split_mask].detach().clone() - math.log(1.6)
        stds = torch.exp(model.scales[split_mask]).max(dim=1).values
        split_means = model.means[split_mask].detach().clone()
        offset = torch.randn_like(split_means) * stds[:, None]

        new_params["means"].append(split_means + offset)
        new_params["means"].append(split_means - offset)
        new_params["scales"].append(split_scales)
        new_params["scales"].append(split_scales)
        for attr in ["quats", "opacities", "sh0", "shN"]:
            val = getattr(model, attr)[split_mask].detach().clone()
            new_params[attr].append(val)
            new_params[attr].append(val)

    # Prune low-opacity and huge Gaussians (only prune those NOT just cloned/split)
    opacities = torch.sigmoid(model.opacities)
    prune_mask = (opacities < min_opacity).squeeze()
    # Also prune very large Gaussians
    prune_mask = prune_mask | (max_scale_per_gauss > max_scale * scene_extent * 5)
    if n_split > 0:
        prune_mask = prune_mask | split_mask
    keep_mask = ~prune_mask
    n_prune = prune_mask.sum().item()

    for attr in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
        old = getattr(model, attr)[keep_mask].detach()
        parts = [old] + new_params[attr]
        setattr(model, attr, torch.cat(parts).detach().clone().requires_grad_(True))

    return n_clone, n_split, n_prune


# ─── Test Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, scene_dir, downsample, near, far, device="cuda", save_dir=None,
             frames_base="frames"):
    """Evaluate on held-out test camera (cam00)."""
    from data import load_scene
    from PIL import Image
    test_images, test_c2w, test_K, _, _ = load_scene(
        scene_dir, frames_subdir=f"{frames_base}/test", downsample=downsample, device=device
    )
    N, H, W, _ = test_images.shape
    total_psnr = 0.0
    total_ssim = 0.0
    for i in range(N):
        rendered, _, _ = render(model, test_c2w[i], test_K[i], W, H, near=near, far=far)
        rendered = rendered.clamp(0, 1)
        mse = F.mse_loss(rendered, test_images[i])
        total_psnr += -10.0 * math.log10(mse.item())
        total_ssim += 1.0 - ssim_loss(rendered, test_images[i]).item()
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            img = (rendered.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(str(save_dir / f"test{i}_rendered.jpg"))
            gt = (test_images[i].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(gt).save(str(save_dir / f"test{i}_gt.jpg"))
    return total_psnr / N, total_ssim / N


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda"
    torch.manual_seed(42)

    # Load data
    print("Loading scene data...")
    images, camtoworlds, K, near, far = load_scene(
        args.scene_dir, frames_subdir=f"{args.frames_base}/train",
        downsample=args.downsample, device=device
    )
    N, H, W, _ = images.shape
    print(f"Training with {N} views at {W}x{H}")

    # Load monocular depth maps for depth supervision
    mono_depths = None
    if args.depth_weight > 0:
        from data import load_mono_depths
        mono_depths = load_mono_depths(
            args.scene_dir, frames_subdir=f"{args.frames_base}/train",
            device=device
        )
        if mono_depths is None:
            print("WARNING: No monocular depth maps found, disabling depth supervision")
            args.depth_weight = 0.0

    # Initialize Gaussians — prefer SfM point cloud (dense > sparse > random)
    model = GaussianModel(device=device)
    sfm_dir = Path(args.scene_dir) / "colmap"
    if args.sfm_dir:
        sfm_dir = Path(args.sfm_dir)
    points, colors = init_gaussians_from_sfm(sfm_dir, device)
    if points is not None:
        print(f"Initialized {len(points)} Gaussians from SfM points")
    else:
        print(f"No SfM points found at {sfm_dir}, using random init ({args.num_points} points)")
        points, colors = init_gaussians_from_cameras(camtoworlds, near, far,
                                                      args.num_points, device)
    model.init_from_points(points, colors)

    # Scene extent
    cam_positions = camtoworlds[:, :3, 3]
    scene_extent = (cam_positions - cam_positions.mean(0)).norm(dim=1).max().item()
    print(f"Scene extent: {scene_extent:.2f}")

    # Optimizer
    lr_means_init = 0.00016 * scene_extent
    lr_means_final = lr_means_init * 0.01
    optimizer = make_optimizer(model, lr_means_init)

    # Gradient accumulation
    grad_accum = torch.zeros(model.num_gaussians, 1, device=device)
    grad_count = torch.zeros(model.num_gaussians, 1, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"Densification: steps {args.densify_from}-{args.densify_until}, every {args.densify_every}")
    print(f"Max Gaussians: {args.max_gaussians:,}")
    t0 = time.time()

    for step in range(args.num_steps):
        # Exponential LR decay for means
        t = step / args.num_steps
        lr_means = math.exp(math.log(lr_means_init) * (1 - t) + math.log(lr_means_final) * t)
        optimizer.param_groups[0]["lr"] = lr_means

        # Random camera
        idx = torch.randint(0, N, (1,)).item()
        gt_image = images[idx]

        # SH degree ramp: 0 → 1 → ... → max
        sh_degree = min(step // 1000, args.sh_degree_max)

        # Render
        use_depth = args.depth_weight > 0 and mono_depths is not None
        result = render(
            model, camtoworlds[idx], K[idx], W, H,
            near=near, far=far, sh_degree=sh_degree,
            render_depth=use_depth, antialiased=False, absgrad=False,
        )
        if use_depth:
            rendered, rendered_depth, alpha, info = result
        else:
            rendered, alpha, info = result

        # Loss: L1 + SSIM
        l1 = F.l1_loss(rendered, gt_image)
        ssim = ssim_loss(rendered, gt_image)
        loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * ssim

        # Monocular depth supervision (Pearson correlation, scale-invariant)
        if use_depth:
            mono_d = mono_depths[idx]  # relative depth [0, 1]
            # Pearson correlation loss: 1 - corr(rendered_depth, mono_depth)
            d_pred = rendered_depth.flatten()
            d_mono = mono_d.flatten()
            # Only use pixels where alpha > 0.5 (rendered something)
            valid = alpha.flatten().squeeze() > 0.5
            if valid.sum() > 100:
                dp = d_pred[valid]
                dm = d_mono[valid]
                dp_centered = dp - dp.mean()
                dm_centered = dm - dm.mean()
                cov = (dp_centered * dm_centered).mean()
                std_p = dp_centered.pow(2).mean().sqrt().clamp(min=1e-6)
                std_m = dm_centered.pow(2).mean().sqrt().clamp(min=1e-6)
                pearson = cov / (std_p * std_m)
                depth_loss = 1.0 - pearson
                loss = loss + args.depth_weight * depth_loss

        # Opacity regularization (suppress floaters)
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(model.opacities).mean()

        loss.backward()

        with torch.no_grad():
            # Track gradients for densification
            if model.means.grad is not None and step < args.densify_until:
                if "gaussian_ids" in info:
                    gid = info["gaussian_ids"]
                    grad_norm = model.means.grad[gid].norm(dim=1, keepdim=True)
                    grad_accum[gid] += grad_norm
                    grad_count[gid] += 1
                else:
                    grad_norm = model.means.grad.norm(dim=1, keepdim=True)
                    mask = grad_norm.squeeze() > 0
                    grad_accum[mask] += grad_norm[mask]
                    grad_count[mask] += 1

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ─── Densification ───
        if step > args.densify_from and step < args.densify_until and step % args.densify_every == 0:
            n_clone, n_split, n_prune = densify_and_prune(
                model, grad_accum, grad_count,
                grad_threshold=args.grad_threshold,
                min_opacity=args.cull_alpha_thresh,
                scene_extent=scene_extent,
                max_gaussians=args.max_gaussians,
            )
            # Reset accumulators and rebuild optimizer
            grad_accum = torch.zeros(model.num_gaussians, 1, device=device)
            grad_count = torch.zeros(model.num_gaussians, 1, device=device)
            optimizer = make_optimizer(model, lr_means)

            if step % args.densify_every == 0:
                print(f"  [{step}] Densify: +{n_clone} clone, +{n_split}x2 split, "
                      f"-{n_prune} prune → {model.num_gaussians:,}")

        # ─── Opacity reset (gentle, disabled by default for few-view) ───
        if (not args.no_opacity_reset and step > 0
                and step % args.opacity_reset_every == 0 and step < args.densify_until):
            with torch.no_grad():
                model.opacities = torch.logit(
                    torch.clamp(torch.sigmoid(model.opacities.detach()), max=0.2)
                ).requires_grad_(True)
            optimizer = make_optimizer(model, lr_means)
            print(f"  [{step}] Opacity reset (max → 0.2)")

        # ─── Logging ───
        if step % 500 == 0:
            with torch.no_grad():
                psnr = -10.0 * math.log10(F.mse_loss(rendered, gt_image).item())
            elapsed = time.time() - t0
            it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"Step {step:6d}/{args.num_steps} | Loss: {loss.item():.4f} | "
                  f"PSNR: {psnr:.2f} dB | #G: {model.num_gaussians:,} | "
                  f"lr_m: {lr_means:.6f} | {it_per_sec:.1f} it/s")

        # ─── Save checkpoint ───
        if step > 0 and step % args.save_every == 0:
            save_ply(model, output_dir / f"point_cloud_{step:06d}.ply")

    # ─── Final evaluation ───
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Evaluate on test
    print("Evaluating on test camera (cam00)...")
    test_psnr, test_ssim = evaluate(model, args.scene_dir, args.downsample, near, far, device,
                                     save_dir=output_dir, frames_base=args.frames_base)
    print(f"Test PSNR: {test_psnr:.2f} dB | Test SSIM: {test_ssim:.4f}")

    # Save final
    final_path = output_dir / "point_cloud_final.ply"
    save_ply(model, final_path)
    print(f"Saved to {final_path} ({model.num_gaussians:,} Gaussians)")


# ─── PLY Export ────────────────────────────────────────────────────────────────

def save_ply(model, path):
    """Save Gaussian model as a .ply file compatible with standard viewers."""
    from plyfile import PlyData, PlyElement

    means = model.means.detach().cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    quats = model.quats.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()
    sh0 = model.sh0.detach().cpu().numpy().reshape(-1, 3)
    shN = model.shN.detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, 45)

    n = means.shape[0]

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(45):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    elements = np.empty(n, dtype=dtype)
    elements["x"] = means[:, 0]
    elements["y"] = means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    elements["f_dc_0"] = sh0[:, 0]
    elements["f_dc_1"] = sh0[:, 1]
    elements["f_dc_2"] = sh0[:, 2]
    for i in range(45):
        elements[f"f_rest_{i}"] = shN[:, i]
    elements["opacity"] = opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = quats[:, 0]
    elements["rot_1"] = quats[:, 1]
    elements["rot_2"] = quats[:, 2]
    elements["rot_3"] = quats[:, 3]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(str(path))


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 3DGS training")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/workspace/4DGS/outputs/baseline")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--frames_base", type=str, default="frames",
                        help="Base frames directory (e.g. 'frames' or 'frames_fullres')")
    parser.add_argument("--num_points", type=int, default=200_000)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    # Densification
    parser.add_argument("--densify_from", type=int, default=500)
    parser.add_argument("--densify_until", type=int, default=40_000)
    parser.add_argument("--densify_every", type=int, default=100)
    parser.add_argument("--grad_threshold", type=float, default=0.00008)
    parser.add_argument("--cull_alpha_thresh", type=float, default=0.002)
    parser.add_argument("--opacity_reset_every", type=int, default=5000)
    parser.add_argument("--max_gaussians", type=int, default=10_000_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--sfm_dir", type=str, default=None,
                        help="Directory with SfM points (points3d.npy, colors3d.npy)")
    parser.add_argument("--sh_degree_max", type=int, default=2,
                        help="Max SH degree (2 for few-view, 3 for many-view)")
    parser.add_argument("--no_opacity_reset", action="store_true",
                        help="Disable opacity reset (recommended for few-view)")
    # Depth supervision
    parser.add_argument("--depth_weight", type=float, default=0.5,
                        help="Weight for monocular depth loss with exponential decay (0 to disable)")
    # Regularization
    parser.add_argument("--opacity_reg", type=float, default=0.001,
                        help="Opacity regularization weight to suppress floaters (0 to disable)")
    args = parser.parse_args()

    train(args)
