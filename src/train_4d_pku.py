"""4D Gaussian Splatting training for PKU-DyMVHumans dataset.

Same Disentangled 4DGS approach as train_4d.py but uses the PKU data format:
  - MVSNet cam_txt cameras (56 cameras, 4K resolution)
  - per_frame/FFFFFF/images/image_c_NNN_f_FFFFFF.png
  - data_COLMAP/ for SfM initialization
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_pku import load_scene_4d_pku, load_sfm_points_pku, read_cam_txt
from train import ssim_loss
from train_4d import (
    FrameCache, GaussianModel4D, make_optimizer_4d,
    render_4d, render_velocity_map, flow_gradient_loss,
    densify_and_prune_4d, save_ply_4d,
)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_4d_pku(model, scene_dir, near, far,
                     num_eval_frames=10, test_every=7,
                     device="cuda", save_dir=None):
    """Evaluate on held-out PKU test cameras across multiple timesteps.

    Uses the test_every split: cameras whose index % test_every == 0
    are held out for evaluation.

    Returns: avg_psnr, avg_ssim
    """
    scene_dir = Path(scene_dir)
    cams_dir = scene_dir / "cams"
    per_frame_dir = scene_dir / "per_frame"

    # Re-read all cameras and identify test set
    cam_files = sorted(cams_dir.glob("*_cam.txt"))
    all_cam_names = [f.stem.replace("_cam", "") for f in cam_files]
    test_indices = [i for i in range(len(all_cam_names)) if i % test_every == 0]

    if not test_indices:
        print("WARNING: No test cameras (test_every=0)")
        return 0.0, 0.0

    # Load test camera params
    test_cam_names = [all_cam_names[i] for i in test_indices]
    test_c2w = []
    test_K = []
    for i in test_indices:
        extrinsic, intrinsic, _ = read_cam_txt(str(cam_files[i]))
        c2w = np.linalg.inv(extrinsic).astype(np.float32)
        test_c2w.append(torch.tensor(c2w, device=device))
        test_K.append(torch.tensor(intrinsic.astype(np.float32), device=device))

    # Discover frame directories
    frame_dirs = sorted(per_frame_dir.glob("[0-9]*"))
    if len(frame_dirs) > num_eval_frames:
        indices = np.linspace(0, len(frame_dirs) - 1,
                              num_eval_frames, dtype=int)
        frame_dirs = [frame_dirs[i] for i in indices]
    T_eval = len(frame_dirs)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for t_idx, fdir in enumerate(frame_dirs):
        t0 = 0.5 if T_eval == 1 else t_idx / (T_eval - 1)
        img_dir = fdir / "images" if (fdir / "images").exists() else fdir
        frame_id = fdir.name  # e.g. "000000"

        for c_idx, cam_name in enumerate(test_cam_names):
            cam_idx = int(cam_name)
            cam_code = f"c_{cam_idx:03d}"
            candidates = [
                img_dir / f"image_{cam_code}_f_{frame_id}.png",
                img_dir / f"image_{cam_name}.png",
                img_dir / f"{cam_name}.png",
            ]
            img_path = None
            for c in candidates:
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                continue

            gt = np.array(Image.open(img_path)).astype(np.float32) / 255.0
            gt = torch.tensor(gt, device=device)
            H, W = gt.shape[:2]

            rendered, _, _ = render_4d(model, test_c2w[c_idx], test_K[c_idx],
                                        W, H, t0, near=near, far=far,
                                        absgrad=False)
            rendered = rendered.clamp(0, 1)

            mse = F.mse_loss(rendered, gt)
            psnr = -10.0 * math.log10(mse.item())
            ssim_val = 1.0 - ssim_loss(rendered, gt).item()

            total_psnr += psnr
            total_ssim += ssim_val
            count += 1

            if save_dir is not None:
                img_out = (rendered.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_out).save(
                    str(save_dir / f"test_t{t_idx:03d}_{cam_name}.jpg"))

    if count == 0:
        print("WARNING: No test images found for evaluation")
        return 0.0, 0.0

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"PKU eval: {count} test images across {T_eval} frames, "
          f"{len(test_cam_names)} test cameras")
    return avg_psnr, avg_ssim


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_4d_pku(args):
    device = "cuda"
    torch.manual_seed(42)

    # Load PKU scene metadata
    print("Loading PKU 4D scene metadata...")
    (frame_paths, cam_names, camtoworlds, K, timestamps,
     near, far, H, W) = load_scene_4d_pku(
        args.scene_dir, num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        test_every=args.test_every, device=device,
    )
    T = len(timestamps)
    C = len(cam_names)

    # Initialize model
    model = GaussianModel4D(device=device)

    if args.init_ply:
        model.init_from_static_ply(args.init_ply)
    else:
        # Cold start from SfM points
        points, colors = load_sfm_points_pku(args.scene_dir, device=device)
        if points is not None:
            print(f"Initialized {len(points)} Gaussians from PKU COLMAP points")
        else:
            from train import init_gaussians_from_cameras
            print("No SfM points found, using random init")
            points, colors = init_gaussians_from_cameras(
                camtoworlds, near, far, args.num_points, device)
        model.init_from_points(points, colors)

    # Scene extent
    cam_positions = camtoworlds[:, :3, 3]
    scene_extent = (cam_positions - cam_positions.mean(0)).norm(dim=1).max().item()
    print(f"Scene extent: {scene_extent:.2f}")

    # Optimizer
    lr_means_init = 0.00016 * scene_extent
    lr_means_final = lr_means_init * 0.01
    optimizer = make_optimizer_4d(model, lr_means_init)

    # Gradient accumulators (spatial + temporal)
    grad_accum = torch.zeros(model.num_gaussians, 1, device=device)
    grad_count = torch.zeros(model.num_gaussians, 1, device=device)
    grad_accum_t = torch.zeros(model.num_gaussians, 1, device=device)
    grad_count_t = torch.zeros(model.num_gaussians, 1, device=device)

    # Frame cache
    cache = FrameCache(max_size=args.frame_cache_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting PKU 4D training for {args.num_steps} steps...")
    print(f"  {T} frames x {C} cameras = {T * C} total training pairs")
    print(f"  Image resolution: {W}x{H}")
    print(f"  Densification: steps {args.densify_from}-{args.densify_until}, "
          f"every {args.densify_every}")
    print(f"  Flow-gradient loss: weight={args.flow_weight}, "
          f"every {args.flow_every} steps after step {args.flow_start}")
    print(f"  Max Gaussians: {args.max_gaussians:,}")
    t_start = time.time()

    for step in range(args.num_steps):
        # LR decay for means
        t = step / args.num_steps
        lr_means = math.exp(
            math.log(lr_means_init) * (1 - t) + math.log(lr_means_final) * t)
        optimizer.param_groups[0]["lr"] = lr_means

        # Random (frame, camera) pair
        frame_idx = torch.randint(0, T, (1,)).item()
        cam_idx = torch.randint(0, C, (1,)).item()
        t0 = float(timestamps[frame_idx])

        # Load GT image from cache
        frame_images = cache.get(frame_paths, frame_idx, device)
        gt_image = frame_images[cam_idx]  # (H, W, 3)

        # SH degree ramp
        sh_degree = min(step // 1000, args.sh_degree_max)

        # Render
        rendered, alpha, info = render_4d(
            model, camtoworlds[cam_idx], K[cam_idx], W, H, t0,
            near=near, far=far, sh_degree=sh_degree, absgrad=True,
        )

        # RGB loss
        l1 = F.l1_loss(rendered, gt_image)
        ssim = ssim_loss(rendered, gt_image)
        loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * ssim

        # Flow-gradient loss
        if (args.flow_weight > 0 and step >= args.flow_start
                and step % args.flow_every == 0):
            vel_map = render_velocity_map(
                model, camtoworlds[cam_idx], K[cam_idx], W, H, t0,
                near=near, far=far)
            l_flow = flow_gradient_loss(vel_map, rendered)
            loss = loss + args.flow_weight * l_flow

        # Opacity regularization
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(model.opacities).mean()

        loss.backward()

        with torch.no_grad():
            # Accumulate gradients for densification
            if step < args.densify_until:
                if model.means.grad is not None:
                    grad_norm = model.means.grad.norm(dim=1, keepdim=True)
                    mask = grad_norm.squeeze() > 0
                    grad_accum[mask] += grad_norm[mask]
                    grad_count[mask] += 1

                if model.mu_t.grad is not None:
                    grad_t = model.mu_t.grad.abs().unsqueeze(1)
                    mask_t = grad_t.squeeze() > 0
                    grad_accum_t[mask_t] += grad_t[mask_t]
                    grad_count_t[mask_t] += 1

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ─── Densification ───
        if (step > args.densify_from and step < args.densify_until
                and step % args.densify_every == 0):
            n_clone, n_split, n_tsplit, n_prune = densify_and_prune_4d(
                model, grad_accum, grad_count,
                grad_accum_t, grad_count_t,
                grad_threshold=args.grad_threshold,
                grad_threshold_t=args.grad_threshold_t,
                min_opacity=args.cull_alpha_thresh,
                scene_extent=scene_extent,
                max_gaussians=args.max_gaussians,
            )
            # Reset accumulators and rebuild optimizer
            grad_accum = torch.zeros(model.num_gaussians, 1, device=device)
            grad_count = torch.zeros(model.num_gaussians, 1, device=device)
            grad_accum_t = torch.zeros(model.num_gaussians, 1, device=device)
            grad_count_t = torch.zeros(model.num_gaussians, 1, device=device)
            optimizer = make_optimizer_4d(model, lr_means)

            print(f"  [{step}] Densify: +{n_clone} clone, +{n_split}x2 split, "
                  f"+{n_tsplit}x2 tsplit, -{n_prune} prune "
                  f"→ {model.num_gaussians:,}")

        # ─── Opacity reset ───
        if (step > 0 and step % args.opacity_reset_every == 0
                and step < args.densify_until):
            with torch.no_grad():
                model.opacities = torch.logit(
                    torch.clamp(torch.sigmoid(model.opacities.detach()), max=0.2)
                ).requires_grad_(True)
            optimizer = make_optimizer_4d(model, lr_means)
            print(f"  [{step}] Opacity reset (max → 0.2)")

        # ─── Logging ───
        if step % 500 == 0:
            with torch.no_grad():
                psnr = -10.0 * math.log10(
                    F.mse_loss(rendered.clamp(0, 1), gt_image).item())
            elapsed = time.time() - t_start
            it_s = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"Step {step:6d}/{args.num_steps} | Loss: {loss.item():.4f} | "
                  f"PSNR: {psnr:.2f} dB | #G: {model.num_gaussians:,} | "
                  f"t={t0:.3f} cam={cam_names[cam_idx]} | {it_s:.1f} it/s")

        # ─── Checkpoint ───
        if step > 0 and step % args.save_every == 0:
            save_ply_4d(model, output_dir / f"point_cloud_4d_{step:06d}.ply")
            print(f"  [{step}] Saved checkpoint")

    # ─── Final ───
    elapsed = time.time() - t_start
    print(f"\nPKU 4D training complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    # Evaluate
    print("Evaluating on held-out test cameras...")
    test_psnr, test_ssim = evaluate_4d_pku(
        model, args.scene_dir, near, far,
        num_eval_frames=min(T, 10), test_every=args.test_every,
        device=device, save_dir=output_dir)
    print(f"Test PSNR: {test_psnr:.2f} dB | Test SSIM: {test_ssim:.4f}")

    # Save final
    final_path = output_dir / "point_cloud_4d_final.ply"
    save_ply_4d(model, final_path)
    print(f"Saved to {final_path} ({model.num_gaussians:,} Gaussians)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4D Gaussian Splatting training for PKU-DyMVHumans")
    # Data
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to PKU scene dir")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/4DGS/outputs/4d_pku")
    parser.add_argument("--test_every", type=int, default=7,
                        help="Hold out every Nth camera for testing (0=no holdout)")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Max frames to use (0 = all)")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_cache_size", type=int, default=20)
    # Init
    parser.add_argument("--init_ply", type=str, default=None,
                        help="Optional: warm-start from trained static PLY")
    parser.add_argument("--num_points", type=int, default=200_000)
    # Training
    parser.add_argument("--num_steps", type=int, default=30_000)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--sh_degree_max", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5_000)
    # Densification
    parser.add_argument("--densify_from", type=int, default=500)
    parser.add_argument("--densify_until", type=int, default=15_000)
    parser.add_argument("--densify_every", type=int, default=100)
    parser.add_argument("--grad_threshold", type=float, default=0.00002)
    parser.add_argument("--grad_threshold_t", type=float, default=0.001)
    parser.add_argument("--cull_alpha_thresh", type=float, default=0.005)
    parser.add_argument("--opacity_reset_every", type=int, default=3000)
    parser.add_argument("--max_gaussians", type=int, default=5_000_000)
    # Losses
    parser.add_argument("--flow_weight", type=float, default=0.01)
    parser.add_argument("--flow_every", type=int, default=5)
    parser.add_argument("--flow_start", type=int, default=1000)
    parser.add_argument("--opacity_reg", type=float, default=0.001)
    args = parser.parse_args()

    train_4d_pku(args)
