"""Multi-resolution coarse-to-fine 3DGS training.

Trains Gaussian layers at increasing image resolutions. Each completed layer
is frozen; new Gaussians are added to capture residual detail at higher
resolution. All layers render jointly via gsplat alpha-compositing.

Usage:
    python train_multires.py --scene lego --data_dir data/nerf_synthetic \
        --output_dir output/multires_v1
"""
import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Add sibling src/ to path for imports from train.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gsplat
from train import (
    GaussianModel,
    densify_and_prune,
    init_gaussians_from_cameras,
    make_optimizer,
    save_ply,
    ssim_loss,
)
from data_nerf_synthetic import load_nerf_synthetic


# ─── Stage Configuration ──────────────────────────────────────────────────────

@dataclass
class StageConfig:
    img_res: int
    max_iters: int
    max_gaussians: int
    densify_from: int
    densify_until: int
    sh_degree: int
    grad_threshold: float
    max_scale: float       # relative to scene_extent
    init_points: int       # initial Gaussians before densification


STAGES = [
    StageConfig(img_res=100, max_iters=15_000, max_gaussians=5_000,
                densify_from=500, densify_until=10_000, sh_degree=0,
                grad_threshold=0.0004, max_scale=0.1, init_points=1_000),
    StageConfig(img_res=200, max_iters=20_000, max_gaussians=30_000,
                densify_from=500, densify_until=15_000, sh_degree=1,
                grad_threshold=0.0002, max_scale=0.05, init_points=5_000),
    StageConfig(img_res=400, max_iters=25_000, max_gaussians=150_000,
                densify_from=500, densify_until=18_000, sh_degree=2,
                grad_threshold=0.0001, max_scale=0.02, init_points=20_000),
    StageConfig(img_res=800, max_iters=30_000, max_gaussians=500_000,
                densify_from=500, densify_until=20_000, sh_degree=3,
                grad_threshold=0.00005, max_scale=0.01, init_points=50_000),
]


# ─── Multi-Resolution Model ──────────────────────────────────────────────────

@dataclass
class FrozenLayer:
    """A completed Gaussian layer — all params frozen (no grad)."""
    means: Tensor
    scales: Tensor       # log-space
    quats: Tensor
    opacities: Tensor    # logit-space
    sh0: Tensor          # (N, 1, 3)
    shN: Tensor          # (N, K, 3) where K depends on stage SH degree
    stage: int
    img_res: int
    sh_degree: int

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]


class MultiResModel:
    """Manages frozen Gaussian layers and one active training layer."""

    def __init__(self, device: str = "cuda"):
        self.frozen_layers: list[FrozenLayer] = []
        self.active: GaussianModel | None = None
        self.device = device

    @property
    def total_gaussians(self) -> int:
        n = sum(l.num_gaussians for l in self.frozen_layers)
        if self.active is not None:
            n += self.active.num_gaussians
        return n

    @property
    def n_frozen(self) -> int:
        return sum(l.num_gaussians for l in self.frozen_layers)

    def freeze_active(self, stage: int, img_res: int, sh_degree: int):
        """Move active layer to frozen list."""
        m = self.active
        layer = FrozenLayer(
            means=m.means.detach(),
            scales=m.scales.detach(),
            quats=m.quats.detach(),
            opacities=m.opacities.detach(),
            sh0=m.sh0.detach(),
            shN=m.shN.detach(),
            stage=stage,
            img_res=img_res,
            sh_degree=sh_degree,
        )
        self.frozen_layers.append(layer)
        self.active = None

    def create_active_layer(self, points: Tensor, colors: Tensor = None):
        """Create a new active layer for the next training stage."""
        self.active = GaussianModel(self.device)
        self.active.init_from_points(points, colors)

    def get_combined_model(self) -> GaussianModel:
        """Return a single GaussianModel with all layers (for PLY export)."""
        parts = {"means": [], "scales": [], "quats": [], "opacities": [],
                 "sh0": [], "shN": []}

        max_shN = 15  # SH degree 3

        for layer in self.frozen_layers:
            parts["means"].append(layer.means)
            parts["scales"].append(layer.scales)
            parts["quats"].append(layer.quats)
            parts["opacities"].append(layer.opacities)
            parts["sh0"].append(layer.sh0)
            # Pad shN to 15 coefficients if needed
            shN = layer.shN
            if shN.shape[1] < max_shN:
                pad = torch.zeros(shN.shape[0], max_shN - shN.shape[1], 3,
                                  device=self.device)
                shN = torch.cat([shN, pad], dim=1)
            parts["shN"].append(shN)

        if self.active is not None:
            parts["means"].append(self.active.means.detach())
            parts["scales"].append(self.active.scales.detach())
            parts["quats"].append(self.active.quats.detach())
            parts["opacities"].append(self.active.opacities.detach())
            parts["sh0"].append(self.active.sh0.detach())
            shN = self.active.shN.detach()
            if shN.shape[1] < max_shN:
                pad = torch.zeros(shN.shape[0], max_shN - shN.shape[1], 3,
                                  device=self.device)
                shN = torch.cat([shN, pad], dim=1)
            parts["shN"].append(shN)

        combined = GaussianModel(self.device)
        combined.means = torch.cat(parts["means"])
        combined.scales = torch.cat(parts["scales"])
        combined.quats = torch.cat(parts["quats"])
        combined.opacities = torch.cat(parts["opacities"])
        combined.sh0 = torch.cat(parts["sh0"])
        combined.shN = torch.cat(parts["shN"])
        return combined


# ─── Rendering ────────────────────────────────────────────────────────────────

def render_multires(model: MultiResModel, camtoworld: Tensor, K: Tensor,
                    W: int, H: int, sh_degree: int = 3,
                    near: float = 0.01, far: float = 1000.0):
    """Render all layers (frozen + active) in one gsplat rasterization pass.

    Frozen parameters are detached (no gradient). Active parameters receive
    gradients through the alpha-compositing pipeline.
    """
    all_means = []
    all_scales = []
    all_quats = []
    all_opacities = []
    all_sh = []

    # Number of SH coefficients at current degree: (sh_degree+1)^2
    n_sh_target = (sh_degree + 1) ** 2  # 1, 4, 9, 16

    # Frozen layers (detached)
    for layer in model.frozen_layers:
        all_means.append(layer.means)
        all_scales.append(torch.exp(layer.scales))
        all_quats.append(layer.quats)
        all_opacities.append(torch.sigmoid(layer.opacities))

        sh = torch.cat([layer.sh0, layer.shN], dim=1)  # (N, 1+K, 3)
        if sh.shape[1] < n_sh_target:
            pad = torch.zeros(sh.shape[0], n_sh_target - sh.shape[1], 3,
                              device=model.device)
            sh = torch.cat([sh, pad], dim=1)
        elif sh.shape[1] > n_sh_target:
            sh = sh[:, :n_sh_target]
        all_sh.append(sh)

    # Active layer (with gradients)
    if model.active is not None:
        m = model.active
        all_means.append(m.means)
        all_scales.append(torch.exp(m.scales))
        all_quats.append(m.quats)
        all_opacities.append(torch.sigmoid(m.opacities))

        sh = torch.cat([m.sh0, m.shN], dim=1)
        if sh.shape[1] < n_sh_target:
            pad = torch.zeros(sh.shape[0], n_sh_target - sh.shape[1], 3,
                              device=model.device)
            sh = torch.cat([sh, pad], dim=1)
        elif sh.shape[1] > n_sh_target:
            sh = sh[:, :n_sh_target]
        all_sh.append(sh)

    means = torch.cat(all_means)
    scales = torch.cat(all_scales)
    quats = torch.cat(all_quats)
    opacities = torch.cat(all_opacities)
    sh_coeffs = torch.cat(all_sh)

    viewmat = torch.linalg.inv(camtoworld)

    renders, alphas, info = gsplat.rasterization(
        means=means,
        quats=quats,
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
        render_mode="RGB",
        rasterize_mode="classic",
    )

    return renders[0], alphas[0], info


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_at_resolution(model: MultiResModel, root: str, scene: str,
                           resolution: int, sh_degree: int,
                           device: str = "cuda", max_views: int = 0):
    """Evaluate on test split at a given resolution."""
    images, c2w, K, near, far = load_nerf_synthetic(
        root, scene, "test", resolution=resolution, device=device
    )
    N, H, W, _ = images.shape
    if max_views > 0:
        N = min(N, max_views)

    total_psnr = 0.0
    total_ssim = 0.0

    for i in range(N):
        rendered, _, _ = render_multires(model, c2w[i], K[i], W, H,
                                         sh_degree=sh_degree, near=near, far=far)
        rendered = rendered.clamp(0, 1)
        mse = F.mse_loss(rendered, images[i])
        psnr = -10.0 * math.log10(max(mse.item(), 1e-10))
        total_psnr += psnr
        total_ssim += 1.0 - ssim_loss(rendered, images[i]).item()

    return total_psnr / N, total_ssim / N


@torch.no_grad()
def evaluate_cross_resolution(model: MultiResModel, root: str, scene: str,
                              current_stage: int, device: str = "cuda"):
    """Evaluate at all resolutions up to current stage."""
    print("  Cross-resolution evaluation:")
    for s in range(current_stage + 1):
        cfg = STAGES[s]
        psnr, ssim = evaluate_at_resolution(
            model, root, scene, cfg.img_res, cfg.sh_degree,
            device=device, max_views=20
        )
        print(f"    {cfg.img_res}px: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")


@torch.no_grad()
def evaluate_lod(model: MultiResModel, root: str, scene: str,
                 resolution: int = 800, device: str = "cuda"):
    """Evaluate LoD: render using only the first K frozen layers."""
    images, c2w, K, near, far = load_nerf_synthetic(
        root, scene, "test", resolution=resolution, device=device
    )
    N, H, W, _ = images.shape
    N = min(N, 20)

    # Save/restore active layer
    saved_active = model.active
    model.active = None

    print("  LoD evaluation at 800px:")
    for n_layers in range(1, len(model.frozen_layers) + 1):
        saved_frozen = model.frozen_layers
        model.frozen_layers = saved_frozen[:n_layers]

        total_psnr = 0.0
        max_sh = max(l.sh_degree for l in model.frozen_layers)
        for i in range(N):
            rendered, _, _ = render_multires(model, c2w[i], K[i], W, H,
                                             sh_degree=max_sh, near=near, far=far)
            rendered = rendered.clamp(0, 1)
            mse = F.mse_loss(rendered, images[i])
            total_psnr += -10.0 * math.log10(max(mse.item(), 1e-10))

        avg_psnr = total_psnr / N
        n_gauss = sum(l.num_gaussians for l in model.frozen_layers)
        print(f"    Layers 0..{n_layers-1}: {n_gauss:,} Gaussians, PSNR={avg_psnr:.2f} dB")

        model.frozen_layers = saved_frozen

    model.active = saved_active


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model: MultiResModel, output_dir: Path, stage: int):
    """Save multi-resolution checkpoint."""
    ckpt = {"stage": stage, "frozen_layers": [], "active": None}

    for layer in model.frozen_layers:
        ckpt["frozen_layers"].append({
            "means": layer.means.cpu(), "scales": layer.scales.cpu(),
            "quats": layer.quats.cpu(), "opacities": layer.opacities.cpu(),
            "sh0": layer.sh0.cpu(), "shN": layer.shN.cpu(),
            "stage": layer.stage, "img_res": layer.img_res,
            "sh_degree": layer.sh_degree,
        })

    if model.active is not None:
        ckpt["active"] = {
            attr: getattr(model.active, attr).detach().cpu()
            for attr in ["means", "scales", "quats", "opacities", "sh0", "shN"]
        }

    torch.save(ckpt, output_dir / f"multires_stage{stage}.pt")

    # Combined PLY for visualization
    combined = model.get_combined_model()
    save_ply(combined, output_dir / f"combined_stage{stage}.ply")
    print(f"  Saved checkpoint: {model.total_gaussians:,} total Gaussians")


def load_checkpoint(path: str, device: str = "cuda") -> MultiResModel:
    """Load multi-resolution checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiResModel(device)

    for ld in ckpt["frozen_layers"]:
        layer = FrozenLayer(
            means=ld["means"].to(device), scales=ld["scales"].to(device),
            quats=ld["quats"].to(device), opacities=ld["opacities"].to(device),
            sh0=ld["sh0"].to(device), shN=ld["shN"].to(device),
            stage=ld["stage"], img_res=ld["img_res"], sh_degree=ld["sh_degree"],
        )
        model.frozen_layers.append(layer)

    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_stage(model: MultiResModel, images: Tensor, camtoworlds: Tensor,
                K: Tensor, near: float, far: float, stage_idx: int,
                cfg: StageConfig, scene_extent: float, ssim_weight: float,
                opacity_reg: float, device: str = "cuda"):
    """Train one resolution stage."""
    N, H, W, _ = images.shape

    # LR scaling: finer stages use smaller learning rates
    lr_scale = 0.5 ** stage_idx
    lr_means_init = 0.00016 * scene_extent * lr_scale
    lr_means_final = lr_means_init * 0.01

    optimizer = make_optimizer(
        model.active, lr_means_init,
        lr_scales=0.005 * lr_scale,
        lr_quats=0.001 * lr_scale,
        lr_opacities=0.05,  # keep constant
        lr_sh0=0.0025 * lr_scale,
        lr_shN=0.000125 * lr_scale,
    )

    grad_accum = torch.zeros(model.active.num_gaussians, 1, device=device)
    grad_count = torch.zeros(model.active.num_gaussians, 1, device=device)

    best_loss = float("inf")
    patience = 10
    no_improve = 0
    eval_interval = 2000

    t0 = time.time()

    for step in range(cfg.max_iters):
        # LR schedule (exponential decay for means)
        t = step / cfg.max_iters
        lr_means = math.exp(
            math.log(lr_means_init) * (1 - t) + math.log(lr_means_final) * t
        )
        optimizer.param_groups[0]["lr"] = lr_means

        # Random training view
        idx = torch.randint(0, N, (1,)).item()
        gt_image = images[idx]

        # Render all layers jointly
        rendered, alpha, info = render_multires(
            model, camtoworlds[idx], K[idx], W, H,
            sh_degree=cfg.sh_degree, near=near, far=far,
        )

        # Loss
        l1 = F.l1_loss(rendered, gt_image)
        ssim = ssim_loss(rendered, gt_image)
        loss = (1.0 - ssim_weight) * l1 + ssim_weight * ssim

        # Opacity regularization on active layer only
        if opacity_reg > 0:
            loss = loss + opacity_reg * torch.sigmoid(model.active.opacities).mean()

        loss.backward()

        # Track gradients for active layer densification
        with torch.no_grad():
            if (model.active.means.grad is not None
                    and step < cfg.densify_until):
                grad_norm = model.active.means.grad.norm(dim=1, keepdim=True)
                mask = grad_norm.squeeze() > 0
                if mask.any():
                    grad_accum[mask] += grad_norm[mask]
                    grad_count[mask] += 1

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Densification (active layer only)
        if (step > cfg.densify_from and step < cfg.densify_until
                and step % 100 == 0):
            n_clone, n_split, n_prune = densify_and_prune(
                model.active, grad_accum, grad_count,
                grad_threshold=cfg.grad_threshold,
                min_opacity=0.005,
                max_scale=cfg.max_scale,
                scene_extent=scene_extent,
                max_gaussians=cfg.max_gaussians,
            )
            # Reset accumulators and rebuild optimizer
            grad_accum = torch.zeros(model.active.num_gaussians, 1, device=device)
            grad_count = torch.zeros(model.active.num_gaussians, 1, device=device)
            optimizer = make_optimizer(
                model.active, lr_means,
                lr_scales=0.005 * lr_scale,
                lr_quats=0.001 * lr_scale,
                lr_opacities=0.05,
                lr_sh0=0.0025 * lr_scale,
                lr_shN=0.000125 * lr_scale,
            )
            if step % 500 == 0:
                print(f"    [{step}] Densify: +{n_clone} clone, +{n_split}x2 split, "
                      f"-{n_prune} prune -> {model.active.num_gaussians:,}")

        # Opacity reset every 5000 steps
        if step > 0 and step % 5000 == 0 and step < cfg.densify_until:
            with torch.no_grad():
                model.active.opacities = torch.logit(
                    torch.clamp(torch.sigmoid(model.active.opacities.detach()), max=0.2)
                ).requires_grad_(True)
            optimizer = make_optimizer(
                model.active, lr_means,
                lr_scales=0.005 * lr_scale,
                lr_quats=0.001 * lr_scale,
                lr_opacities=0.05,
                lr_sh0=0.0025 * lr_scale,
                lr_shN=0.000125 * lr_scale,
            )

        # Logging
        if step % 500 == 0:
            with torch.no_grad():
                psnr = -10.0 * math.log10(max(F.mse_loss(rendered, gt_image).item(), 1e-10))
            elapsed = time.time() - t0
            it_s = (step + 1) / elapsed if elapsed > 0 else 0
            total_g = model.total_gaussians
            print(f"    [{step:5d}/{cfg.max_iters}] Loss={loss.item():.4f} "
                  f"PSNR={psnr:.2f} #G={model.active.num_gaussians:,} "
                  f"(total={total_g:,}) lr_m={lr_means:.6f} {it_s:.1f}it/s")

        # Early stopping (after densification window)
        if step > 0 and step % eval_interval == 0 and step > cfg.densify_until:
            avg_loss = loss.item()
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at step {step}")
                break

    elapsed = time.time() - t0
    print(f"  Stage {stage_idx} done: {elapsed:.0f}s, "
          f"{model.active.num_gaussians:,} active Gaussians")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def train_multires(args):
    device = "cuda"
    torch.manual_seed(42)

    model = MultiResModel(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute scene extent from cameras (use highest-res split)
    _, c2w_full, _, _, _ = load_nerf_synthetic(
        args.data_dir, args.scene, "train", resolution=100, device=device
    )
    cam_positions = c2w_full[:, :3, 3]
    scene_extent = (cam_positions - cam_positions.mean(0)).norm(dim=1).max().item()
    print(f"Scene extent: {scene_extent:.2f}")

    # Resume from a specific stage if requested
    start_stage = 0
    if args.resume:
        model = load_checkpoint(args.resume, device)
        start_stage = len(model.frozen_layers)
        print(f"Resumed from {args.resume}: {start_stage} frozen layers, "
              f"{model.n_frozen:,} Gaussians")

    stages = STAGES[:args.max_stages] if args.max_stages else STAGES

    for stage_idx in range(start_stage, len(stages)):
        cfg = stages[stage_idx]

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {cfg.img_res}px, max {cfg.max_gaussians:,} "
              f"new Gaussians, SH degree {cfg.sh_degree}")
        print(f"  Frozen layers: {len(model.frozen_layers)} "
              f"({model.n_frozen:,} Gaussians)")
        print(f"{'='*60}")

        # Load data at this resolution
        images, camtoworlds, K, near, far = load_nerf_synthetic(
            args.data_dir, args.scene, "train",
            resolution=cfg.img_res, device=device,
        )
        N, H, W, _ = images.shape

        # Initialize new active layer
        points, colors = init_gaussians_from_cameras(
            camtoworlds, near, far, num_points=cfg.init_points, device=device,
        )
        model.create_active_layer(points, colors)
        print(f"  Initialized {cfg.init_points:,} new Gaussians")

        # Train this stage
        train_stage(
            model, images, camtoworlds, K, near, far,
            stage_idx=stage_idx, cfg=cfg, scene_extent=scene_extent,
            ssim_weight=args.ssim_weight, opacity_reg=args.opacity_reg,
            device=device,
        )

        # Freeze active layer
        model.freeze_active(stage_idx, cfg.img_res, cfg.sh_degree)

        # Save checkpoint
        save_checkpoint(model, output_dir, stage_idx)

        # Cross-resolution evaluation
        evaluate_cross_resolution(
            model, args.data_dir, args.scene, stage_idx, device=device,
        )

    # Final full evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    psnr, ssim = evaluate_at_resolution(
        model, args.data_dir, args.scene,
        resolution=800, sh_degree=3, device=device,
    )
    print(f"  Test 800px: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    print(f"  Total Gaussians: {model.total_gaussians:,}")

    # LoD evaluation
    if len(model.frozen_layers) > 1:
        evaluate_lod(model, args.data_dir, args.scene, device=device)

    # Save final combined PLY
    combined = model.get_combined_model()
    save_ply(combined, output_dir / "combined_final.ply")
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-resolution coarse-to-fine 3DGS")
    parser.add_argument("--scene", type=str, default="lego",
                        help="NeRF Synthetic scene name")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to nerf_synthetic directory")
    parser.add_argument("--output_dir", type=str, default="output/multires_v1")
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--opacity_reg", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_stages", type=int, default=None,
                        help="Run only first N stages (for debugging)")
    args = parser.parse_args()

    train_multires(args)
