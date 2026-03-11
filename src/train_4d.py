"""Disentangled 4D Gaussian Splatting training (arXiv 2503.22159).

Extends static 3DGS with per-Gaussian temporal parameters:
  - mu_t: temporal center
  - s_t: temporal log-scale (lifespan)
  - velocity: 3D velocity vector

Position and opacity are modulated per-timestep:
  means_t = means + velocity * (mu_t - t)
  opacity_t = sigmoid(opacity) * exp(-0.5 * ((mu_t - t) / exp(s_t))^2)
"""
import argparse
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import gsplat
from data import load_llff_poses
from train import ssim_loss, init_gaussians_from_sfm, init_gaussians_from_cameras


# ─── Multi-timestep Data Loading ─────────────────────────────────────────────

def load_scene_4d(scene_dir: str, frames_base: str = "frames",
                  downsample: int = 2, num_frames: int = 0,
                  frame_stride: int = 1, device: str = "cuda"):
    """Load camera info and build frame path index for 4D training.

    Does NOT load images — returns paths for lazy on-demand loading.

    Args:
        scene_dir: Path to scene (e.g. /workspace/Data/Neu3D/coffee_martini)
        frames_base: Base frames directory
        downsample: Downsample factor used during extraction
        num_frames: Max frames to use (0 = all available)
        frame_stride: Temporal subsampling stride
        device: torch device

    Returns:
        frame_paths: list of lists — frame_paths[t][c] = path to image
        cam_names: list of camera names
        camtoworlds: (C, 4, 4) float32 tensor
        K: (C, 3, 3) float32 intrinsics matrices
        timestamps: (T,) float32 tensor in [0, 1]
        near, far: floats
        H, W: image dimensions
    """
    scene_dir = Path(scene_dir)
    train_dir = scene_dir / frames_base / "train"

    # Discover available frame directories
    frame_dirs = sorted(train_dir.glob("frame_*"))
    if not frame_dirs:
        # Single-frame fallback: treat the train dir itself as frame_0000
        frame_dirs = [train_dir]

    # Apply stride and limit
    frame_dirs = frame_dirs[::frame_stride]
    if num_frames > 0:
        frame_dirs = frame_dirs[:num_frames]
    T = len(frame_dirs)

    # Load camera poses once
    c2w, hwf, bounds = load_llff_poses(str(scene_dir / "poses_bounds.npy"))

    # Map camera names to pose indices
    all_cams = sorted(scene_dir.glob("cam*.mp4"))
    cam_name_to_idx = {v.stem: i for i, v in enumerate(all_cams)}

    # Discover cameras from first frame directory
    first_frame_images = sorted(frame_dirs[0].glob("cam*.jpg"))
    if not first_frame_images:
        raise FileNotFoundError(f"No images found in {frame_dirs[0]}")

    cam_names = []
    for img_file in first_frame_images:
        cam_name = img_file.stem.split("_")[0] if "_" in img_file.stem else img_file.stem
        cam_names.append(cam_name)

    C = len(cam_names)

    # Build frame_paths[t][c] = path
    frame_paths = []
    for fdir in frame_dirs:
        paths = []
        for cam_name in cam_names:
            p = fdir / f"{cam_name}.jpg"
            if not p.exists():
                raise FileNotFoundError(f"Missing image: {p}")
            paths.append(str(p))
        frame_paths.append(paths)

    # Build camera matrices
    camtoworlds_list = []
    focals = []
    for cam_name in cam_names:
        idx = cam_name_to_idx[cam_name]
        camtoworlds_list.append(c2w[idx])
        focals.append(float(hwf[idx, 2] / downsample))

    camtoworlds = torch.tensor(np.stack(camtoworlds_list), device=device)

    # Get image dimensions from first image
    sample = np.array(Image.open(frame_paths[0][0]))
    H, W = sample.shape[:2]

    # Build intrinsics
    K = torch.zeros((C, 3, 3), device=device)
    for i, f in enumerate(focals):
        K[i, 0, 0] = f
        K[i, 1, 1] = f
        K[i, 0, 2] = W / 2.0
        K[i, 1, 2] = H / 2.0
        K[i, 2, 2] = 1.0

    # Timestamps normalized to [0, 1]
    if T == 1:
        timestamps = torch.tensor([0.5], device=device)
    else:
        timestamps = torch.linspace(0.0, 1.0, T, device=device)

    near = float(bounds[:, 0].min())
    far = float(bounds[:, 1].max())

    print(f"4D scene: {T} frames x {C} cameras at {H}x{W}, "
          f"near={near:.2f}, far={far:.2f}")
    return frame_paths, cam_names, camtoworlds, K, timestamps, near, far, H, W


class FrameCache:
    """Batch-swap cache: loads a batch of frames onto GPU, swaps periodically."""

    def __init__(self, max_size: int = 50, swap_every: int = 1000):
        self.max_size = max_size
        self.swap_every = swap_every
        self.cache = {}  # frame_idx -> (C, H, W, 3) tensor on GPU
        self.loaded_indices = []

    def maybe_swap(self, step: int, frame_paths: list, device: str = "cuda"):
        """Load a new random batch of frames every swap_every steps."""
        T = len(frame_paths)
        if step % self.swap_every == 0 or not self.loaded_indices:
            batch_size = min(self.max_size, T)
            indices = torch.randperm(T)[:batch_size].tolist()
            # Clear old batch
            self.cache.clear()
            torch.cuda.empty_cache()
            # Load new batch
            for idx in indices:
                images = []
                for path in frame_paths[idx]:
                    img = np.array(Image.open(path)).astype(np.float32) / 255.0
                    images.append(img)
                self.cache[idx] = torch.tensor(np.stack(images), device=device)
            self.loaded_indices = indices
            print(f"  [step {step}] Loaded {batch_size}/{T} frames onto GPU")

    def sample_frame_idx(self):
        """Return a random frame index from the currently loaded batch."""
        i = torch.randint(0, len(self.loaded_indices), (1,)).item()
        return self.loaded_indices[i]

    def get(self, frame_paths: list, frame_idx: int, device: str = "cuda"):
        """Get cached frame images. Must be in the current batch."""
        return self.cache[frame_idx]


# ─── 4D Gaussian Model ───────────────────────────────────────────────────────

class GaussianModel4D:
    """3D Gaussians with temporal parameters for 4D scenes."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def init_from_points(self, points: Tensor, colors: Tensor = None):
        """Initialize from a point cloud (cold start from SfM)."""
        n = points.shape[0]

        # Spatial params (same as static 3DGS)
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

        # Temporal params
        self.mu_t = torch.full((n,), 0.5, device=self.device).requires_grad_(True)
        self.s_t = torch.full((n,), math.log(0.3), device=self.device).requires_grad_(True)
        self.velocity = torch.zeros(n, 3, device=self.device).requires_grad_(True)

        # Temporal derivatives for scale, rotation, and SH
        self.d_scales = torch.zeros(n, 3, device=self.device).requires_grad_(True)
        self.d_quats = torch.zeros(n, 4, device=self.device).requires_grad_(True)
        self.d_sh0 = torch.zeros(n, 1, 3, device=self.device).requires_grad_(True)
        self.d_shN = torch.zeros(n, 15, 3, device=self.device).requires_grad_(True)

    def init_from_static_ply(self, ply_path: str):
        """Optional warm-start from a trained static PLY."""
        from plyfile import PlyData

        ply = PlyData.read(ply_path)
        v = ply["vertex"]
        n = len(v)

        means = np.stack([v["x"], v["y"], v["z"]], axis=1)
        self.means = torch.tensor(means, dtype=torch.float32,
                                  device=self.device).requires_grad_(True)

        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        self.scales = torch.tensor(scales, dtype=torch.float32,
                                   device=self.device).requires_grad_(True)

        quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
        self.quats = torch.tensor(quats, dtype=torch.float32,
                                  device=self.device).requires_grad_(True)

        opacities = np.array(v["opacity"])
        self.opacities = torch.tensor(opacities, dtype=torch.float32,
                                      device=self.device).requires_grad_(True)

        sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        self.sh0 = torch.tensor(sh0, dtype=torch.float32,
                                device=self.device).reshape(n, 1, 3).requires_grad_(True)

        shN = np.stack([v[f"f_rest_{i}"] for i in range(45)], axis=1)
        self.shN = torch.tensor(shN, dtype=torch.float32,
                                device=self.device).reshape(n, 15, 3).requires_grad_(True)

        # Initialize temporal params fresh
        self.mu_t = torch.full((n,), 0.5, device=self.device).requires_grad_(True)
        self.s_t = torch.full((n,), math.log(0.3), device=self.device).requires_grad_(True)
        self.velocity = torch.zeros(n, 3, device=self.device).requires_grad_(True)

        # Temporal derivatives — load from PLY if present, else zero-init
        if "d_scale_0" in v.data.dtype.names:
            d_scales = np.stack([v["d_scale_0"], v["d_scale_1"], v["d_scale_2"]], axis=1)
            self.d_scales = torch.tensor(d_scales, dtype=torch.float32,
                                         device=self.device).requires_grad_(True)
            d_quats = np.stack([v["d_rot_0"], v["d_rot_1"], v["d_rot_2"], v["d_rot_3"]], axis=1)
            self.d_quats = torch.tensor(d_quats, dtype=torch.float32,
                                        device=self.device).requires_grad_(True)
            d_sh0 = np.stack([v["d_f_dc_0"], v["d_f_dc_1"], v["d_f_dc_2"]], axis=1)
            self.d_sh0 = torch.tensor(d_sh0, dtype=torch.float32,
                                      device=self.device).reshape(n, 1, 3).requires_grad_(True)
            d_shN = np.stack([v[f"d_f_rest_{i}"] for i in range(45)], axis=1)
            self.d_shN = torch.tensor(d_shN, dtype=torch.float32,
                                      device=self.device).reshape(n, 15, 3).requires_grad_(True)
            print(f"Loaded {n} Gaussians from {ply_path}, with temporal derivatives")
        else:
            self.d_scales = torch.zeros(n, 3, device=self.device).requires_grad_(True)
            self.d_quats = torch.zeros(n, 4, device=self.device).requires_grad_(True)
            self.d_sh0 = torch.zeros(n, 1, 3, device=self.device).requires_grad_(True)
            self.d_shN = torch.zeros(n, 15, 3, device=self.device).requires_grad_(True)
            print(f"Loaded {n} Gaussians from {ply_path}, added temporal params")

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def all_param_names(self):
        return ["means", "scales", "quats", "opacities", "sh0", "shN",
                "mu_t", "s_t", "velocity",
                "d_scales", "d_quats", "d_sh0", "d_shN"]


# ─── Optimizer ────────────────────────────────────────────────────────────────

def make_optimizer_4d(model, lr_means, lr_scales=0.005, lr_quats=0.001,
                      lr_opacities=0.05, lr_sh0=0.0025, lr_shN=0.000125,
                      lr_mu_t=0.001, lr_s_t=0.001, lr_velocity=0.001,
                      lr_d_scales=0.001, lr_d_quats=0.0005,
                      lr_d_sh0=0.001, lr_d_shN=0.00005):
    """Create Adam optimizer with 13 parameter groups."""
    return torch.optim.Adam([
        {"params": [model.means], "lr": lr_means, "name": "means"},
        {"params": [model.scales], "lr": lr_scales, "name": "scales"},
        {"params": [model.quats], "lr": lr_quats, "name": "quats"},
        {"params": [model.opacities], "lr": lr_opacities, "name": "opacities"},
        {"params": [model.sh0], "lr": lr_sh0, "name": "sh0"},
        {"params": [model.shN], "lr": lr_shN, "name": "shN"},
        {"params": [model.mu_t], "lr": lr_mu_t, "name": "mu_t"},
        {"params": [model.s_t], "lr": lr_s_t, "name": "s_t"},
        {"params": [model.velocity], "lr": lr_velocity, "name": "velocity"},
        {"params": [model.d_scales], "lr": lr_d_scales, "name": "d_scales"},
        {"params": [model.d_quats], "lr": lr_d_quats, "name": "d_quats"},
        {"params": [model.d_sh0], "lr": lr_d_sh0, "name": "d_sh0"},
        {"params": [model.d_shN], "lr": lr_d_shN, "name": "d_shN"},
    ])


# ─── 4D Rendering ────────────────────────────────────────────────────────────

def render_4d(model, camtoworld, K, W, H, t0,
              near=0.01, far=1000.0, sh_degree=3, absgrad=True):
    """Render a single view at timestamp t0.

    Applies velocity offset and temporal opacity before gsplat rasterization.
    """
    viewmat = torch.linalg.inv(camtoworld)

    # Temporal position offset (paper: P_t0 = mu + V * (mu_t - t0))
    dt = model.mu_t - t0  # [N]
    means_t0 = model.means + model.velocity * dt.unsqueeze(-1)  # [N, 3]

    # Temporal opacity modulation
    s_t = torch.exp(model.s_t).clamp(min=1e-6)  # [N]
    temporal_weight = torch.exp(-0.5 * (dt / s_t) ** 2)  # [N]
    opacities_t0 = torch.sigmoid(model.opacities) * temporal_weight  # [N]

    # Time-varying scales (linear in log-space)
    scales_t0 = model.scales + model.d_scales * dt.unsqueeze(-1)  # [N, 3]
    scales = torch.exp(scales_t0)

    # Time-varying quaternions (linear perturbation + renormalize)
    quats_t0 = model.quats + model.d_quats * dt.unsqueeze(-1)  # [N, 4]
    quats_t0 = F.normalize(quats_t0, dim=-1)

    # Time-varying SH coefficients
    dt_sh = dt.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
    sh0_t0 = model.sh0 + model.d_sh0 * dt_sh  # [N, 1, 3]
    shN_t0 = model.shN + model.d_shN * dt_sh  # [N, 15, 3]
    sh_coeffs = torch.cat([sh0_t0, shN_t0], dim=1)

    renders, alphas, info = gsplat.rasterization(
        means=means_t0,
        quats=quats_t0,
        scales=scales,
        opacities=opacities_t0,
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
        absgrad=absgrad,
    )

    return renders[0], alphas[0], info


def render_velocity_map(model, camtoworld, K, W, H, t0,
                        near=0.01, far=1000.0):
    """Render per-pixel 2D velocity map by splatting 3D velocity as color.

    Used for flow-gradient loss computation.
    Returns: (H, W, 3) velocity map
    """
    viewmat = torch.linalg.inv(camtoworld)

    # Same temporal modulation as render_4d
    dt = model.mu_t - t0
    means_t0 = model.means + model.velocity * dt.unsqueeze(-1)
    s_t = torch.exp(model.s_t).clamp(min=1e-6)
    temporal_weight = torch.exp(-0.5 * (dt / s_t) ** 2)
    opacities_t0 = torch.sigmoid(model.opacities) * temporal_weight

    # Time-varying scales and quats (must match render_4d for consistency)
    scales_t0 = model.scales + model.d_scales * dt.unsqueeze(-1)
    scales = torch.exp(scales_t0)
    quats_t0 = model.quats + model.d_quats * dt.unsqueeze(-1)
    quats_t0 = F.normalize(quats_t0, dim=-1)

    # gsplat applies SH evaluation: output = C0 * coeff + 0.5 for degree 0.
    # Inverse-transform so the rendered output equals raw velocity.
    C0 = 0.28209479177387814
    vel_color = ((model.velocity - 0.5) / C0).unsqueeze(1)  # [N, 1, 3]

    renders, _, _ = gsplat.rasterization(
        means=means_t0,
        quats=quats_t0,
        scales=scales,
        opacities=opacities_t0,
        colors=vel_color,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        near_plane=near,
        far_plane=far,
        sh_degree=0,
        render_mode="RGB",
        rasterize_mode="classic",
        absgrad=False,
    )

    return renders[0]  # (H, W, 3)


# ─── Loss Functions ──────────────────────────────────────────────────────────

def sobel_edges(img):
    """Compute edge magnitude using Sobel filters.

    Args:
        img: (H, W, C) tensor

    Returns:
        edge_mag: (H, W) tensor with edge magnitudes normalized to [0, 1]
    """
    # Convert to (1, C, H, W) for conv2d
    x = img.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    sobel_x = sobel_x.expand(C, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1, groups=C)
    gy = F.conv2d(x, sobel_y, padding=1, groups=C)

    # Magnitude across channels
    edge_mag = (gx.pow(2) + gy.pow(2)).sum(dim=1).add(1e-8).sqrt()  # (1, H, W)
    edge_mag = edge_mag.squeeze(0)  # (H, W)

    # Normalize to [0, 1]
    emax = edge_mag.max()
    if emax > 0:
        edge_mag = edge_mag / emax

    return edge_mag


def flow_gradient_loss(vel_map, rendered_img):
    """Flow-gradient loss: penalize velocity gradients where no image edges exist.

    L_fg = mean(||∇M|| * (1 - ||∇I||))

    Args:
        vel_map: (H, W, 3) rendered velocity map
        rendered_img: (H, W, 3) rendered RGB image

    Returns:
        scalar loss
    """
    # Flow magnitude per pixel (with epsilon for numerical stability, per paper Eq. 17)
    flow_mag = (vel_map.pow(2).sum(dim=2) + 1e-8).sqrt()  # (H, W)

    # Sobel edges of flow magnitude
    flow_edges = sobel_edges(flow_mag.unsqueeze(-1))  # (H, W)

    # Sobel edges of rendered image
    img_edges = sobel_edges(rendered_img.detach())  # (H, W)

    # Penalize flow gradients where image is smooth
    loss = (flow_edges * (1.0 - img_edges)).mean()
    return loss


# ─── Densification ────────────────────────────────────────────────────────────

def densify_and_prune_4d(model, grad_accum, grad_count,
                          grad_accum_t, grad_count_t,
                          grad_threshold=0.0002, grad_threshold_t=0.001,
                          min_opacity=0.005, max_scale=0.05,
                          scene_extent=1.0, max_gaussians=3_000_000):
    """Densify and prune with spatial + temporal splitting.

    Returns: (n_clone, n_split, n_temporal_split, n_prune)
    """
    avg_grad = grad_accum / grad_count.clamp(min=1)
    avg_grad[avg_grad.isnan()] = 0.0

    avg_grad_t = grad_accum_t / grad_count_t.clamp(min=1)
    avg_grad_t[avg_grad_t.isnan()] = 0.0

    scales = torch.exp(model.scales)
    max_scale_per = scales.max(dim=1).values
    high_grad = avg_grad.squeeze() > grad_threshold

    # Spatial clone (small Gaussians with high gradient)
    clone_mask = high_grad & (max_scale_per < max_scale * scene_extent)
    n_clone = clone_mask.sum().item()

    # Spatial split (large Gaussians with high gradient)
    split_mask = high_grad & (max_scale_per >= max_scale * scene_extent)
    n_split = split_mask.sum().item()

    # Temporal split (high temporal gradient, not already spatially split)
    high_grad_t = avg_grad_t.squeeze() > grad_threshold_t
    temporal_split_mask = high_grad_t & ~split_mask & ~clone_mask
    n_temporal_split = temporal_split_mask.sum().item()

    # Cap growth
    total_after = model.num_gaussians + n_clone + n_split + n_temporal_split
    if total_after > max_gaussians:
        n_clone = n_split = n_temporal_split = 0
        clone_mask[:] = False
        split_mask[:] = False
        temporal_split_mask[:] = False

    param_names = model.all_param_names
    new_params = {k: [] for k in param_names}

    # --- Clone ---
    if n_clone > 0:
        for attr in param_names:
            new_params[attr].append(getattr(model, attr)[clone_mask].detach().clone())

    # --- Spatial split ---
    if n_split > 0:
        split_scales = model.scales[split_mask].detach().clone() - math.log(1.6)
        stds = torch.exp(model.scales[split_mask]).max(dim=1).values
        split_means = model.means[split_mask].detach().clone()
        offset = torch.randn_like(split_means) * stds[:, None]

        for child_offset in [offset, -offset]:
            new_params["means"].append(split_means + child_offset)
            new_params["scales"].append(split_scales)
            for attr in ["quats", "opacities", "sh0", "shN",
                          "mu_t", "s_t", "velocity",
                          "d_scales", "d_quats", "d_sh0", "d_shN"]:
                new_params[attr].append(
                    getattr(model, attr)[split_mask].detach().clone()
                )

    # --- Temporal split ---
    if n_temporal_split > 0:
        s_t_vals = torch.exp(model.s_t[temporal_split_mask]).detach()
        mu_t_vals = model.mu_t[temporal_split_mask].detach()
        new_s_t = (model.s_t[temporal_split_mask].detach().clone()
                   - math.log(2.0))  # halve lifespan

        for sign in [0.5, -0.5]:
            new_mu_t = mu_t_vals + sign * s_t_vals
            new_params["mu_t"].append(new_mu_t)
            new_params["s_t"].append(new_s_t.clone())
            for attr in ["means", "scales", "quats", "opacities",
                          "sh0", "shN", "velocity",
                          "d_scales", "d_quats", "d_sh0", "d_shN"]:
                new_params[attr].append(
                    getattr(model, attr)[temporal_split_mask].detach().clone()
                )

    # --- Prune ---
    opacities = torch.sigmoid(model.opacities)
    prune_mask = (opacities < min_opacity).squeeze()
    prune_mask = prune_mask | (max_scale_per > max_scale * scene_extent * 5)
    # Prune temporally dead Gaussians
    prune_mask = prune_mask | (torch.exp(model.s_t) < 1e-4)
    # Remove originals that were split
    if n_split > 0:
        prune_mask = prune_mask | split_mask
    if n_temporal_split > 0:
        prune_mask = prune_mask | temporal_split_mask

    keep_mask = ~prune_mask
    # Count only "natural" pruning (exclude split originals that were replaced)
    n_prune = prune_mask.sum().item() - n_split - n_temporal_split

    # Rebuild all parameters
    for attr in param_names:
        old = getattr(model, attr)[keep_mask].detach()
        parts = [old] + new_params[attr]
        setattr(model, attr,
                torch.cat(parts).detach().clone().requires_grad_(True))

    return n_clone, n_split, n_temporal_split, n_prune


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_4d(model, scene_dir, frames_base, downsample, near, far,
                num_eval_frames=10, device="cuda", save_dir=None):
    """Evaluate on held-out test camera across multiple timesteps.

    Returns: avg_psnr, avg_ssim
    """
    scene_dir = Path(scene_dir)
    test_dir = scene_dir / frames_base / "test"

    # Discover test frame directories
    test_frame_dirs = sorted(test_dir.glob("frame_*"))
    if not test_frame_dirs:
        # Single-frame fallback
        test_frame_dirs = [test_dir]

    # Subsample evenly for evaluation
    if len(test_frame_dirs) > num_eval_frames:
        indices = np.linspace(0, len(test_frame_dirs) - 1,
                              num_eval_frames, dtype=int)
        test_frame_dirs = [test_frame_dirs[i] for i in indices]

    T_eval = len(test_frame_dirs)

    # Load camera poses
    c2w, hwf, bounds = load_llff_poses(str(scene_dir / "poses_bounds.npy"))
    all_cams = sorted(scene_dir.glob("cam*.mp4"))
    cam_name_to_idx = {v.stem: i for i, v in enumerate(all_cams)}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for t_idx, fdir in enumerate(test_frame_dirs):
        # Timestamp
        if T_eval == 1:
            t0 = 0.5
        else:
            t0 = t_idx / (T_eval - 1)

        image_files = sorted(fdir.glob("cam*.jpg"))
        for img_file in image_files:
            cam_name = (img_file.stem.split("_")[0]
                        if "_" in img_file.stem else img_file.stem)
            idx = cam_name_to_idx[cam_name]

            # Load GT
            gt = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            gt = torch.tensor(gt, device=device)
            H, W = gt.shape[:2]

            # Camera
            c2w_i = torch.tensor(c2w[idx], device=device)
            f = float(hwf[idx, 2] / downsample)
            K_i = torch.zeros(3, 3, device=device)
            K_i[0, 0] = f
            K_i[1, 1] = f
            K_i[0, 2] = W / 2.0
            K_i[1, 2] = H / 2.0
            K_i[2, 2] = 1.0

            # Render
            rendered, _, _ = render_4d(model, c2w_i, K_i, W, H, t0,
                                       near=near, far=far, absgrad=False)
            rendered = rendered.clamp(0, 1)

            mse = F.mse_loss(rendered, gt)
            psnr = -10.0 * math.log10(mse.item())
            ssim = 1.0 - ssim_loss(rendered, gt).item()

            total_psnr += psnr
            total_ssim += ssim
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
    return avg_psnr, avg_ssim


# ─── PLY Export ───────────────────────────────────────────────────────────────

def save_ply_4d(model, path):
    """Save 4D Gaussian model as .ply with temporal parameters."""
    from plyfile import PlyData, PlyElement

    means = model.means.detach().cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    quats = model.quats.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()
    sh0 = model.sh0.detach().cpu().numpy().reshape(-1, 3)
    shN = model.shN.detach().cpu().numpy().reshape(-1, 45)
    mu_t = model.mu_t.detach().cpu().numpy()
    s_t = model.s_t.detach().cpu().numpy()
    velocity = model.velocity.detach().cpu().numpy()
    d_scales = model.d_scales.detach().cpu().numpy()
    d_quats = model.d_quats.detach().cpu().numpy()
    d_sh0 = model.d_sh0.detach().cpu().numpy().reshape(-1, 3)
    d_shN = model.d_shN.detach().cpu().numpy().reshape(-1, 45)

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
        # Temporal params
        ("mu_t", "f4"), ("s_t", "f4"),
        ("velocity_0", "f4"), ("velocity_1", "f4"), ("velocity_2", "f4"),
        # Temporal derivatives
        ("d_scale_0", "f4"), ("d_scale_1", "f4"), ("d_scale_2", "f4"),
        ("d_rot_0", "f4"), ("d_rot_1", "f4"), ("d_rot_2", "f4"), ("d_rot_3", "f4"),
        ("d_f_dc_0", "f4"), ("d_f_dc_1", "f4"), ("d_f_dc_2", "f4"),
    ]
    for i in range(45):
        dtype.append((f"d_f_rest_{i}", "f4"))

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
    elements["mu_t"] = mu_t
    elements["s_t"] = s_t
    elements["velocity_0"] = velocity[:, 0]
    elements["velocity_1"] = velocity[:, 1]
    elements["velocity_2"] = velocity[:, 2]
    elements["d_scale_0"] = d_scales[:, 0]
    elements["d_scale_1"] = d_scales[:, 1]
    elements["d_scale_2"] = d_scales[:, 2]
    elements["d_rot_0"] = d_quats[:, 0]
    elements["d_rot_1"] = d_quats[:, 1]
    elements["d_rot_2"] = d_quats[:, 2]
    elements["d_rot_3"] = d_quats[:, 3]
    elements["d_f_dc_0"] = d_sh0[:, 0]
    elements["d_f_dc_1"] = d_sh0[:, 1]
    elements["d_f_dc_2"] = d_sh0[:, 2]
    for i in range(45):
        elements[f"d_f_rest_{i}"] = d_shN[:, i]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(str(path))


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_4d(args):
    device = "cuda"
    torch.manual_seed(42)

    # Load scene metadata (lazy — images loaded on demand)
    print("Loading 4D scene metadata...")
    (frame_paths, cam_names, camtoworlds, K, timestamps,
     near, far, H, W) = load_scene_4d(
        args.scene_dir, frames_base=args.frames_base,
        downsample=args.downsample, num_frames=args.num_frames,
        frame_stride=args.frame_stride, device=device,
    )
    T = len(timestamps)
    C = len(cam_names)

    # Initialize model
    model = GaussianModel4D(device=device)

    if args.init_ply:
        model.init_from_static_ply(args.init_ply)
    else:
        # Cold start from SfM points (recommended)
        sfm_dir = Path(args.scene_dir) / "colmap"
        if args.sfm_dir:
            sfm_dir = Path(args.sfm_dir)
        points, colors = init_gaussians_from_sfm(sfm_dir, device)
        if points is not None:
            print(f"Initialized {len(points)} Gaussians from SfM points")
        else:
            print(f"No SfM points at {sfm_dir}, using random init")
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

    # Frame cache with batch swapping
    cache = FrameCache(max_size=args.frame_cache_size,
                       swap_every=args.cache_swap_every)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting 4D training for {args.num_steps} steps...")
    print(f"  {T} frames x {C} cameras = {T * C} total training pairs")
    print(f"  Frame cache: {args.frame_cache_size} frames on GPU, "
          f"swap every {args.cache_swap_every} steps")
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

        # Swap frame batch if needed
        cache.maybe_swap(step, frame_paths, device)

        # Random (frame, camera) pair from loaded batch
        frame_idx = cache.sample_frame_idx()
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
            if not torch.isnan(l_flow) and not torch.isinf(l_flow):
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
    print(f"\n4D training complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    # Evaluate
    print("Evaluating on test camera across timesteps...")
    test_psnr, test_ssim = evaluate_4d(
        model, args.scene_dir, args.frames_base, args.downsample,
        near, far, num_eval_frames=min(T, 10), device=device,
        save_dir=output_dir)
    print(f"Test PSNR: {test_psnr:.2f} dB | Test SSIM: {test_ssim:.4f}")

    # Save final
    final_path = output_dir / "point_cloud_4d_final.ply"
    save_ply_4d(model, final_path)
    print(f"Saved to {final_path} ({model.num_gaussians:,} Gaussians)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disentangled 4D Gaussian Splatting training")
    # Data
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/4DGS/outputs/4d_baseline")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--frames_base", type=str, default="frames")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Max frames to use (0 = all)")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_cache_size", type=int, default=60,
                        help="Number of frames to keep on GPU at once")
    parser.add_argument("--cache_swap_every", type=int, default=1000,
                        help="Swap frame batch every N steps")
    # Init
    parser.add_argument("--init_ply", type=str, default=None,
                        help="Optional: warm-start from trained static PLY")
    parser.add_argument("--sfm_dir", type=str, default=None)
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

    train_4d(args)
