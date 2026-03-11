"""Differentiable volume renderer with hierarchical ray marching.

Supports both dense WaveletVolume (for low-res training) and
TiledWaveletVolume (for high-res 1024³+ training).
"""

import torch

from config import Config
from rays import stratified_sampling, importance_sampling
from wavelet_volume import WaveletVolume


def volume_render_weights(
    density: torch.Tensor,
    t_vals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rendering weights from density and distances.

    Args:
        density: (N, S) volume density at each sample
        t_vals: (N, S) distances along ray

    Returns:
        weights: (N, S) rendering weights
        transmittance: (N, S) accumulated transmittance
    """
    # Distances between consecutive samples
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N, S-1)
    # Last delta: extend to infinity (large number)
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)

    # Alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-density * deltas)

    # Transmittance: T_i = prod(1 - alpha_j for j < i)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]

    weights = transmittance * alpha

    return weights, transmittance


def _generate_rays(pose, H, W, focal, device):
    """Generate rays for a full image from a camera pose.

    Returns:
        rays_o: (H*W, 3)
        rays_d: (H*W, 3)
    """
    ii, jj = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy",
    )
    dirs = torch.stack([
        (ii - W * 0.5) / focal,
        -(jj - H * 0.5) / focal,
        -torch.ones_like(ii),
    ], dim=-1)

    rays_d = (dirs[..., None, :] * pose[:3, :3]).sum(-1)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    rays_o = pose[:3, 3].expand_as(rays_d)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def render_rays(
    model,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    config: Config,
    volume_fine: torch.Tensor | None = None,
    volume_coarse: torch.Tensor | None = None,
    base_volume: torch.Tensor | None = None,
    perturb: bool = True,
) -> dict:
    """Render rays through the wavelet volume.

    Uses hierarchical sampling: coarse pass finds surfaces,
    fine pass concentrates samples there.

    Supports both dense WaveletVolume and TiledWaveletVolume:
    - Dense: pass volume_fine and volume_coarse (pre-reconstructed).
    - Tiled: pass base_volume (coarse levels). Fine pass uses query_tiled().

    Args:
        model: WaveletVolume or TiledWaveletVolume
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions (normalized)
        config: hyperparameters
        volume_fine: pre-reconstructed fine volume (dense path only)
        volume_coarse: pre-reconstructed coarse volume (dense path only)
        base_volume: pre-reconstructed base volume (tiled path only)
        perturb: whether to jitter samples

    Returns:
        dict with 'rgb', 'rgb_coarse', 'depth', 'opacity', 'weights_coarse'
    """
    from tiled_wavelet_volume import TiledWaveletVolume

    N = rays_o.shape[0]
    is_tiled = isinstance(model, TiledWaveletVolume)

    # When explicit volumes are passed, always use the dense query path.
    # This allows TiledWaveletVolume to be used in Phase 1 (base-only
    # training) without tiled reconstruction.
    use_tiled_path = is_tiled and volume_fine is None and volume_coarse is None

    # --- Coarse pass ---
    if use_tiled_path:
        # Tiled path: coarse pass uses the base volume (always small)
        if base_volume is None:
            base_volume = model.reconstruct_base()
        coarse_volume = base_volume
    else:
        # Dense path (also used for TiledWaveletVolume Phase 1)
        if volume_coarse is None and volume_fine is None:
            coarse_level = max(-1, model.decomp_levels - 2)
            volume_fine, volume_coarse = model.reconstruct_pair(
                model.decomp_levels - 1, coarse_level
            )
        elif volume_coarse is None:
            coarse_level = max(-1, model.decomp_levels - 2)
            volume_coarse = model.reconstruct(max_level=coarse_level)
        coarse_volume = volume_coarse

    pts_c, t_c = stratified_sampling(
        rays_o, rays_d, config.near, config.far,
        config.coarse_samples, perturb=perturb,
    )

    pts_flat = pts_c.reshape(-1, 3)
    features_c = model.query(pts_flat, coarse_volume)
    view_dirs_c = rays_d[:, None, :].expand_as(pts_c).reshape(-1, 3)
    density_c, rgb_c = model.decode(features_c, view_dirs_c)

    density_c = density_c.view(N, config.coarse_samples)
    rgb_c = rgb_c.view(N, config.coarse_samples, 3)

    weights_c, _ = volume_render_weights(density_c, t_c)
    rgb_coarse = (weights_c.unsqueeze(-1) * rgb_c).sum(dim=1)
    opacity_coarse = weights_c.sum(dim=1)

    # Background compositing (white bg for NeRF Synthetic)
    if config.white_background:
        rgb_coarse = rgb_coarse + (1.0 - opacity_coarse.unsqueeze(-1))

    # --- Fine pass ---
    pts_f, t_f = importance_sampling(
        rays_o, rays_d, t_c, weights_c.detach(),
        config.fine_samples, perturb=perturb,
    )

    total_samples = t_f.shape[1]
    pts_flat = pts_f.reshape(-1, 3)

    if use_tiled_path:
        # Tiled path: query full-resolution via tiled reconstruction
        features_f = model.query_tiled(pts_flat, base_volume)
    else:
        # Dense path
        if volume_fine is None:
            volume_fine = model.reconstruct()
        features_f = model.query(pts_flat, volume_fine)

    view_dirs_f = rays_d[:, None, :].expand(-1, total_samples, -1).reshape(-1, 3)
    density_f, rgb_f = model.decode(features_f, view_dirs_f)

    density_f = density_f.view(N, total_samples)
    rgb_f = rgb_f.view(N, total_samples, 3)

    weights_f, _ = volume_render_weights(density_f, t_f)

    rgb_fine = (weights_f.unsqueeze(-1) * rgb_f).sum(dim=1)
    depth = (weights_f * t_f).sum(dim=1)
    opacity = weights_f.sum(dim=1)

    # Background compositing (white bg for NeRF Synthetic)
    if config.white_background:
        rgb_fine = rgb_fine + (1.0 - opacity.unsqueeze(-1))

    return {
        "rgb": rgb_fine,           # (N, 3)
        "rgb_coarse": rgb_coarse,  # (N, 3)
        "depth": depth,            # (N,)
        "opacity": opacity,        # (N,)
        "density_fine": density_f,  # (N, S) for sparsity loss
        "weights_coarse": weights_c,
    }


def render_image(
    model,
    pose: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    config: Config,
    chunk: int = 4096,
    max_level: int | None = None,
) -> dict:
    """Render a full image by chunking rays.

    Supports both dense WaveletVolume and TiledWaveletVolume.

    Args:
        model: WaveletVolume or TiledWaveletVolume
        pose: (4, 4) camera-to-world matrix
        H, W: image dimensions
        focal: focal length in pixels
        config: hyperparameters
        chunk: number of rays per chunk
        max_level: if set, reconstruct volume only up to this detail level
                   (for LoD visualization). None = full resolution.

    Returns:
        dict with 'rgb' (H, W, 3), 'depth' (H, W), 'opacity' (H, W)
    """
    from tiled_wavelet_volume import TiledWaveletVolume

    rays_o, rays_d = _generate_rays(pose, H, W, focal, pose.device)
    is_tiled = isinstance(model, TiledWaveletVolume)

    # Pre-reconstruct volumes that stay constant across chunks
    with torch.no_grad():
        if is_tiled and max_level is None:
            # Full tiled path: base volume + tiled fine pass
            base_volume = model.reconstruct_base()
            volume_fine = None
            volume_coarse = None
        elif is_tiled:
            # TiledWaveletVolume with explicit max_level (Phase 1 / LoD):
            # use dense path with base volumes at different levels
            fine_lvl = min(max_level, model.base_level)
            coarse_lvl = max(-1, fine_lvl - 1)
            volume_fine = model.reconstruct_base(max_level=fine_lvl)
            volume_coarse = model.reconstruct_base(max_level=coarse_lvl)
            base_volume = None
        else:
            fine_lvl = max_level if max_level is not None else model.decomp_levels - 1
            coarse_lvl = max(-1, fine_lvl - 1)
            volume_fine, volume_coarse = model.reconstruct_pair(fine_lvl, coarse_lvl)
            base_volume = None

    # Render in chunks
    all_rgb = []
    all_depth = []
    all_opacity = []

    for ci in range(0, rays_o.shape[0], chunk):
        ro = rays_o[ci:ci + chunk]
        rd = rays_d[ci:ci + chunk]
        with torch.no_grad():
            result = render_rays(
                model, ro, rd, config,
                volume_fine=volume_fine,
                volume_coarse=volume_coarse,
                base_volume=base_volume,
                perturb=False,
            )
        all_rgb.append(result["rgb"])
        all_depth.append(result["depth"])
        all_opacity.append(result["opacity"])

    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    depth = torch.cat(all_depth, dim=0).reshape(H, W)
    opacity = torch.cat(all_opacity, dim=0).reshape(H, W)

    return {"rgb": rgb, "depth": depth, "opacity": opacity}
