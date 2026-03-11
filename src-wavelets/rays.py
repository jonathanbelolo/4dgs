"""Ray sampling utilities for volume rendering."""

import torch


def stratified_sampling(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified sampling along rays.

    Divides [near, far] into num_samples bins and samples one point
    uniformly within each bin.

    Args:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near: near plane distance
        far: far plane distance
        num_samples: number of samples per ray
        perturb: if True, jitter samples within each bin

    Returns:
        pts: (N, num_samples, 3) sample positions
        t_vals: (N, num_samples) distances along ray
    """
    N = rays_o.shape[0]
    device = rays_o.device

    # Bin edges: num_samples + 1 edges defining num_samples bins
    bin_edges = torch.linspace(near, far, num_samples + 1, device=device)  # (S+1,)
    lower = bin_edges[:-1].unsqueeze(0).expand(N, -1)  # (N, S)
    upper = bin_edges[1:].unsqueeze(0).expand(N, -1)   # (N, S)

    if perturb:
        # Uniform random within each bin
        t_vals = lower + (upper - lower) * torch.rand(N, num_samples, device=device)
    else:
        # Bin centers
        t_vals = 0.5 * (lower + upper)

    # Sample positions
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]  # (N, S, 3)

    return pts, t_vals


def importance_sampling(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_vals_coarse: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Importance sampling based on coarse pass weights.

    Samples more points where the coarse pass found high density.

    Args:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        t_vals_coarse: (N, S_coarse) coarse sample distances
        weights: (N, S_coarse) rendering weights from coarse pass
        num_samples: number of fine samples to add
        perturb: if True, add noise

    Returns:
        pts_combined: (N, S_coarse + num_samples, 3) all sample positions, sorted
        t_vals_combined: (N, S_coarse + num_samples) all distances, sorted
    """
    # Prevent division by zero
    weights = weights + 1e-5
    # Normalize to PDF
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    # CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, S+1)

    # Sample from CDF via inverse transform sampling
    if perturb:
        u = torch.rand(rays_o.shape[0], num_samples, device=rays_o.device)
    else:
        u = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)
        u = u.unsqueeze(0).expand(rays_o.shape[0], -1)

    # Find bins in CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    t_below = torch.gather(t_vals_coarse, 1, below.clamp(max=t_vals_coarse.shape[-1] - 1))
    t_above = torch.gather(t_vals_coarse, 1, above.clamp(max=t_vals_coarse.shape[-1] - 1))

    # Linear interpolation
    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine = t_below + (u - cdf_below) / denom * (t_above - t_below)

    # Combine coarse and fine, sort by distance
    t_combined, _ = torch.sort(torch.cat([t_vals_coarse, t_fine], dim=-1), dim=-1)

    # Compute positions
    pts_combined = rays_o[:, None, :] + rays_d[:, None, :] * t_combined[..., None]

    return pts_combined, t_combined
