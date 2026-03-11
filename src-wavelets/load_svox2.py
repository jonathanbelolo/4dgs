"""Load svox2 (Plenoxels) checkpoint and extract dense voxel grid channels.

svox2 stores voxels sparsely: only occupied voxels have density/SH data.
The full 28ch × 1030³ grid is ~114 GB — too large to materialize at once.
Instead, we load sparse data (~5 GB) and expand channels on demand in
batches for the DWT conversion.
"""

import numpy as np
import torch


def load_svox2_sparse(ckpt_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load raw sparse svox2 data without expanding to dense.

    Args:
        ckpt_path: Path to svox2 .npz checkpoint file.

    Returns:
        links: (R, R, R) int32 tensor. -1 = empty, >= 0 = index into data.
        density_data: (N, 1) float32 density values.
        sh_data: (N, 27) float32 SH coefficients.
        meta: Dict with radius, center, resolution, n_sparse.
    """
    data = dict(np.load(ckpt_path))

    links = torch.from_numpy(data["links"])                 # (R, R, R) int32
    density_data = torch.from_numpy(data["density_data"]).float()  # (N, 1) → fp32
    sh_data = torch.from_numpy(data["sh_data"]).float()     # (N, 27) fp16 → fp32

    R = links.shape[0]
    N = density_data.shape[0]

    print(f"svox2 grid: {R}³, {N:,} sparse voxels "
          f"({100.0 * N / R**3:.1f}% occupied)")

    meta = {
        "radius": data["radius"].tolist(),
        "center": data["center"].tolist(),
        "resolution": R,
        "n_sparse": N,
    }

    return links, density_data, sh_data, meta


def expand_channels_to_dense(
    links: torch.Tensor,
    density_data: torch.Tensor,
    sh_data: torch.Tensor,
    channels: list[int],
) -> torch.Tensor:
    """Expand specified channels from sparse to dense grid.

    Channel 0 = density, channels 1-27 = SH coefficients.

    Args:
        links: (R, R, R) int32 sparse-to-dense mapping.
        density_data: (N, 1) density values.
        sh_data: (N, 27) SH coefficients.
        channels: List of channel indices to expand (e.g. [0,1,2,3]).

    Returns:
        grid: (1, len(channels), R, R, R) float32 dense grid.
    """
    R = links.shape[0]
    C = len(channels)
    mask = links >= 0
    indices = links[mask].long()

    grid = torch.zeros(1, C, R, R, R, dtype=torch.float32)
    for i, ch in enumerate(channels):
        if ch == 0:
            grid[0, i][mask] = density_data[indices, 0]
        else:
            grid[0, i][mask] = sh_data[indices, ch - 1]

    return grid


def get_occupancy_from_sparse(
    links: torch.Tensor,
    density_data: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Get binary occupancy mask from svox2 sparse data.

    Args:
        links: (R, R, R) int32 sparse-to-dense mapping.
        density_data: (N, 1) density values.
        threshold: Density threshold for occupancy.

    Returns:
        occupancy: (R, R, R) boolean tensor.
    """
    R = links.shape[0]
    mask = links >= 0
    if threshold <= 0.0:
        # Any non-empty voxel is occupied
        return mask

    # Threshold on density values
    occupancy = torch.zeros(R, R, R, dtype=torch.bool)
    indices = links[mask].long()
    density_vals = density_data[indices, 0]
    occupied_voxels = density_vals > threshold
    # mask.nonzero gives positions; filter by threshold
    positions = mask.nonzero(as_tuple=False)  # (N_occ, 3)
    occupied_positions = positions[occupied_voxels]
    occupancy[
        occupied_positions[:, 0],
        occupied_positions[:, 1],
        occupied_positions[:, 2],
    ] = True
    return occupancy
