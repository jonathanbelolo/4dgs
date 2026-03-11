"""Spatial occupancy estimation from a trained coarse model.

Determines which regions of the volume contain scene content by evaluating
density on a regular grid. The occupancy mask is used to allocate sparse
wavelet detail coefficients only where needed.
"""

import torch
import torch.nn.functional as F

from wavelet_volume import WaveletVolume


def estimate_occupancy(
    model: WaveletVolume,
    grid_resolution: int = 128,
    density_threshold: float = 0.1,
    dilate_kernel: int = 3,
) -> torch.Tensor:
    """Estimate spatial occupancy from a trained coarse model.

    Reconstructs the volume, evaluates density on a regular grid, and returns
    a boolean occupancy volume.

    Args:
        model: Trained WaveletVolume (can be any resolution).
        grid_resolution: Resolution of the occupancy grid.
        density_threshold: Density values above this are considered occupied.
        dilate_kernel: Dilation kernel size (adds safety margin at boundaries).

    Returns:
        occupancy: (grid_resolution, grid_resolution, grid_resolution) boolean
                   tensor on the same device as the model.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Reconstruct at whatever level the model supports
        from tiled_wavelet_volume import TiledWaveletVolume
        if isinstance(model, TiledWaveletVolume):
            volume = model.reconstruct_base()
        else:
            volume = model.reconstruct()

        # Build regular grid of query points
        coords = torch.linspace(
            -model.scene_bound, model.scene_bound, grid_resolution,
            device=device,
        )
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        # Query density (use model.decode for consistency with activation)
        features = model.query(xyz, volume)
        raw_density = features[:, 0]
        density = F.relu(raw_density)

        occupancy = density.reshape(
            grid_resolution, grid_resolution, grid_resolution,
        ) > density_threshold

    # Dilate to add safety margin
    if dilate_kernel > 0:
        occupancy = dilate_3d(occupancy, dilate_kernel)

    return occupancy


def dilate_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """3D binary dilation using max pooling.

    Args:
        mask: (D, H, W) boolean tensor.
        kernel_size: Dilation kernel size (odd integer).

    Returns:
        Dilated (D, H, W) boolean tensor.
    """
    padding = kernel_size // 2
    dilated = F.max_pool3d(
        mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return dilated.squeeze(0).squeeze(0) > 0


def occupancy_to_tile_mask(
    occupancy: torch.Tensor,
    tiles_per_axis: int,
) -> torch.Tensor:
    """Convert a dense occupancy grid to a coarser tile-level mask.

    A tile is marked occupied if ANY voxel within its spatial extent is occupied.

    Args:
        occupancy: (R, R, R) boolean dense occupancy grid.
        tiles_per_axis: Number of tiles per axis at the target detail level.

    Returns:
        tile_mask: (T, T, T) boolean tensor.
    """
    R = occupancy.shape[0]

    if tiles_per_axis > R:
        # Tile grid is finer than occupancy grid — upsample occupancy
        occupancy_up = F.interpolate(
            occupancy.float().unsqueeze(0).unsqueeze(0),
            size=(tiles_per_axis, tiles_per_axis, tiles_per_axis),
            mode='nearest',
        ).squeeze(0).squeeze(0)
        return occupancy_up > 0

    # Pool: any occupied voxel in the block → tile is occupied.
    # adaptive_max_pool3d handles non-divisible sizes correctly
    # (unlike max_pool3d which drops remainder voxels).
    tile_mask = F.adaptive_max_pool3d(
        occupancy.float().unsqueeze(0).unsqueeze(0),
        output_size=(tiles_per_axis, tiles_per_axis, tiles_per_axis),
    ).squeeze(0).squeeze(0) > 0

    return tile_mask


def compute_occupancy_stats(
    occupancy: torch.Tensor,
    level_sizes: list[int],
    block_size: int,
    num_channels: int,
) -> list[dict]:
    """Compute per-level occupancy and memory statistics.

    Args:
        occupancy: (R, R, R) boolean occupancy grid.
        level_sizes: Spatial size of detail coefficients at each level.
        block_size: Tile size for sparse storage.
        num_channels: Feature channels per voxel.

    Returns:
        List of dicts with per-level statistics.
    """
    stats = []
    for level, size in enumerate(level_sizes):
        tiles_per_axis = max(1, (size + block_size - 1) // block_size)
        tile_mask = occupancy_to_tile_mask(occupancy, tiles_per_axis)
        n_occupied = tile_mask.sum().item()
        n_total = tile_mask.numel()

        # Memory: 7 subbands × C channels × block³ per occupied tile
        params_per_tile = 7 * num_channels * block_size ** 3
        total_params = n_occupied * params_per_tile
        dense_params = 7 * num_channels * size ** 3

        stats.append({
            'level': level,
            'detail_size': size,
            'tiles_per_axis': tiles_per_axis,
            'occupied': n_occupied,
            'total_tiles': n_total,
            'occupancy_pct': 100.0 * n_occupied / n_total,
            'sparse_params_M': total_params / 1e6,
            'dense_params_M': dense_params / 1e6,
            'savings_x': dense_params / max(total_params, 1),
            'sparse_mb_fp16': total_params * 2 / 1024 ** 2,
            'dense_mb_fp16': dense_params * 2 / 1024 ** 2,
        })

    return stats
