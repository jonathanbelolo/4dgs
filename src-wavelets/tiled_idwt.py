"""Tiled inverse wavelet transform for memory-efficient high-resolution reconstruction.

Reconstructs spatial tiles of the output volume without materializing the full
dense volume. Each tile is produced by cascading single-level IDWTs on
coefficient subregions with halo padding to handle filter boundary effects.

For a 2048³ target volume, a 64³ output tile only requires a ~20³ patch from
the base volume — making reconstruction of arbitrarily large volumes feasible
on a single GPU.
"""

import pywt
import torch
import ptwt

from wavelet_volume import SUBBAND_KEYS


def _get_rec_halo(wavelet: str) -> int:
    """Half-width of the reconstruction filter halo.

    For bior4.4: reconstruction filter length = 10 → halo = 4.
    The halo is the number of boundary-affected output samples on each side
    when running IDWT on a subregion of coefficients.
    """
    w = pywt.Wavelet(wavelet)
    return (w.rec_len - 1) // 2


def _idwt_output_size(input_size: int, rec_len: int) -> int:
    """Exact IDWT output size for a given input size.

    For bior4.4 (rec_len=10): output = 2*input - 8.
    General formula: output = 2*input - (rec_len - 2).
    """
    return 2 * input_size - (rec_len - 2)


def compute_input_region(
    output_start: int,
    output_end: int,
    halo: int,
    parent_size: int,
    rec_len: int,
) -> tuple[int, int, int, int]:
    """Compute the required input region for a single-level IDWT.

    Given a desired output range [output_start, output_end) in the IDWT output,
    determine the input range needed from the parent (half-resolution) level,
    accounting for halo overlap and IDWT output shrinkage.

    The IDWT on input of size d produces output of size 2*d - (rec_len - 2).
    Output sample k corresponds to full position 2*input_start + k.
    The valid interior (matching full reconstruction) is [halo, output_size - halo).

    Args:
        output_start: Start index in the full output volume (inclusive).
        output_end: End index in the full output volume (exclusive).
        halo: Filter overlap in output-space voxels.
        parent_size: Size of the parent coefficient tensor along this axis.
        rec_len: Reconstruction filter length.

    Returns:
        (input_start, input_end, output_crop_start, output_crop_end):
            input_start/end: range to extract from parent coefficients (clamped).
            output_crop_start/end: where the desired output sits within the
                                   IDWT output of the subregion.
    """
    shrinkage = rec_len - 2  # = 2 * halo for typical wavelets

    # Required input range to produce valid output covering [output_start, output_end):
    # Valid region starts at: 2*in_start + halo  (must be <= output_start)
    # Valid region ends at:   2*in_start + 2*d - shrinkage - halo  (must be >= output_end)
    #
    # Solving: in_start <= (output_start - halo) / 2
    #          in_end >= (output_end + shrinkage + halo) / 2
    input_start = max(0, (output_start - halo) // 2)
    input_end = min(parent_size, -(-(output_end + shrinkage + halo) // 2))  # ceiling division

    # After IDWT on input[input_start:input_end], output sample k corresponds
    # to full position 2*input_start + k.
    output_crop_start = output_start - 2 * input_start
    output_crop_end = output_crop_start + (output_end - output_start)

    return input_start, input_end, output_crop_start, output_crop_end


def trace_tile_regions(
    tile_start: tuple[int, int, int],
    tile_size: int,
    level_sizes: list[int],
    halo: int,
    start_level: int,
    rec_len: int,
) -> list[dict]:
    """Trace backward through IDWT levels to find required coefficient regions.

    Starting from a desired output tile at the finest level, works backward
    through each IDWT level to determine what coefficient subregions (with halo)
    are needed at each level, accounting for the actual IDWT output size formula
    (output = 2*input - (rec_len - 2), NOT 2*input).

    Args:
        tile_start: (x, y, z) start corner of the output tile in the finest volume.
        tile_size: Size of the output tile along each axis.
        level_sizes: List of coefficient sizes at each detail level.
                     level_sizes[i] is the spatial extent of detail_i coefficients.
        halo: Reconstruction filter halo (from _get_rec_halo).
        start_level: Index of the first fine level (tiled reconstruction starts here).
        rec_len: Reconstruction filter length.

    Returns:
        List of dicts, one per fine level from start_level to the finest.
        Each dict contains:
            'input_slices': (slice, slice, slice) for extracting from parent
            'detail_slices': (slice, slice, slice) for extracting detail coefficients
            'output_crops': (slice, slice, slice) for cropping the IDWT output
    """
    num_fine_levels = len(level_sizes) - start_level
    regions = [None] * num_fine_levels

    # Start from the desired output
    current_start = list(tile_start)
    current_end = [s + tile_size for s in tile_start]

    # Work backward from finest to coarsest fine level
    for idx in range(num_fine_levels - 1, -1, -1):
        level = start_level + idx
        parent_size = level_sizes[level]  # detail coeff size at this level

        slices_in = []
        slices_detail = []
        slices_crop = []

        for axis in range(3):
            in_s, in_e, crop_s, crop_e = compute_input_region(
                current_start[axis], current_end[axis],
                halo, parent_size, rec_len,
            )
            slices_in.append(slice(in_s, in_e))
            slices_detail.append(slice(in_s, in_e))
            slices_crop.append(slice(crop_s, crop_e))

        regions[idx] = {
            'input_slices': tuple(slices_in),
            'detail_slices': tuple(slices_detail),
            'output_crops': tuple(slices_crop),
        }

        # The input to this level is the output of the previous level
        current_start = [s.start for s in slices_in]
        current_end = [s.stop for s in slices_in]

    return regions


def extract_detail_subregion(
    detail_coeffs,
    slices: tuple[slice, slice, slice],
) -> dict:
    """Extract a spatial subregion from detail coefficients.

    Args:
        detail_coeffs: Either a nn.Parameter of shape (7, C, D, D, D) for dense
                       levels, or a SparseDetailLevel for sparse levels.
        slices: (slice, slice, slice) spatial subregion to extract.

    Returns:
        ptwt-compatible detail dict: {subband_key: (1, C, d, d, d)}
    """
    from sparse_coefficients import SparseDetailLevel

    if isinstance(detail_coeffs, SparseDetailLevel):
        return detail_coeffs.extract_region(slices)

    # Dense parameter: (7, C, D, D, D)
    param = detail_coeffs
    result = {}
    for j, key in enumerate(SUBBAND_KEYS):
        sub = param[j]  # (C, D, D, D)
        sub_region = sub[:, slices[0], slices[1], slices[2]]  # (C, d, d, d)
        result[key] = sub_region.unsqueeze(0)  # (1, C, d, d, d)
    return result


def tiled_waverec3(
    base_volume: torch.Tensor,
    detail_coeffs: list,
    tile_start: tuple[int, int, int],
    tile_size: int,
    wavelet: str,
    start_level: int,
    level_sizes: list[int],
) -> torch.Tensor:
    """Reconstruct a single spatial tile through cascaded single-level IDWT.

    Reconstructs only the requested tile region by extracting coefficient
    subregions with appropriate halo at each IDWT level and cascading forward.

    Args:
        base_volume: (1, C, R, R, R) fully reconstructed volume up to start_level.
                     This is the "coarse" volume that fits in memory.
        detail_coeffs: List of detail coefficient tensors/modules for levels
                       start_level through decomp_levels-1.
        tile_start: (x, y, z) start corner of the output tile in the finest volume.
        tile_size: Size of the output tile along each axis.
        wavelet: Wavelet family string (e.g., "bior4.4").
        start_level: Index of the first tiled level.
        level_sizes: Coefficient sizes at each detail level (from dry-run wavedec3).

    Returns:
        tile: (1, C, tile_size, tile_size, tile_size) reconstructed output tile.
    """
    w = pywt.Wavelet(wavelet)
    halo = (w.rec_len - 1) // 2
    rec_len = w.rec_len

    # Trace backward to find required regions at each level
    regions = trace_tile_regions(
        tile_start, tile_size, level_sizes, halo, start_level, rec_len,
    )

    # Extract the base patch (input to the first fine-level IDWT)
    base_slices = regions[0]['input_slices']
    current = base_volume[:, :, base_slices[0], base_slices[1], base_slices[2]]

    # Cascade forward through fine levels
    # Disable autocast: ptwt does not support bf16.
    with torch.amp.autocast('cuda', enabled=False):
        for idx, region in enumerate(regions):
            # Get detail coefficients for this subregion
            detail_dict = extract_detail_subregion(
                detail_coeffs[idx], region['detail_slices'],
            )

            # Single-level IDWT
            current = ptwt.waverec3(
                (current, detail_dict), wavelet=wavelet,
            )

            # Crop to the valid output region for this level
            crops = region['output_crops']
            current = current[:, :, crops[0], crops[1], crops[2]]

    return current


def get_tile_grid_info(
    target_resolution: int,
    tile_size: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Compute tile grid dimensions and enumerate all tile positions.

    Args:
        target_resolution: Full volume resolution (e.g., 2048).
        tile_size: Tile size along each axis (e.g., 64).

    Returns:
        (tiles_per_axis, tile_starts):
            tiles_per_axis: Number of tiles along each axis.
            tile_starts: List of (x, y, z) start positions for all tiles.
    """
    tiles_per_axis = (target_resolution + tile_size - 1) // tile_size
    tile_starts = []
    for tz in range(tiles_per_axis):
        for ty in range(tiles_per_axis):
            for tx in range(tiles_per_axis):
                tile_starts.append((
                    tx * tile_size,
                    ty * tile_size,
                    tz * tile_size,
                ))
    return tiles_per_axis, tile_starts


def xyz_to_tile_index(
    xyz: torch.Tensor,
    scene_bound: float,
    target_resolution: int,
    tile_size: int,
) -> torch.Tensor:
    """Map world-space coordinates to tile indices.

    Args:
        xyz: (N, 3) world-space coordinates in [-scene_bound, scene_bound].
        scene_bound: Scene bounding box half-extent.
        target_resolution: Full volume resolution.
        tile_size: Tile size along each axis.

    Returns:
        tile_indices: (N, 3) integer tile indices (tx, ty, tz).
    """
    # Normalize to [0, 1]
    normalized = (xyz / scene_bound + 1.0) * 0.5  # [0, 1]
    # Map to voxel coordinates
    voxel_coords = (normalized * target_resolution).long()
    voxel_coords = voxel_coords.clamp(0, target_resolution - 1)
    # Map to tile indices
    tile_indices = voxel_coords // tile_size
    return tile_indices
