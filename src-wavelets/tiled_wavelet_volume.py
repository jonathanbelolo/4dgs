"""Tiled wavelet volume for high-resolution (1024³+) reconstruction.

Extends the base WaveletVolume with:
- Tiled IDWT reconstruction (only materializes tiles, not the full volume)
- Sparse coefficient storage (only stores coefficients in occupied regions)
- Per-level channel reduction (fine levels use fewer channels)

The coarse levels (0..base_level) are still fully reconstructed and fit in
memory. Fine levels (base_level+1..decomp_levels-1) use tiled reconstruction
with sparse coefficient storage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from sh import eval_sh_color
from wavelet_volume import WaveletVolume, SUBBAND_KEYS
from sparse_coefficients import SparseDetailLevel
from tiled_idwt import (
    tiled_waverec3,
    xyz_to_tile_index,
    get_tile_grid_info,
)


def _get_level_sizes(target_resolution: int, decomp_levels: int, wavelet: str) -> list[int]:
    """Get the spatial size of detail coefficients at each level.

    Uses iterative single-level 1D decompositions to discover coefficient sizes
    without allocating the full-resolution 3D volume. The 3D DWT is separable
    (applies 1D along each axis), so 1D sizes are valid for each spatial
    dimension of the cubic volume.

    Returns:
        List of length decomp_levels + 1:
        [approx_size, detail_0_size, detail_1_size, ..., detail_N-1_size]
    """
    sizes = []
    current_size = target_resolution

    for _ in range(decomp_levels):
        dummy = torch.zeros(1, 1, current_size)
        coeffs = ptwt.wavedec(dummy, wavelet=wavelet, level=1)
        approx_size = coeffs[0].shape[-1]
        detail_size = coeffs[1].shape[-1]
        sizes.append(detail_size)
        current_size = approx_size

    sizes.append(current_size)  # final approx size
    sizes.reverse()  # [approx, detail_0, ..., detail_N-1]
    return sizes


class TiledWaveletVolume(nn.Module):
    """Wavelet volume with tiled IDWT for high-resolution reconstruction.

    Architecture:
        Dense base levels → full IDWT → base volume (fits in memory)
        Sparse fine levels → tiled IDWT per query batch → tile volumes (temporary)
        Trilinear query per tile → features → SH decode → density + RGB

    Args:
        base_resolution: Coarsest level resolution.
        decomp_levels: Total number of wavelet decomposition levels.
        num_channels: Feature channels for dense (base) levels.
        channels_per_level: Per-level channel counts. If provided, fine levels
                           can use fewer channels (e.g., [28, 28, 28, 16, 8, 4]).
                           Length must equal decomp_levels. If None, all levels
                           use num_channels.
        wavelet: Wavelet family string.
        scene_bound: Scene bounding box half-extent.
        tile_size: Output tile size for tiled reconstruction.
        base_level: Levels 0..base_level are dense. Levels above are sparse/tiled.
        occupancy_masks: Dict mapping level index → (T,T,T) boolean mask for
                        sparse levels. Required for levels > base_level.
    """

    def __init__(
        self,
        base_resolution: int = 32,
        decomp_levels: int = 6,
        num_channels: int = 28,
        channels_per_level: list[int] | None = None,
        wavelet: str = "bior4.4",
        scene_bound: float = 1.5,
        tile_size: int = 64,
        base_level: int = 2,
        occupancy_masks: dict[int, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.base_resolution = base_resolution
        self.decomp_levels = decomp_levels
        self.num_channels = num_channels
        self.wavelet = wavelet
        self.scene_bound = scene_bound
        self.tile_size = tile_size
        self.base_level = base_level
        self.target_resolution = base_resolution * (2 ** decomp_levels)

        # Determine per-level channels
        if channels_per_level is not None:
            assert len(channels_per_level) == decomp_levels
            self.channels_per_level = channels_per_level
        else:
            self.channels_per_level = [num_channels] * decomp_levels

        # Discover coefficient sizes via dry-run
        # level_sizes[0] = approx size, level_sizes[i+1] = detail_i size
        self.level_sizes = _get_level_sizes(
            self.target_resolution, decomp_levels, wavelet,
        )

        # --- Dense base levels (0..base_level) ---
        # These use the same structure as WaveletVolume
        approx_size = self.level_sizes[0]
        self.approx = nn.Parameter(
            torch.randn(1, num_channels, approx_size, approx_size, approx_size) * 0.01
        )

        self.dense_details = nn.ParameterList()
        for i in range(base_level + 1):
            D = self.level_sizes[i + 1]
            C = self.channels_per_level[i]
            self.dense_details.append(nn.Parameter(
                torch.randn(7, C, D, D, D) * 0.001
            ))

        # --- Sparse fine levels (base_level+1..decomp_levels-1) ---
        self.sparse_details = nn.ModuleList()
        for i in range(base_level + 1, decomp_levels):
            D = self.level_sizes[i + 1]
            C = self.channels_per_level[i]

            if occupancy_masks is not None and i in occupancy_masks:
                occ = occupancy_masks[i]
            else:
                # Default: fully occupied (no sparsity)
                tiles_per_axis = max(1, (D + tile_size - 1) // tile_size)
                occ = torch.ones(tiles_per_axis, tiles_per_axis, tiles_per_axis, dtype=torch.bool)

            self.sparse_details.append(SparseDetailLevel(
                full_size=D,
                num_channels=C,
                block_size=min(tile_size, D),
                occupancy_mask=occ,
            ))

        # Pre-compute tile grid for the finest level
        self._tiles_per_axis, self._tile_starts = get_tile_grid_info(
            self.target_resolution, tile_size,
        )


    def _make_dense_detail_dict(self, level: int) -> dict:
        """Build ptwt detail dict from a dense detail parameter."""
        param = self.dense_details[level]  # (7, C, D, D, D)
        C = param.shape[1]

        result = {}
        for j, key in enumerate(SUBBAND_KEYS):
            sub = param[j].unsqueeze(0)  # (1, C, D, D, D)
            # Pad channels to num_channels if this level has fewer
            if C < self.num_channels:
                pad = torch.zeros(
                    1, self.num_channels - C, *sub.shape[2:],
                    device=sub.device, dtype=sub.dtype,
                )
                sub = torch.cat([sub, pad], dim=1)
            result[key] = sub
        return result

    def reconstruct_base(self, max_level: int | None = None) -> torch.Tensor:
        """Reconstruct the volume up to a given base detail level.

        Args:
            max_level: Maximum dense detail level to include (0-indexed).
                       None = use all base levels (0..base_level).
                       -1 = approx only.

        Returns:
            base_volume: (1, num_channels, R, R, R)
        """
        if max_level is None:
            max_level = self.base_level
        max_level = min(max_level, self.base_level)

        if max_level < 0:
            return self.approx

        volume = self.approx
        # Disable autocast: ptwt does not support bf16.
        with torch.amp.autocast('cuda', enabled=False):
            for i in range(max_level + 1):
                detail_dict = self._make_dense_detail_dict(i)
                volume = ptwt.waverec3((volume, detail_dict), wavelet=self.wavelet)
                # Truncate to expected size: the IDWT may produce 1 extra voxel
                # when the original signal had odd length (DWT rounds up, IDWT
                # doesn't know the original length).
                if i + 2 < len(self.level_sizes):
                    expected = self.level_sizes[i + 2]
                    volume = volume[:, :, :expected, :expected, :expected]
        return volume

    def reconstruct_tile(
        self,
        tile_start: tuple[int, int, int],
        base_volume: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a single tile at the finest resolution.

        Args:
            tile_start: (x, y, z) start corner in the target volume.
            base_volume: Pre-reconstructed base volume from reconstruct_base().

        Returns:
            tile: (1, C_fine, tile_size, tile_size, tile_size)
                  where C_fine = channels_per_level of the finest level.
        """
        # Collect fine-level detail coefficients (sparse)
        fine_details = list(self.sparse_details)

        return tiled_waverec3(
            base_volume=base_volume,
            detail_coeffs=fine_details,
            tile_start=tile_start,
            tile_size=self.tile_size,
            wavelet=self.wavelet,
            start_level=self.base_level + 1,
            level_sizes=self.level_sizes[1:],  # skip approx size
        )

    def query_tiled(
        self,
        xyz: torch.Tensor,
        base_volume: torch.Tensor,
    ) -> torch.Tensor:
        """Query the full-resolution volume at arbitrary points using tiled reconstruction.

        Routes query points to tiles, reconstructs only needed tiles, and
        samples features via grid_sample within each tile.

        Args:
            xyz: (N, 3) world-space coordinates in [-scene_bound, scene_bound].
            base_volume: Pre-reconstructed base volume.

        Returns:
            features: (N, num_channels) interpolated features.
                      Fine-level channels come from tiles, remaining from base.
        """
        N = xyz.shape[0]
        device = xyz.device

        # Map points to tile indices
        tile_indices = xyz_to_tile_index(
            xyz, self.scene_bound, self.target_resolution, self.tile_size,
        )  # (N, 3) int

        # Find unique tiles
        unique_tiles, inverse = torch.unique(
            tile_indices, dim=0, return_inverse=True,
        )

        # Query base volume for all points (provides coarse features + SH channels)
        base_features = self._query_volume(xyz, base_volume)  # (N, num_channels)

        # Determine finest-level channel count
        finest_channels = self.channels_per_level[-1]

        if len(self.sparse_details) == 0:
            # No fine levels — base volume is the full volume
            return base_features

        # For each unique tile, reconstruct and sample.
        # Collect (indices, features) pairs to merge at the end.
        # Avoids accumulating into a torch.zeros tensor which would break autograd.
        tile_updates = []

        for i in range(unique_tiles.shape[0]):
            tile_idx = (
                unique_tiles[i, 0].item(),
                unique_tiles[i, 1].item(),
                unique_tiles[i, 2].item(),
            )

            # Check if any sparse level has this tile occupied
            # (check finest level — if it's unoccupied, skip)
            finest_sparse = self.sparse_details[-1]
            # Map output tile to finest detail coefficient tile
            coeff_tiles_per_axis = finest_sparse.tiles_per_axis
            scale = coeff_tiles_per_axis / self._tiles_per_axis
            coeff_tile = (
                min(int(tile_idx[0] * scale), coeff_tiles_per_axis - 1),
                min(int(tile_idx[1] * scale), coeff_tiles_per_axis - 1),
                min(int(tile_idx[2] * scale), coeff_tiles_per_axis - 1),
            )
            if not finest_sparse.is_occupied(*coeff_tile):
                continue

            # Reconstruct this tile
            tile_start = (
                tile_idx[0] * self.tile_size,
                tile_idx[1] * self.tile_size,
                tile_idx[2] * self.tile_size,
            )
            tile_volume = self.reconstruct_tile(tile_start, base_volume)

            # Find points in this tile
            mask_indices = (inverse == i).nonzero(as_tuple=True)[0]
            if mask_indices.numel() == 0:
                continue

            # Convert world coords to tile-local [-1, 1] for grid_sample
            xyz_tile = xyz[mask_indices]
            xyz_local = self._xyz_to_tile_local(xyz_tile, tile_start)

            # Sample from tile (preserves autograd to tile coefficients)
            local_features = self._query_volume(xyz_local, tile_volume, local=True)
            tile_updates.append((mask_indices, local_features))

        # Merge: replace fine channels from tiles, keep SH channels from base.
        # Uses differentiable scatter so gradient flow works even when
        # base is frozen (requires_grad=False) during fine-level training.
        if tile_updates:
            all_indices = torch.cat([idx for idx, _ in tile_updates])
            all_tile_feats = torch.cat([feats for _, feats in tile_updates])
            C_tile = min(finest_channels, self.num_channels)

            # Scatter tile features into full-size tensor (differentiable)
            idx_exp = all_indices.unsqueeze(1).expand(-1, C_tile)
            tile_scattered = torch.zeros(
                N, C_tile, device=device, dtype=all_tile_feats.dtype,
            ).scatter(0, idx_exp, all_tile_feats[:, :C_tile])

            # Binary mask: 1 at tile positions, 0 elsewhere
            mask = torch.zeros(N, C_tile, device=device).scatter(
                0, idx_exp,
                torch.ones(all_indices.shape[0], C_tile, device=device),
            )

            # Mix: tile features where mask=1, base features elsewhere
            base = base_features.detach()
            fine_part = base[:, :C_tile] * (1 - mask) + tile_scattered * mask
            if C_tile < self.num_channels:
                features = torch.cat([fine_part, base[:, C_tile:]], dim=1)
            else:
                features = fine_part
            return features
        else:
            return base_features

    def _query_volume(
        self,
        xyz: torch.Tensor,
        volume: torch.Tensor,
        local: bool = False,
    ) -> torch.Tensor:
        """Trilinear interpolation query on a volume.

        Args:
            xyz: (N, 3) coordinates. If local=False, world-space [-scene_bound, scene_bound].
                 If local=True, already in [-1, 1] tile-local coordinates.
            volume: (1, C, R, R, R) volume tensor.
            local: Whether xyz is already in [-1, 1] grid coordinates.

        Returns:
            features: (N, C) interpolated features.
        """
        if local:
            grid = xyz
        else:
            grid = xyz / self.scene_bound

        grid = grid.view(1, -1, 1, 1, 3)

        features = F.grid_sample(
            volume, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)

    def _xyz_to_tile_local(
        self,
        xyz: torch.Tensor,
        tile_start: tuple[int, int, int],
    ) -> torch.Tensor:
        """Convert world-space coordinates to tile-local [-1, 1] for grid_sample.

        Args:
            xyz: (N, 3) world-space coordinates.
            tile_start: (x, y, z) start corner of the tile in voxel coordinates.

        Returns:
            local: (N, 3) coordinates in [-1, 1] relative to the tile.
        """
        R = self.target_resolution
        # World → [0, 1]
        normalized = (xyz / self.scene_bound + 1.0) * 0.5
        # [0, 1] → voxel coordinates
        voxel = normalized * R
        # Voxel → tile-local [0, tile_size]
        local_voxel = voxel - torch.tensor(
            tile_start, device=xyz.device, dtype=xyz.dtype,
        )
        # [0, tile_size] → [-1, 1]
        local = local_voxel / self.tile_size * 2.0 - 1.0
        return local

    def decode(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode features into density and view-dependent RGB.

        Same as WaveletVolume.decode — channel 0 is density, channels 1:28 are SH.
        """
        raw_density = features[:, 0]
        sh_coeffs = features[:, 1:]

        density = F.relu(raw_density)
        rgb = eval_sh_color(sh_coeffs, view_dirs)

        return density, rgb

    def query(self, xyz: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """Query a volume at arbitrary 3D points (dense path compatibility).

        Same interface as WaveletVolume.query.
        """
        return self._query_volume(xyz, volume, local=False)

    def set_sparse_levels(self, occupancy_masks: dict[int, torch.Tensor]) -> None:
        """Replace sparse detail levels with properly occupied ones.

        Call after Phase 1 training to set up sparse levels based on
        estimated occupancy. Reinitializes sparse_details with new
        tile allocations.

        Args:
            occupancy_masks: Dict mapping level index → (T,T,T) boolean mask.
        """
        self.sparse_details = nn.ModuleList()
        for i in range(self.base_level + 1, self.decomp_levels):
            D = self.level_sizes[i + 1]
            C = self.channels_per_level[i]

            if i in occupancy_masks:
                occ = occupancy_masks[i]
            else:
                tiles_per_axis = max(1, (D + self.tile_size - 1) // self.tile_size)
                occ = torch.ones(tiles_per_axis, tiles_per_axis, tiles_per_axis, dtype=torch.bool)

            self.sparse_details.append(SparseDetailLevel(
                full_size=D,
                num_channels=C,
                block_size=min(self.tile_size, D),
                occupancy_mask=occ,
            ))

    def load_coarse_from(self, coarse_model: WaveletVolume) -> None:
        """Initialize dense base levels from a pretrained coarse WaveletVolume.

        Copies matching parameters from the coarse model into this model's
        approx and dense detail levels. Uses trilinear interpolation when
        sizes differ (bior4.4 decomposition from different starting
        resolutions produces slightly different intermediate sizes).

        Args:
            coarse_model: A trained WaveletVolume with <= base_level+1 detail levels.
        """
        with torch.no_grad():
            # Copy approximation (may need interpolation for size mismatch)
            src = coarse_model.approx.data
            if src.shape == self.approx.data.shape:
                self.approx.data.copy_(src)
            else:
                self.approx.data.copy_(
                    F.interpolate(src, size=self.approx.shape[2:], mode='trilinear', align_corners=True)
                )

            # Copy dense detail levels
            for i in range(min(len(self.dense_details), len(coarse_model.details))):
                src = coarse_model.details[i].data  # (7, C_src, D_src, D_src, D_src)
                dst = self.dense_details[i].data     # (7, C_dst, D_dst, D_dst, D_dst)
                C_src = src.shape[1]
                C_dst = dst.shape[1]
                C_copy = min(C_src, C_dst)

                if src.shape[2:] == dst.shape[2:]:
                    dst[:, :C_copy] = src[:, :C_copy]
                else:
                    # Interpolate each subband: reshape (7, C, D, D, D) → (7*C, 1, D, D, D)
                    s = src[:, :C_copy]
                    s_flat = s.reshape(-1, 1, *s.shape[2:])
                    resized = F.interpolate(s_flat, size=dst.shape[2:], mode='trilinear', align_corners=True)
                    dst[:, :C_copy] = resized.reshape(7, C_copy, *dst.shape[2:])

    def total_params(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def memory_summary(self) -> dict:
        """Compute memory usage breakdown."""
        approx_bytes = self.approx.numel() * self.approx.element_size()
        dense_bytes = sum(
            d.numel() * d.element_size() for d in self.dense_details
        )
        sparse_bytes = 0
        sparse_stats = []
        for i, sparse in enumerate(self.sparse_details):
            level = self.base_level + 1 + i
            stats = sparse.sparsity_stats()
            stats['level'] = level
            sparse_stats.append(stats)
            sparse_bytes += sum(
                t.numel() * t.element_size() for t in sparse.tiles
            )

        total = approx_bytes + dense_bytes + sparse_bytes
        return {
            'approx_mb': approx_bytes / 1024 ** 2,
            'dense_mb': dense_bytes / 1024 ** 2,
            'sparse_mb': sparse_bytes / 1024 ** 2,
            'total_mb': total / 1024 ** 2,
            'total_params_M': self.total_params() / 1e6,
            'sparse_levels': sparse_stats,
        }
