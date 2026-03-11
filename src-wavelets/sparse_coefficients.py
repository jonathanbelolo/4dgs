"""Sparse wavelet detail coefficient storage.

Stores detail coefficients only for occupied spatial regions, using structural
sparsity: unoccupied tiles have no parameters at all (not zero-valued parameters).
This avoids allocating optimizer state for empty regions.

For the lego scene at 2048³, ~5-20% spatial occupancy per level reduces
coefficient memory from 448 GB to ~19 GB (fp16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_volume import SUBBAND_KEYS


class SparseDetailLevel(nn.Module):
    """Sparse storage for one wavelet detail level.

    Instead of a dense (7, C, D, D, D) parameter tensor, stores only the tiles
    that overlap occupied regions. Each tile is (7, C, block, block, block).

    Args:
        full_size: Full spatial extent of this detail level (D).
        num_channels: Number of feature channels (C).
        block_size: Tile size for sparse storage.
        occupancy_mask: (T, T, T) boolean tensor where T = full_size // block_size.
                        True = occupied, allocate parameters.
    """

    def __init__(
        self,
        full_size: int,
        num_channels: int,
        block_size: int,
        occupancy_mask: torch.Tensor,
    ):
        super().__init__()
        self.full_size = full_size
        self.num_channels = num_channels
        self.block_size = block_size

        # Store occupancy as a buffer (not a parameter)
        self.register_buffer('occupancy', occupancy_mask.bool())

        # Build tile index: (tx, ty, tz) → position in ParameterList
        occupied_positions = occupancy_mask.nonzero(as_tuple=False)  # (N_occ, 3)
        self.register_buffer('occupied_positions', occupied_positions)
        self.n_occupied = len(occupied_positions)

        # Spatial hash for fast lookup
        self._tile_to_idx: dict[tuple[int, int, int], int] = {}
        for idx in range(self.n_occupied):
            pos = occupied_positions[idx]
            key = (pos[0].item(), pos[1].item(), pos[2].item())
            self._tile_to_idx[key] = idx

        # Allocate parameters only for occupied tiles
        # Each tile: (7, C, block, block, block)
        self.tiles = nn.ParameterList([
            nn.Parameter(
                torch.randn(7, num_channels, block_size, block_size, block_size) * 0.001
            )
            for _ in range(self.n_occupied)
        ])

    @property
    def tiles_per_axis(self) -> int:
        return self.occupancy.shape[0]

    def is_occupied(self, tx: int, ty: int, tz: int) -> bool:
        """Check if a tile is occupied."""
        return (tx, ty, tz) in self._tile_to_idx

    def get_tile(self, tx: int, ty: int, tz: int) -> torch.Tensor | None:
        """Get coefficient tile at (tx, ty, tz), or None if unoccupied.

        Returns:
            (7, C, block, block, block) parameter tensor, or None.
        """
        idx = self._tile_to_idx.get((tx, ty, tz))
        if idx is None:
            return None
        return self.tiles[idx]

    def extract_region(
        self,
        slices: tuple[slice, slice, slice],
    ) -> dict:
        """Extract detail coefficients for a spatial region.

        Assembles a dense subregion by gathering tiles and zero-padding gaps.
        Uses F.pad + addition (not in-place assignment) to preserve autograd
        gradient flow back to sparse tile parameters.

        The returned dict is compatible with ptwt.waverec3.

        Args:
            slices: (slice_x, slice_y, slice_z) defining the subregion in the
                    full detail coefficient space.

        Returns:
            Detail dict: {subband_key: (1, C, sx, sy, sz)} where sx/sy/sz are
            the sizes of the sliced region.
        """
        sx = slices[0].stop - slices[0].start
        sy = slices[1].stop - slices[1].start
        sz = slices[2].stop - slices[2].start
        device = self.occupancy.device
        dtype = self.tiles[0].dtype if self.n_occupied > 0 else torch.float32

        # Determine which tiles overlap the requested region
        tile_x_start = slices[0].start // self.block_size
        tile_x_end = (slices[0].stop - 1) // self.block_size + 1
        tile_y_start = slices[1].start // self.block_size
        tile_y_end = (slices[1].stop - 1) // self.block_size + 1
        tile_z_start = slices[2].start // self.block_size
        tile_z_end = (slices[2].stop - 1) // self.block_size + 1

        # Collect tile contributions per subband then sum.
        # Using F.pad + addition preserves autograd (unlike in-place assignment
        # to a torch.zeros tensor which breaks gradient flow).
        result = {}
        for j, key in enumerate(SUBBAND_KEYS):
            contributions = []

            for tx in range(tile_x_start, tile_x_end):
                for ty in range(tile_y_start, tile_y_end):
                    for tz in range(tile_z_start, tile_z_end):
                        tile = self.get_tile(tx, ty, tz)
                        if tile is None:
                            continue

                        # Tile covers [tx*block, (tx+1)*block) in full space
                        tile_origin_x = tx * self.block_size
                        tile_origin_y = ty * self.block_size
                        tile_origin_z = tz * self.block_size

                        # Overlap between tile and requested region
                        ox_s = max(tile_origin_x, slices[0].start)
                        ox_e = min(tile_origin_x + self.block_size, slices[0].stop)
                        oy_s = max(tile_origin_y, slices[1].start)
                        oy_e = min(tile_origin_y + self.block_size, slices[1].stop)
                        oz_s = max(tile_origin_z, slices[2].start)
                        oz_e = min(tile_origin_z + self.block_size, slices[2].stop)

                        if ox_s >= ox_e or oy_s >= oy_e or oz_s >= oz_e:
                            continue

                        # Source slice within the tile
                        src_x = slice(ox_s - tile_origin_x, ox_e - tile_origin_x)
                        src_y = slice(oy_s - tile_origin_y, oy_e - tile_origin_y)
                        src_z = slice(oz_s - tile_origin_z, oz_e - tile_origin_z)

                        # Destination offset within the dense output
                        dst_x_s = ox_s - slices[0].start
                        dst_y_s = oy_s - slices[1].start
                        dst_z_s = oz_s - slices[2].start
                        dst_x_e = ox_e - slices[0].start
                        dst_y_e = oy_e - slices[1].start
                        dst_z_e = oz_e - slices[2].start

                        # Extract tile slice (indexing preserves autograd)
                        tile_slice = tile[j, :, src_x, src_y, src_z].unsqueeze(0)

                        # Pad to full output size (F.pad is differentiable)
                        # F.pad order: (left_z, right_z, left_y, right_y, left_x, right_x)
                        pad = (
                            dst_z_s, sz - dst_z_e,
                            dst_y_s, sy - dst_y_e,
                            dst_x_s, sx - dst_x_e,
                        )
                        contributions.append(F.pad(tile_slice, pad))

            if contributions:
                dense = contributions[0]
                for c in contributions[1:]:
                    dense = dense + c
            else:
                dense = torch.zeros(1, self.num_channels, sx, sy, sz,
                                    device=device, dtype=dtype)

            result[key] = dense

        return result

    def allocate_tile(self, tx: int, ty: int, tz: int) -> None:
        """Allocate a new tile at (tx, ty, tz).

        Called when occupancy expands during training refinement.
        """
        if self.is_occupied(tx, ty, tz):
            return

        device = self.occupancy.device
        dtype = self.tiles[0].dtype if self.n_occupied > 0 else torch.float32
        new_param = nn.Parameter(
            torch.randn(
                7, self.num_channels, self.block_size, self.block_size, self.block_size,
                device=device, dtype=dtype,
            ) * 0.001
        )
        idx = len(self.tiles)
        self.tiles.append(new_param)
        self._tile_to_idx[(tx, ty, tz)] = idx
        self.n_occupied += 1
        self.occupancy[tx, ty, tz] = True

    def deallocate_tile(self, tx: int, ty: int, tz: int) -> None:
        """Mark a tile as unoccupied (zeros its data but keeps the slot).

        Full deallocation (removing from ParameterList) would invalidate
        optimizer state indices. Instead, we zero the data and remove from
        the spatial hash so it's never queried.
        """
        idx = self._tile_to_idx.get((tx, ty, tz))
        if idx is None:
            return
        self.tiles[idx].data.zero_()
        self.tiles[idx].requires_grad_(False)
        del self._tile_to_idx[(tx, ty, tz)]
        self.occupancy[tx, ty, tz] = False

    def set_from_dense(self, dense_coeffs: torch.Tensor) -> None:
        """Populate occupied tiles from a dense DWT coefficient tensor.

        Used during svox2-to-wavelet conversion: the forward DWT produces
        dense detail coefficients, but we store only occupied tiles.

        Args:
            dense_coeffs: (7, C, D, D, D) dense detail coefficients from DWT.
                          C must equal self.num_channels. D must equal self.full_size.
        """
        assert dense_coeffs.shape[0] == 7
        assert dense_coeffs.shape[1] == self.num_channels, (
            f"Channel mismatch: dense has {dense_coeffs.shape[1]}, "
            f"sparse expects {self.num_channels}"
        )
        with torch.no_grad():
            for idx in range(self.n_occupied):
                pos = self.occupied_positions[idx]
                tx, ty, tz = pos[0].item(), pos[1].item(), pos[2].item()
                x_s = tx * self.block_size
                x_e = min(x_s + self.block_size, self.full_size)
                y_s = ty * self.block_size
                y_e = min(y_s + self.block_size, self.full_size)
                z_s = tz * self.block_size
                z_e = min(z_s + self.block_size, self.full_size)
                src = dense_coeffs[:, :, x_s:x_e, y_s:y_e, z_s:z_e]
                bx, by, bz = x_e - x_s, y_e - y_s, z_e - z_s
                self.tiles[idx].data[:, :, :bx, :by, :bz].copy_(src)

    def set_channels_from_dense(
        self, dense_coeffs: torch.Tensor, channel_offset: int,
    ) -> None:
        """Populate a slice of channels in occupied tiles from dense coefficients.

        For batched DWT processing: call multiple times with different channel
        ranges to avoid materializing the full 28-channel dense grid.

        Args:
            dense_coeffs: (7, C_batch, D, D, D) dense coefficients for a channel batch.
            channel_offset: Starting channel index in the tile parameters.
        """
        C_batch = dense_coeffs.shape[1]
        ch_end = channel_offset + C_batch
        assert ch_end <= self.num_channels
        with torch.no_grad():
            for idx in range(self.n_occupied):
                pos = self.occupied_positions[idx]
                tx, ty, tz = pos[0].item(), pos[1].item(), pos[2].item()
                x_s = tx * self.block_size
                x_e = min(x_s + self.block_size, self.full_size)
                y_s = ty * self.block_size
                y_e = min(y_s + self.block_size, self.full_size)
                z_s = tz * self.block_size
                z_e = min(z_s + self.block_size, self.full_size)
                src = dense_coeffs[:, :, x_s:x_e, y_s:y_e, z_s:z_e]
                bx, by, bz = x_e - x_s, y_e - y_s, z_e - z_s
                self.tiles[idx].data[:, channel_offset:ch_end, :bx, :by, :bz].copy_(src)

    def sparsity_stats(self) -> dict:
        """Return sparsity statistics."""
        total_tiles = self.occupancy.numel()
        return {
            'occupied': self.n_occupied,
            'total': total_tiles,
            'occupancy_ratio': self.n_occupied / total_tiles,
            'params': self.n_occupied * 7 * self.num_channels * self.block_size ** 3,
            'memory_mb': (
                self.n_occupied * 7 * self.num_channels * self.block_size ** 3 * 4
            ) / 1024 ** 2,
        }
