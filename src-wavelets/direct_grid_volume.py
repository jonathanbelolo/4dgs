"""Direct voxel grid volume for Plenoxels-style training.

Stores a dense 3D feature grid as a raw nn.Parameter — no wavelet transform.
Gradients flow directly to each voxel for fast convergence. Supports
coarse-to-fine resolution upsampling and weight-based voxel pruning.

After training, convert to wavelet representation via forward DWT
(ptwt.wavedec3) for compression, LoD, and optional super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sh import eval_sh_color


class DirectGridVolume(nn.Module):
    """Dense voxel grid with direct parameter optimization.

    Architecture:
        nn.Parameter grid → trilinear query → decode → density + RGB

    Follows Plenoxels (svox2) conventions:
    - Density stored as raw values, activated with ReLU
    - Initialized to small positive (0.1) so ReLU gradient flows immediately
    - SH coefficients initialized near zero
    - Supports weight-based pruning to remove fog
    """

    def __init__(
        self,
        resolution: int = 128,
        num_channels: int = 28,
        scene_bound: float = 1.5,
        init_sigma: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_channels = num_channels
        self.scene_bound = scene_bound

        # Separate density and SH parameters for different LR groups
        self.density_grid = nn.Parameter(
            torch.full(
                (1, 1, resolution, resolution, resolution),
                init_sigma,
            )
        )
        self.sh_grid = nn.Parameter(
            torch.randn(
                1, num_channels - 1, resolution, resolution, resolution,
            ) * 0.01
        )

    @property
    def grid(self) -> torch.Tensor:
        """Combined (1, C, R, R, R) grid for compatibility."""
        return torch.cat([self.density_grid, self.sh_grid], dim=1)

    @property
    def decomp_levels(self) -> int:
        """Compatibility with renderer (no wavelet levels)."""
        return 0

    def reconstruct(self, max_level=None) -> torch.Tensor:
        """Return the grid directly."""
        return self.grid

    def reconstruct_pair(self, fine_level, coarse_level):
        """Return (fine, coarse) volumes for hierarchical sampling."""
        g = self.grid
        return g, g

    def query(self, xyz: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation at arbitrary 3D points.

        Args:
            xyz: (N, 3) world-space coordinates in [-scene_bound, scene_bound]
            volume: (1, C, R, R, R) volume tensor

        Returns:
            features: (N, C) interpolated features
        """
        grid = xyz / self.scene_bound  # → [-1, 1]
        grid = grid.view(1, -1, 1, 1, 3)
        features = F.grid_sample(
            volume, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)

    def decode(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode features into density and view-dependent RGB.

        Args:
            features: (N, C) raw features. Channel 0 = density, rest = color.
                      If C=4: channels 1-3 are direct RGB (view-independent).
                      If C=28: channels 1-27 are SH coefficients (view-dependent).
            view_dirs: (N, 3) unit view direction vectors

        Returns:
            density: (N,) non-negative density
            rgb: (N, 3) color in [0, 1]
        """
        raw_density = features[:, 0]
        density = F.relu(raw_density)

        color_features = features[:, 1:]
        if color_features.shape[1] == 3:
            # Direct RGB (no SH, view-independent)
            rgb = torch.sigmoid(color_features)
        else:
            # Full SH evaluation (view-dependent)
            rgb = eval_sh_color(color_features, view_dirs)
        return density, rgb

    def upsample(self, new_resolution: int) -> None:
        """Upsample grid to higher resolution via trilinear interpolation.

        Caller must create a new optimizer after this (old state is invalid).
        """
        with torch.no_grad():
            new_density = F.interpolate(
                self.density_grid.data,
                size=(new_resolution, new_resolution, new_resolution),
                mode="trilinear",
                align_corners=True,
            )
            new_sh = F.interpolate(
                self.sh_grid.data,
                size=(new_resolution, new_resolution, new_resolution),
                mode="trilinear",
                align_corners=True,
            )
        self.density_grid = nn.Parameter(new_density)
        self.sh_grid = nn.Parameter(new_sh)
        self.resolution = new_resolution

    def prune(self, max_weights: torch.Tensor, threshold: float = 0.01) -> int:
        """Zero out voxels whose max rendering weight is below threshold.

        Args:
            max_weights: (R, R, R) tensor of maximum rendering weight per voxel,
                         accumulated across training rays.
            threshold: Voxels below this weight are pruned.

        Returns:
            Number of voxels pruned.
        """
        mask = max_weights < threshold
        n_pruned = mask.sum().item()
        with torch.no_grad():
            self.density_grid.data[0, 0][mask] = 0.0
            self.sh_grid.data[0, :, mask] = 0.0
        return n_pruned

    def tv_loss(self) -> torch.Tensor:
        """L1 total variation on density channel only."""
        d = self.density_grid
        dx = (d[:, :, 1:] - d[:, :, :-1]).abs().mean()
        dy = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
        dz = (d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).abs().mean()
        return dx + dy + dz

    def tv_loss_sh(self) -> torch.Tensor:
        """L1 total variation on SH channels."""
        s = self.sh_grid
        dx = (s[:, :, 1:] - s[:, :, :-1]).abs().mean()
        dy = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs().mean()
        dz = (s[:, :, :, :, 1:] - s[:, :, :, :, :-1]).abs().mean()
        return dx + dy + dz

    def total_params(self) -> int:
        """Total number of learnable parameters."""
        return self.density_grid.numel() + self.sh_grid.numel()

    def state_dict_combined(self) -> dict:
        """Return state dict with a single 'grid' key for DWT conversion."""
        return {"grid": self.grid.detach()}


class ResidualGridVolume(nn.Module):
    """Frozen base + trainable detail grid (residual architecture).

    Implements V_ℓ = base + detail, where:
    - base: upsampled coarse reconstruction, frozen (no gradients)
    - detail: initialized to zero, learns the residual

    The forward pass queries (base + detail) via trilinear interpolation.
    Only the detail grid receives gradients.
    """

    def __init__(
        self,
        base_grid: torch.Tensor,
        num_channels: int = 28,
        scene_bound: float = 1.5,
    ):
        """
        Args:
            base_grid: (1, C, R, R, R) frozen base volume (upsampled from
                       previous stage). Will be registered as a buffer (no grad).
            num_channels: must match base_grid.shape[1]
            scene_bound: scene fits in [-bound, bound]³
        """
        super().__init__()
        assert base_grid.shape[1] == num_channels
        R = base_grid.shape[-1]
        self.resolution = R
        self.num_channels = num_channels
        self.scene_bound = scene_bound

        # Frozen base — registered as buffer, not parameter
        self.register_buffer("base_density", base_grid[:, :1].clone())
        self.register_buffer("base_sh", base_grid[:, 1:].clone())

        # Trainable detail — initialized to zero (residual default)
        self.detail_density = nn.Parameter(
            torch.zeros(1, 1, R, R, R, device=base_grid.device)
        )
        self.detail_sh = nn.Parameter(
            torch.zeros(1, num_channels - 1, R, R, R, device=base_grid.device)
        )

    @property
    def density_grid(self) -> torch.Tensor:
        """Combined density for compatibility (base + detail)."""
        return self.base_density + self.detail_density

    @property
    def sh_grid(self) -> torch.Tensor:
        """Combined SH for compatibility (base + detail)."""
        return self.base_sh + self.detail_sh

    @property
    def grid(self) -> torch.Tensor:
        """Combined (1, C, R, R, R) grid."""
        return torch.cat([self.density_grid, self.sh_grid], dim=1)

    @property
    def decomp_levels(self) -> int:
        return 0

    def reconstruct(self, max_level=None) -> torch.Tensor:
        return self.grid

    def reconstruct_pair(self, fine_level, coarse_level):
        g = self.grid
        return g, g

    def query(self, xyz: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation at arbitrary 3D points."""
        grid_coords = xyz / self.scene_bound
        grid_coords = grid_coords.view(1, -1, 1, 1, 3)
        features = F.grid_sample(
            volume, grid_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)

    def decode(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode features into density and RGB."""
        raw_density = features[:, 0]
        density = F.relu(raw_density)

        color_features = features[:, 1:]
        if color_features.shape[1] == 3:
            rgb = torch.sigmoid(color_features)
        else:
            rgb = eval_sh_color(color_features, view_dirs)
        return density, rgb

    def tv_loss(self) -> torch.Tensor:
        """L1 total variation on detail density only."""
        d = self.detail_density
        dx = (d[:, :, 1:] - d[:, :, :-1]).abs().mean()
        dy = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
        dz = (d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).abs().mean()
        return dx + dy + dz

    def tv_loss_sh(self) -> torch.Tensor:
        """L1 total variation on detail SH channels."""
        s = self.detail_sh
        dx = (s[:, :, 1:] - s[:, :, :-1]).abs().mean()
        dy = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs().mean()
        dz = (s[:, :, :, :, 1:] - s[:, :, :, :, :-1]).abs().mean()
        return dx + dy + dz

    def total_params(self) -> int:
        """Total number of learnable parameters (detail only)."""
        return self.detail_density.numel() + self.detail_sh.numel()

    def detail_sparsity(self, threshold: float = 1e-4) -> float:
        """Fraction of detail coefficients near zero."""
        with torch.no_grad():
            detail = torch.cat([
                self.detail_density.reshape(-1),
                self.detail_sh.reshape(-1),
            ])
            return (detail.abs() < threshold).float().mean().item()

    def merged_grid(self) -> torch.Tensor:
        """Return the final merged (base + detail) grid, detached."""
        return self.grid.detach()
