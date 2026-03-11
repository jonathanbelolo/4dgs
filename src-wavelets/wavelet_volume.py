"""Sparse multi-resolution wavelet coefficient volume.

Stores learnable wavelet coefficients and reconstructs a dense 3D feature
volume via inverse DWT (ptwt). The volume is queried via trilinear
interpolation at arbitrary 3D points.

No MLP — the wavelet IDWT reconstruction IS the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from sh import eval_sh_color

# 3D DWT detail subband keys in ptwt convention.
# Each 3-char string: 'a' = approximation, 'd' = detail along that axis.
SUBBAND_KEYS = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")


class WaveletVolume(nn.Module):
    """3D wavelet coefficient volume with IDWT reconstruction.

    Architecture:
        Learnable wavelet coefficients → IDWT → dense feature volume → trilinear query

    Coefficient sizes are determined by a dry-run wavedec3 decomposition of the
    target volume, so they are correct for any wavelet family (Haar gives exact
    powers of 2; CDF 9/7 / bior4.4 gives irregular sizes due to filter length).

    In ptwt's convention, detail level 0 (coarsest) has the same spatial
    size as the approximation. Each subsequent level doubles in resolution.
    """

    def __init__(
        self,
        base_resolution: int = 32,
        decomp_levels: int = 3,
        num_channels: int = 28,
        wavelet: str = "bior4.4",
        scene_bound: float = 1.5,
    ):
        super().__init__()
        self.base_resolution = base_resolution
        self.decomp_levels = decomp_levels
        self.num_channels = num_channels
        self.wavelet = wavelet
        self.scene_bound = scene_bound
        self.target_resolution = base_resolution * (2 ** decomp_levels)

        # Dry-run wavedec3 on a dummy tensor to discover actual coefficient
        # sizes for this wavelet family and target resolution.
        R = self.target_resolution
        dummy = torch.zeros(1, 1, R, R, R)
        coeffs = ptwt.wavedec3(dummy, wavelet=wavelet, level=decomp_levels)
        approx_size = coeffs[0].shape[2]  # cubic, so D=H=W

        # Approximation coefficients (coarsest level)
        self.approx = nn.Parameter(
            torch.randn(1, num_channels, approx_size, approx_size, approx_size) * 0.01
        )

        # Detail coefficients at each level.
        # Sizes come from the dry-run decomposition (coeffs[1..N] are detail dicts).
        # 7 subbands per level, stored as (7, C, D, D, D).
        self.details = nn.ParameterList()
        for i in range(decomp_levels):
            # coeffs[i+1] is a dict of 7 subbands; grab size from any key
            detail_dict = coeffs[i + 1]
            D = next(iter(detail_dict.values())).shape[2]
            self.details.append(nn.Parameter(
                torch.randn(7, num_channels, D, D, D) * 0.001
            ))

    def _make_detail_dict(self, level: int) -> dict:
        """Build ptwt detail dict from stored parameter tensor."""
        detail_param = self.details[level]  # (7, C, D, D, D)
        return {
            key: detail_param[j].unsqueeze(0)  # → (1, C, D, D, D)
            for j, key in enumerate(SUBBAND_KEYS)
        }

    def reconstruct(self, max_level: int | None = None) -> torch.Tensor:
        """Reconstruct dense volume via incremental inverse DWT.

        Args:
            max_level: Maximum detail level to include (0-indexed).
                       None = use all levels → target_resolution³
                       -1 = approx only (no detail) → approx_size³
                       0 = approx + detail_0 → next resolution level

        Returns:
            volume: (1, C, R, R, R) dense feature volume
        """
        if max_level is None:
            max_level = self.decomp_levels - 1
        max_level = min(max_level, self.decomp_levels - 1)

        if max_level < 0:
            return self.approx

        # Incremental IDWT: one level at a time, each building on the previous
        # Disable autocast: ptwt does not support bf16.
        volume = self.approx
        with torch.amp.autocast('cuda', enabled=False):
          for i in range(max_level + 1):
            detail_dict = self._make_detail_dict(i)
            volume = ptwt.waverec3((volume, detail_dict), wavelet=self.wavelet)
            # Truncate: IDWT may produce 1 extra voxel when the original
            # signal had odd length (DWT rounds up, IDWT overshoots by 1).
            if i + 1 < self.decomp_levels:
                expected = self.details[i + 1].shape[2]
                if volume.shape[2] > expected:
                    volume = volume[:, :, :expected, :expected, :expected]

        return volume

    def reconstruct_pair(
        self, fine_level: int, coarse_level: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct fine and coarse volumes in a single pass.

        The coarse volume is captured as an intermediate result of the
        fine IDWT, so no computation is duplicated.

        Args:
            fine_level: detail level for the fine volume
            coarse_level: detail level for the coarse volume (must be <= fine_level)

        Returns:
            (fine_volume, coarse_volume) — both (1, C, R, R, R)
        """
        fine_level = min(fine_level, self.decomp_levels - 1)
        coarse_level = min(coarse_level, fine_level)

        volume = self.approx
        coarse_volume = self.approx if coarse_level < 0 else None

        # Disable autocast: ptwt does not support bf16.
        with torch.amp.autocast('cuda', enabled=False):
            for i in range(fine_level + 1):
                detail_dict = self._make_detail_dict(i)
                volume = ptwt.waverec3((volume, detail_dict), wavelet=self.wavelet)
                # Truncate off-by-one from IDWT (see reconstruct())
                if i + 1 < self.decomp_levels:
                    expected = self.details[i + 1].shape[2]
                    if volume.shape[2] > expected:
                        volume = volume[:, :, :expected, :expected, :expected]
                if i == coarse_level:
                    coarse_volume = volume

        # If coarse_level == fine_level, both point to the same tensor
        if coarse_volume is None:
            coarse_volume = volume

        return volume, coarse_volume

    def query(self, xyz: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """Query the volume at arbitrary 3D points via trilinear interpolation.

        Args:
            xyz: (N, 3) world-space coordinates in [-scene_bound, scene_bound]
            volume: (1, C, R, R, R) reconstructed volume

        Returns:
            features: (N, C) interpolated features
        """
        # Normalize to [-1, 1] for grid_sample
        grid = xyz / self.scene_bound  # (N, 3)

        # grid_sample 5D: input (1, C, D, H, W), grid (1, D_out, H_out, W_out, 3)
        # Treat N query points as (N, 1, 1) spatial output grid
        grid = grid.view(1, -1, 1, 1, 3)  # (1, N, 1, 1, 3)

        features = F.grid_sample(
            volume, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (1, C, N, 1, 1)

        # → (N, C)
        features = features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)

        return features

    def decode(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode features into density and view-dependent RGB.

        Args:
            features: (N, C) raw features from volume query.
                      Channel 0 = raw density, channels 1:28 = SH coefficients.
            view_dirs: (N, 3) unit view direction vectors

        Returns:
            density: (N,) volume density (non-negative)
            rgb: (N, 3) color in [0, 1]
        """
        raw_density = features[:, 0]
        sh_coeffs = features[:, 1:]  # (N, 27) = 9 SH × 3 colors

        # Density: softplus with bias toward transparency
        density = F.relu(raw_density)

        # Color: SH evaluation with view direction
        rgb = eval_sh_color(sh_coeffs, view_dirs)

        return density, rgb

    def prune(self, keep_ratio: float) -> dict:
        """Zero out small detail coefficients by magnitude.

        Args:
            keep_ratio: fraction of detail coefficients to retain

        Returns:
            dict with per-level sparsity info
        """
        # Gather all detail coefficient magnitudes
        all_magnitudes = torch.cat([d.data.abs().flatten() for d in self.details])
        total = all_magnitudes.numel()
        n_keep = int(total * keep_ratio)

        if n_keep >= total:
            return {"kept": total, "total": total, "ratio": 1.0, "per_level": []}

        # Find threshold via kth-largest value
        threshold = torch.kthvalue(all_magnitudes, total - n_keep + 1).values

        # Apply mask per level
        stats = {"per_level": [], "threshold": threshold.item()}
        total_kept = 0
        total_count = 0
        for i, detail in enumerate(self.details):
            mask = detail.data.abs() >= threshold
            detail.data *= mask.float()
            kept = mask.sum().item()
            count = mask.numel()
            total_kept += kept
            total_count += count
            stats["per_level"].append({
                "level": i,
                "kept": kept,
                "total": count,
                "sparsity": 1.0 - kept / count,
            })

        stats["kept"] = total_kept
        stats["total"] = total_count
        stats["ratio"] = total_kept / total_count
        return stats

    def effective_size_bytes(self) -> int:
        """Count non-zero parameters × 2 bytes (fp16 equivalent)."""
        count = (self.approx.data != 0).sum().item()
        for detail in self.details:
            count += (detail.data != 0).sum().item()
        return count * 2

    def total_params(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())
