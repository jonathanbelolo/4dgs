"""Real Spherical Harmonics evaluation up to degree 2.

SH degree 2 has 9 basis functions (l=0: 1, l=1: 3, l=2: 5).
We use the real-valued SH basis (not complex), which is standard
for graphics applications.

Convention: coefficients are stored as (9, 3) per voxel — 9 SH
basis functions x 3 color channels = 27 values.
"""

import torch

# Precomputed normalization constants for real SH
C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))
C1 = 0.4886025119029199    # sqrt(3) / (2 * sqrt(pi))
C2_0 = 1.0925484305920792  # sqrt(15) / (2 * sqrt(pi))
C2_1 = 0.31539156525252005 # sqrt(5) / (4 * sqrt(pi))
C2_2 = 0.5462742152960396  # sqrt(15) / (4 * sqrt(pi))


def eval_sh_bases(directions: torch.Tensor) -> torch.Tensor:
    """Evaluate real SH basis functions at given directions.

    Args:
        directions: (..., 3) unit vectors (x, y, z)

    Returns:
        basis: (..., 9) SH basis values for degree 0, 1, 2
    """
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    basis = torch.stack([
        # l=0
        torch.full_like(x, C0),                       # Y_0^0
        # l=1
        C1 * y,                                        # Y_1^{-1}
        C1 * z,                                        # Y_1^0
        C1 * x,                                        # Y_1^1
        # l=2
        C2_0 * x * y,                                  # Y_2^{-2}
        C2_0 * y * z,                                  # Y_2^{-1}
        C2_1 * (2.0 * z * z - x * x - y * y),         # Y_2^0
        C2_0 * x * z,                                  # Y_2^1
        C2_2 * (x * x - y * y),                        # Y_2^2
    ], dim=-1)

    return basis


def eval_sh_color(
    sh_coeffs: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate SH-encoded color at given view directions.

    Args:
        sh_coeffs: (N, 27) SH coefficients — 9 basis x 3 colors, stored as
                   [sh0_r, sh0_g, sh0_b, sh1_r, sh1_g, sh1_b, ..., sh8_r, sh8_g, sh8_b]
        directions: (N, 3) unit view direction vectors

    Returns:
        rgb: (N, 3) color values in [0, 1]
    """
    basis = eval_sh_bases(directions)  # (N, 9)

    # Reshape SH coeffs to (N, 9, 3)
    sh = sh_coeffs.view(-1, 9, 3)

    # Dot product: sum over SH basis functions
    # (N, 9, 1) * (N, 9, 3) → sum over dim=1 → (N, 3)
    rgb = (basis.unsqueeze(-1) * sh).sum(dim=1)

    # Sigmoid activation for bounded output
    rgb = torch.sigmoid(rgb)

    return rgb
