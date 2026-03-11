"""Image quality metrics: PSNR, SSIM."""

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between two images.

    Args:
        prediction: (..., 3) predicted RGB in [0, 1]
        target: (..., 3) ground truth RGB in [0, 1]

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(prediction, target)
    return (-10.0 * torch.log10(mse)).item()


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM between two images (differentiable).

    Args:
        prediction: (H, W, 3) predicted RGB in [0, 1]
        target: (H, W, 3) ground truth RGB in [0, 1]

    Returns:
        Scalar SSIM value (higher is better, max 1.0)
    """
    # Reshape to (1, 3, H, W) for conv2d
    pred = prediction.permute(2, 0, 1).unsqueeze(0)
    tgt = target.permute(2, 0, 1).unsqueeze(0)

    # Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=prediction.device)
    coords -= window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    window = g.outer(g)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=3)
    mu2 = F.conv2d(tgt, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(tgt ** 2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * tgt, window, padding=pad, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()
