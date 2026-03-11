"""Post-training wavelet analysis for frequency-matched PoC.

Analyzes trained grids via forward DWT to compare wavelet coefficient
distributions, rate-distortion curves, and coarse-only reconstruction
quality across training methods.

Usage:
    python wavelet_analysis.py \
        --checkpoints output/fm_poc/lego/fm/final.pt \
                      output/fm_poc/lego/sp/final.pt \
                      output/fm_poc/lego/ss/final.pt \
        --labels FM SP SS \
        --scene lego
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import ptwt
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from direct_grid_volume import DirectGridVolume
from renderer import render_image
from metrics import psnr, ssim


def load_grid_from_checkpoint(ckpt_path: str, device: str = "cpu") -> torch.Tensor:
    """Load the merged (base+detail) grid from a checkpoint.

    Handles both DirectGridVolume and ResidualGridVolume checkpoints.

    Returns:
        (1, C, R, R, R) grid tensor
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer pre-computed merged grid (saved by train_frequency_matched)
    if "merged_grid" in ckpt:
        return ckpt["merged_grid"]

    # Fall back to reconstructing from state dict
    state = ckpt["model_state_dict"]
    if "density_grid" in state:
        # DirectGridVolume
        return torch.cat([state["density_grid"], state["sh_grid"]], dim=1)
    elif "detail_density" in state:
        # ResidualGridVolume
        density = state["base_density"] + state["detail_density"]
        sh = state["base_sh"] + state["detail_sh"]
        return torch.cat([density, sh], dim=1)
    else:
        raise ValueError(f"Unknown checkpoint format: {list(state.keys())[:5]}")


def load_model_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load a model from a checkpoint for rendering."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    resolution = ckpt.get("resolution", 512)

    # Always load as DirectGridVolume with the merged grid
    grid = load_grid_from_checkpoint(ckpt_path, device="cpu")
    model = DirectGridVolume(
        resolution=resolution,
        num_channels=config.num_channels,
        scene_bound=config.scene_bound,
    )
    with torch.no_grad():
        model.density_grid.data.copy_(grid[:, :1])
        model.sh_grid.data.copy_(grid[:, 1:])
    return model.to(device), config


def analyze_coefficients(grid: torch.Tensor, wavelet: str = "bior4.4",
                         levels: int = 3) -> dict:
    """Compute DWT and analyze coefficient statistics.

    Args:
        grid: (1, C, R, R, R) dense volume
        wavelet: wavelet family
        levels: number of decomposition levels

    Returns:
        dict with per-level statistics
    """
    with torch.no_grad():
        coeffs = ptwt.wavedec3(grid.cpu().float(), wavelet, level=levels)

    results = {"levels": levels, "wavelet": wavelet}

    # Approximation statistics
    approx = coeffs[0]
    results["approx"] = {
        "shape": list(approx.shape),
        "mean_abs": approx.abs().mean().item(),
        "max_abs": approx.abs().max().item(),
        "energy": (approx ** 2).sum().item(),
        "numel": approx.numel(),
    }

    total_energy = (approx ** 2).sum().item()
    total_numel = approx.numel()

    # Detail level statistics
    results["details"] = []
    for level_idx in range(levels):
        detail_dict = coeffs[level_idx + 1]
        # Stack all 7 subbands
        all_details = torch.cat(
            [detail_dict[k].reshape(-1) for k in detail_dict], dim=0
        )

        energy = (all_details ** 2).sum().item()
        total_energy += energy
        total_numel += all_details.numel()

        # Sparsity: fraction of coefficients below threshold
        thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
        sparsity = {
            f"below_{t}": (all_details.abs() < t).float().mean().item()
            for t in thresholds
        }

        detail_size = list(detail_dict.values())[0].shape[-1]
        results["details"].append({
            "level": level_idx,
            "detail_size": detail_size,
            "mean_abs": all_details.abs().mean().item(),
            "max_abs": all_details.abs().max().item(),
            "std": all_details.std().item(),
            "energy": energy,
            "numel": all_details.numel(),
            "sparsity": sparsity,
        })

    results["total_energy"] = total_energy
    results["total_numel"] = total_numel

    # Energy distribution across levels
    results["approx"]["energy_fraction"] = results["approx"]["energy"] / total_energy
    for d in results["details"]:
        d["energy_fraction"] = d["energy"] / total_energy

    return results, coeffs


def rate_distortion(
    grid: torch.Tensor,
    test_data: NerfSyntheticDataset,
    config: Config,
    wavelet: str = "bior4.4",
    levels: int = 3,
    n_thresholds: int = 20,
    n_views: int = 10,
) -> list[dict]:
    """Compute rate-distortion curve by thresholding wavelet coefficients.

    Progressively zeroes out small coefficients and measures reconstruction
    quality, showing how many coefficients are needed for a given PSNR.

    Returns:
        list of dicts with (threshold, retained_fraction, psnr, ssim)
    """
    device = next(iter([grid.device]))
    if device.type == "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        coeffs = ptwt.wavedec3(grid.cpu().float(), wavelet, level=levels)

    # Collect all coefficient magnitudes to determine thresholds
    all_magnitudes = []
    all_magnitudes.append(coeffs[0].abs().reshape(-1))
    for level_idx in range(levels):
        for k in coeffs[level_idx + 1]:
            all_magnitudes.append(coeffs[level_idx + 1][k].abs().reshape(-1))
    all_magnitudes = torch.cat(all_magnitudes)

    # Log-spaced thresholds from 0 (keep all) to max magnitude
    max_mag = all_magnitudes.max().item()
    thresholds = [0.0] + torch.logspace(-4, np.log10(max_mag), n_thresholds - 1).tolist()

    results = []
    for threshold in thresholds:
        # Threshold coefficients
        thresholded_coeffs = [coeffs[0].clone()]
        if threshold > 0:
            thresholded_coeffs[0][thresholded_coeffs[0].abs() < threshold] = 0

        for level_idx in range(levels):
            d = {}
            for k in coeffs[level_idx + 1]:
                c = coeffs[level_idx + 1][k].clone()
                if threshold > 0:
                    c[c.abs() < threshold] = 0
                d[k] = c
            thresholded_coeffs.append(d)

        # Count retained coefficients
        total = all_magnitudes.numel()
        retained = (all_magnitudes >= threshold).sum().item() if threshold > 0 else total
        retained_fraction = retained / total

        # Reconstruct grid via IDWT
        reconstructed = ptwt.waverec3(thresholded_coeffs, wavelet)
        R = grid.shape[-1]
        reconstructed = reconstructed[:, :, :R, :R, :R]

        # Create temporary model for rendering
        model = DirectGridVolume(
            resolution=R,
            num_channels=grid.shape[1],
            scene_bound=config.scene_bound,
        ).to(device)
        with torch.no_grad():
            model.density_grid.data.copy_(reconstructed[:, :1].to(device))
            model.sh_grid.data.copy_(reconstructed[:, 1:].to(device))

        # Evaluate on a few test views
        psnr_vals = []
        ssim_vals = []
        model.eval()
        for idx in range(min(n_views, len(test_data))):
            with torch.no_grad():
                result = render_image(
                    model, test_data.poses[idx],
                    test_data.resolution, test_data.resolution,
                    test_data.focal, config,
                )
            psnr_vals.append(psnr(result["rgb"], test_data.images[idx]))
            ssim_vals.append(ssim(result["rgb"], test_data.images[idx]).item())

        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)

        results.append({
            "threshold": threshold,
            "retained_fraction": retained_fraction,
            "retained_count": retained,
            "total_count": total,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
        })
        print(f"  threshold={threshold:.4e}: retain={retained_fraction:.3f}, "
              f"PSNR={avg_psnr:.2f} dB")

        del model
        torch.cuda.empty_cache()

    return results


def render_coarse_only(
    grid: torch.Tensor,
    test_data: NerfSyntheticDataset,
    config: Config,
    wavelet: str = "bior4.4",
    levels: int = 3,
    out_dir: str = "output",
    tag: str = "",
    n_views: int = 10,
    eval_resolution: int | None = None,
) -> dict:
    """Reconstruct from approximation coefficients only and render.

    Zeroes all detail coefficients, keeping only the low-frequency
    approximation. This tests the quality of the coarse representation.

    Args:
        eval_resolution: if set, evaluate at this resolution instead of test_data's
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = grid.shape[-1]

    with torch.no_grad():
        coeffs = ptwt.wavedec3(grid.cpu().float(), wavelet, level=levels)

    # Zero all detail coefficients
    zero_coeffs = [coeffs[0]]
    for level_idx in range(levels):
        d = {}
        for k in coeffs[level_idx + 1]:
            d[k] = torch.zeros_like(coeffs[level_idx + 1][k])
        zero_coeffs.append(d)

    # Reconstruct coarse-only grid
    coarse_grid = ptwt.waverec3(zero_coeffs, wavelet)
    coarse_grid = coarse_grid[:, :, :R, :R, :R]

    # Create model for rendering
    model = DirectGridVolume(
        resolution=R,
        num_channels=grid.shape[1],
        scene_bound=config.scene_bound,
    ).to(device)
    with torch.no_grad():
        model.density_grid.data.copy_(coarse_grid[:, :1].to(device))
        model.sh_grid.data.copy_(coarse_grid[:, 1:].to(device))

    # If evaluating at a different resolution, reload test data
    if eval_resolution is not None and eval_resolution != test_data.resolution:
        eval_data = NerfSyntheticDataset(
            config.data_dir, config.scene, "test",
            resolution=eval_resolution,
            white_bg=config.white_background,
            device=device,
        )
    else:
        eval_data = test_data
        eval_resolution = test_data.resolution

    render_dir = os.path.join(out_dir, f"coarse_{tag}")
    os.makedirs(render_dir, exist_ok=True)

    model.eval()
    psnr_vals = []
    ssim_vals = []

    for idx in range(min(n_views, len(eval_data))):
        with torch.no_grad():
            result = render_image(
                model, eval_data.poses[idx],
                eval_resolution, eval_resolution,
                eval_data.focal, config,
            )
        val_p = psnr(result["rgb"], eval_data.images[idx])
        val_s = ssim(result["rgb"], eval_data.images[idx]).item()
        psnr_vals.append(val_p)
        ssim_vals.append(val_s)

        img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(render_dir, f"coarse_{idx:03d}.png")
        )

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)

    del model
    torch.cuda.empty_cache()

    return {
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "eval_resolution": eval_resolution,
        "n_views": len(psnr_vals),
    }


def compare_methods(
    checkpoints: list[str],
    labels: list[str],
    config: Config,
    out_dir: str,
    wavelet: str = "bior4.4",
    levels: int = 3,
):
    """Run full comparison analysis across multiple trained models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Load test data at multiple resolutions for coarse evaluation
    test_800 = NerfSyntheticDataset(
        config.data_dir, config.scene, "test",
        resolution=800, white_bg=config.white_background, device=device,
    )

    all_results = {}

    for ckpt_path, label in zip(checkpoints, labels):
        print(f"\n{'='*60}")
        print(f"Analyzing: {label} ({ckpt_path})")
        print(f"{'='*60}")

        grid = load_grid_from_checkpoint(ckpt_path)
        print(f"  Grid shape: {list(grid.shape)}")

        # 1. Coefficient analysis
        print("\n  [1] Wavelet coefficient analysis...")
        coeff_stats, coeffs = analyze_coefficients(grid, wavelet, levels)
        print(f"      Approx energy fraction: "
              f"{coeff_stats['approx']['energy_fraction']:.3f}")
        for d in coeff_stats["details"]:
            print(f"      Detail {d['level']}: energy={d['energy_fraction']:.3f}, "
                  f"sparse(1e-3)={d['sparsity']['below_0.001']:.3f}")

        # 2. Rate-distortion curve
        print("\n  [2] Rate-distortion curve...")
        rd = rate_distortion(
            grid, test_800, config, wavelet, levels,
            n_thresholds=15, n_views=5,
        )

        # 3. Coarse-only reconstruction at 100px and 800px
        print("\n  [3] Coarse-only reconstruction...")
        coarse_100 = render_coarse_only(
            grid, test_800, config, wavelet, levels,
            out_dir=out_dir, tag=f"{label}_100px",
            n_views=5, eval_resolution=100,
        )
        print(f"      Coarse at 100px: PSNR={coarse_100['avg_psnr']:.2f} dB")

        coarse_800 = render_coarse_only(
            grid, test_800, config, wavelet, levels,
            out_dir=out_dir, tag=f"{label}_800px",
            n_views=5, eval_resolution=800,
        )
        print(f"      Coarse at 800px: PSNR={coarse_800['avg_psnr']:.2f} dB")

        all_results[label] = {
            "coefficient_stats": coeff_stats,
            "rate_distortion": rd,
            "coarse_100px": coarse_100,
            "coarse_800px": coarse_800,
        }

    # Save comparison report
    # Convert non-serializable types
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    report_path = os.path.join(out_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=sanitize)
    print(f"\nComparison report saved to: {report_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Method':<10} {'Coarse 100px':>14} {'Coarse 800px':>14} "
          f"{'RD@50%':>10}")
    print(f"{'-'*58}")
    for label in labels:
        r = all_results[label]
        c100 = r["coarse_100px"]["avg_psnr"]
        c800 = r["coarse_800px"]["avg_psnr"]
        # Find PSNR at ~50% retention
        rd = r["rate_distortion"]
        rd_50 = min(rd, key=lambda x: abs(x["retained_fraction"] - 0.5))
        print(f"{label:<10} {c100:>12.2f} dB {c800:>12.2f} dB "
              f"{rd_50['psnr']:>8.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wavelet analysis for frequency-matched PoC"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to final.pt checkpoints")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each checkpoint (e.g. FM SP SS)")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--data_dir", default="data/nerf_synthetic")
    parser.add_argument("--output_dir", default="output/fm_poc/analysis")
    parser.add_argument("--wavelet", default="bior4.4")
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()

    config = Config(scene=args.scene, data_dir=args.data_dir)

    compare_methods(
        args.checkpoints, args.labels, config,
        args.output_dir, args.wavelet, args.levels,
    )
