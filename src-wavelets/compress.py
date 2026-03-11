"""Compression analysis — prune coefficients and measure rate-distortion."""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from wavelet_volume import WaveletVolume
from renderer import render_image
from metrics import psnr


def compression_analysis(checkpoint_path: str):
    # Load
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    device = torch.device(config.device)

    # Data
    data = NerfSyntheticDataset(
        config.data_dir, config.scene, "test",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    # Test views to evaluate (subset for speed)
    test_indices = list(range(min(10, len(data))))

    out_dir = os.path.join(config.output_dir, config.scene, "compression")
    os.makedirs(out_dir, exist_ok=True)

    # Sweep keep ratios
    keep_ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]

    results = []

    for ratio in keep_ratios:
        # Load fresh model for each ratio (pruning is destructive)
        model = WaveletVolume(
            base_resolution=config.base_resolution,
            decomp_levels=config.decomp_levels,
            num_channels=config.num_channels,
            wavelet=config.wavelet,
            scene_bound=config.scene_bound,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Prune
        if ratio < 1.0:
            stats = model.prune(ratio)
            actual_ratio = stats["ratio"]
            print(f"\nKeep ratio={ratio:.1%} → actual={actual_ratio:.1%}")
            for level_stat in stats["per_level"]:
                print(f"  Level {level_stat['level']}: "
                      f"sparsity={level_stat['sparsity']:.1%}")
        else:
            actual_ratio = 1.0
            print(f"\nKeep ratio=100% (no pruning)")

        # Measure effective size
        size_bytes = model.effective_size_bytes()
        size_mb = size_bytes / 1024**2

        # Evaluate
        psnr_vals = []
        for idx in test_indices:
            with torch.no_grad():
                result = render_image(
                    model, data.poses[idx],
                    config.train_resolution, config.train_resolution,
                    data.focal, config,
                )
            psnr_vals.append(psnr(result["rgb"], data.images[idx]))

        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        print(f"  Size={size_mb:.1f} MB, PSNR={avg_psnr:.2f} dB")

        results.append({
            "keep_ratio": ratio,
            "actual_ratio": actual_ratio,
            "size_mb": size_mb,
            "psnr": avg_psnr,
        })

    # Plot rate-distortion curve
    sizes = [r["size_mb"] for r in results]
    psnrs = [r["psnr"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(sizes, psnrs, "o-", linewidth=2, markersize=8)
    for r in results:
        ax.annotate(
            f"{r['keep_ratio']:.0%}",
            (r["size_mb"], r["psnr"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )
    ax.set_xlabel("Model Size (MB, fp16 non-zero)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Rate-Distortion: {config.scene} ({config.wavelet} wavelets)")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rate_distortion.png"), dpi=150)
    print(f"\nSaved rate-distortion plot to {out_dir}/rate_distortion.png")

    # LoD visualization: render at each reconstruction level
    print("\n--- LoD Visualization ---")
    model_full = WaveletVolume(
        base_resolution=config.base_resolution,
        decomp_levels=config.decomp_levels,
        num_channels=config.num_channels,
        wavelet=config.wavelet,
        scene_bound=config.scene_bound,
    ).to(device)
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.eval()

    test_pose = data.poses[0]
    for level in range(config.decomp_levels):
        res = config.base_resolution * (2 ** (level + 1))
        print(f"  Level {level} ({res}³)...", end="", flush=True)

        with torch.no_grad():
            result = render_image(
                model_full, test_pose,
                config.train_resolution, config.train_resolution,
                data.focal, config,
                max_level=level,
            )

        val_psnr = psnr(result["rgb"], data.images[0])
        print(f" PSNR={val_psnr:.2f}")

        img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(out_dir, f"lod_level{level}_{res}.png")
        )

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    compression_analysis(args.checkpoint)
