"""Evaluate trained wavelet volume — render test views and compute metrics.

Supports WaveletVolume (dense, <= 512³) and TiledWaveletVolume (sparse, 1024³+).
"""

import argparse
import os

import torch
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from wavelet_volume import WaveletVolume
from tiled_wavelet_volume import TiledWaveletVolume
from renderer import render_image
from metrics import psnr, ssim


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, dispatching to the correct class.

    Returns:
        (model, config, ckpt_info) tuple.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Determine model type from state dict keys
    state = ckpt["model_state_dict"]
    is_tiled = any(key.startswith("sparse_details.") for key in state)

    if is_tiled:
        # Reconstruct occupancy masks from state dict
        # SparseDetailLevel stores occupancy as a buffer
        occupancy_masks = {}
        for key in state:
            if key.startswith("sparse_details.") and key.endswith(".occupancy"):
                # e.g. "sparse_details.0.occupancy" → sparse index 0
                parts = key.split(".")
                sparse_idx = int(parts[1])
                level_idx = config.base_level + 1 + sparse_idx
                occupancy_masks[level_idx] = state[key]

        model = TiledWaveletVolume(
            base_resolution=config.base_resolution,
            decomp_levels=config.decomp_levels,
            num_channels=config.num_channels,
            channels_per_level=getattr(config, "channels_per_level", None),
            wavelet=config.wavelet,
            scene_bound=config.scene_bound,
            tile_size=getattr(config, "tile_size", 64),
            base_level=getattr(config, "base_level", 2),
            occupancy_masks=occupancy_masks if occupancy_masks else None,
        ).to(device)
    else:
        model = WaveletVolume(
            base_resolution=config.base_resolution,
            decomp_levels=config.decomp_levels,
            num_channels=config.num_channels,
            wavelet=config.wavelet,
            scene_bound=config.scene_bound,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config, ckpt


def evaluate(checkpoint_path: str, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, config, ckpt = load_model(checkpoint_path, device)

    print(f"Loaded checkpoint: iter={ckpt.get('iteration', 'N/A')}, "
          f"train PSNR={ckpt.get('psnr', 'N/A')}")

    is_tiled = isinstance(model, TiledWaveletVolume)
    if is_tiled:
        mem = model.memory_summary()
        print(f"Model: TiledWaveletVolume, {mem['total_params_M']:.0f}M params, "
              f"{mem['total_mb']:.0f} MB")
    else:
        print(f"Model: WaveletVolume, {config.target_resolution}³ × {config.num_channels}ch, "
              f"{model.total_params() / 1e6:.1f}M params")

    # Data
    data = NerfSyntheticDataset(
        config.data_dir, config.scene, split,
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    # Output
    out_dir = os.path.join(config.output_dir, config.scene, f"eval_{split}")
    os.makedirs(out_dir, exist_ok=True)

    # Render all views
    psnr_vals = []
    ssim_vals = []

    for idx in range(len(data)):
        print(f"  Rendering {split} view {idx + 1}/{len(data)}...", end="", flush=True)

        with torch.no_grad():
            result = render_image(
                model, data.poses[idx],
                config.train_resolution, config.train_resolution,
                data.focal, config,
            )

        val_psnr = psnr(result["rgb"], data.images[idx])
        val_ssim = ssim(result["rgb"], data.images[idx]).item()
        psnr_vals.append(val_psnr)
        ssim_vals.append(val_ssim)

        print(f" PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

        # Save image
        img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img).save(os.path.join(out_dir, f"{idx:04d}.png"))

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    print(f"\n{split} results ({len(data)} views):")
    print(f"  Avg PSNR: {avg_psnr:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim:.4f}")

    if not is_tiled:
        print(f"  Model size: {model.total_params() * 4 / 1024**2:.0f} MB (fp32), "
              f"{model.effective_size_bytes() / 1024**2:.0f} MB effective (fp16 non-zero)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.split)
