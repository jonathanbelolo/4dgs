"""Verify DWT conversion quality by rendering test views.

Lightweight alternative to eval.py — renders only a few views with
detailed progress logging to diagnose loading/memory issues.

Patches the SH decode to match svox2's convention:
  - Color-major channel order: [R0..R8, G0..G8, B0..B8]
  - SH basis signs: negative on bases 1,3,5,7

Usage:
    python verify_conversion.py \
        --checkpoint output/lego/wavelet_converted.pt \
        --num_views 5
"""

import argparse
import gc
import os
import time
import traceback

import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from tiled_wavelet_volume import TiledWaveletVolume
from renderer import render_image
from metrics import psnr, ssim


# svox2's SH normalization constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]


def eval_sh_bases_svox2(directions: torch.Tensor) -> torch.Tensor:
    """Evaluate SH bases with svox2's sign convention."""
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    return torch.stack([
        torch.full_like(x, SH_C0),            # 0: Y_0^0
        -SH_C1 * y,                            # 1: Y_1^{-1} (NEGATIVE y)
        SH_C1 * z,                             # 2: Y_1^0
        -SH_C1 * x,                            # 3: Y_1^1 (NEGATIVE x)
        SH_C2[0] * x * y,                      # 4: Y_2^{-2}
        SH_C2[1] * y * z,                      # 5: Y_2^{-1} (NEGATIVE yz)
        SH_C2[2] * (2 * z * z - x * x - y * y),  # 6: Y_2^0
        SH_C2[3] * x * z,                      # 7: Y_2^1 (NEGATIVE xz)
        SH_C2[4] * (x * x - y * y),            # 8: Y_2^2
    ], dim=-1)


def decode_svox2(self, features, view_dirs):
    """Decode with svox2's SH convention.

    svox2 stores SH as color-major: [R0..R8, G0..G8, B0..B8]
    and uses different basis signs than our default.
    """
    raw_density = features[:, 0]
    sh_coeffs = features[:, 1:]  # (N, 27) in svox2's color-major order

    density = F.relu(raw_density)

    # Evaluate SH bases with svox2 convention
    basis = eval_sh_bases_svox2(view_dirs)  # (N, 9)

    # Reshape as color-major: (N, 3, 9) — matches svox2's storage
    sh = sh_coeffs.view(-1, 3, 9)  # [R_bases, G_bases, B_bases]

    # Dot product: for each color c, rgb_c = sum_b(basis[b] * sh[c, b])
    # basis: (N, 9) → (N, 1, 9), sh: (N, 3, 9)
    rgb = (basis.unsqueeze(1) * sh).sum(dim=2)  # (N, 3)

    rgb = torch.sigmoid(rgb)
    return density, rgb


def verify(checkpoint_path: str, num_views: int = 5, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, {total_mem:.0f} GB")

    # --- Step 1: Load checkpoint to CPU ---
    t0 = time.time()
    print(f"\n[1/5] Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    del ckpt
    gc.collect()

    n_keys = len(state_dict)
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"  Loaded in {time.time() - t0:.1f}s: {n_keys} keys, "
          f"{total_bytes / 1e9:.1f} GB")

    # --- Step 2: Reconstruct occupancy masks from state dict ---
    t1 = time.time()
    print(f"\n[2/5] Reconstructing model structure...")
    occupancy_masks = {}
    for key in state_dict:
        if key.startswith("sparse_details.") and key.endswith(".occupancy"):
            sparse_idx = int(key.split(".")[1])
            level_idx = config.base_level + 1 + sparse_idx
            occupancy_masks[level_idx] = state_dict[key]
            occ = state_dict[key]
            n_occ = occ.sum().item()
            print(f"  Level {level_idx}: {n_occ}/{occ.numel()} tiles occupied "
                  f"({100*n_occ/occ.numel():.1f}%)")

    # --- Step 3: Create model on CPU, load state dict, move to GPU ---
    print(f"\n[3/5] Creating TiledWaveletVolume on CPU...")
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
    )

    mem = model.memory_summary()
    print(f"  Model: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")

    print(f"  Loading state dict...")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    print(f"  State dict loaded in {time.time() - t1:.1f}s")

    # Patch decode to use svox2's SH convention
    import types
    model.decode = types.MethodType(decode_svox2, model)
    print("  Patched decode() for svox2 SH convention")

    print(f"  Moving to {device}...")
    t_move = time.time()
    model = model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  On GPU in {time.time() - t_move:.1f}s, {used:.1f} GB allocated")
    gc.collect()

    # --- Step 4: Load dataset ---
    t2 = time.time()
    print(f"\n[4/5] Loading {config.scene} {split} dataset...")
    data = NerfSyntheticDataset(
        config.data_dir, config.scene, split,
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    num_views = min(num_views, len(data))
    print(f"  {len(data)} views, rendering {num_views}")

    # --- Step 5: Render and evaluate ---
    print(f"\n[5/5] Rendering {num_views} views at {config.train_resolution}px...")

    out_dir = os.path.join(config.output_dir, config.scene, "verify")
    os.makedirs(out_dir, exist_ok=True)

    psnr_vals = []
    ssim_vals = []

    for idx in range(num_views):
        t_render = time.time()
        print(f"  View {idx + 1}/{num_views}...", end="", flush=True)

        try:
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

            render_time = time.time() - t_render
            print(f" PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f} ({render_time:.1f}s)")

            # Save rendered image
            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(os.path.join(out_dir, f"view_{idx:03d}.png"))

        except Exception:
            print(f" ERROR!")
            traceback.print_exc()
            continue

    if psnr_vals:
        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)
        print(f"\n{'='*50}")
        print(f"Results ({num_views} {split} views):")
        print(f"  Avg PSNR: {avg_psnr:.2f} dB")
        print(f"  Avg SSIM: {avg_ssim:.4f}")
        print(f"  (svox2 baseline: 33.20 dB)")
        print(f"  Total time: {time.time() - t0:.0f}s")
        print(f"{'='*50}")
    else:
        print("\nNo views rendered successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    verify(args.checkpoint, args.num_views, args.split)
