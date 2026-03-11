"""svox2 → Wavelet conversion and fine-level training.

Pipeline:
1. Load svox2 1030³ checkpoint (sparse → dense, channel-batched)
2. Forward DWT (5 levels, batched by channel) → exact wavelet coefficients
3. Populate TiledWaveletVolume (dense base + sparse DWT levels + empty fine levels)
4. Train sparse fine levels through tiled IDWT

The DWT is lossless — wavelet coefficients exactly represent the trained grid.
Fine levels learn the super-resolution refinement (1030→2052→4096).
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import ptwt
from PIL import Image

from config import Config
from data import NerfSyntheticDataset
from load_svox2 import load_svox2_sparse, expand_channels_to_dense, get_occupancy_from_sparse
from tiled_wavelet_volume import TiledWaveletVolume, _get_level_sizes
from sparse_coefficients import SparseDetailLevel
from wavelet_volume import SUBBAND_KEYS
from renderer import render_rays, render_image
from metrics import psnr, ssim
from occupancy import occupancy_to_tile_mask, dilate_3d, compute_occupancy_stats


def convert_svox2_to_wavelet(config: Config) -> tuple[TiledWaveletVolume, int]:
    """Convert svox2 checkpoint to TiledWaveletVolume via batched forward DWT.

    The full 28ch × 1030³ grid is ~114 GB, so we process channels in batches
    to keep peak memory manageable (~20 GB per batch).

    Returns:
        (tiled_model, n_dwt_sparse): Populated TiledWaveletVolume with DWT
        coefficients in dense base and sparse DWT-initialized levels (fine
        levels are empty), and the count of DWT-initialized sparse levels.
    """
    device = torch.device(config.device)

    # --- Step 1: Load sparse svox2 data (~5 GB) ---
    print("=== Step 1: Loading svox2 checkpoint ===")
    links, density_data, sh_data, meta = load_svox2_sparse(config.svox2_ckpt)
    R = meta["resolution"]

    # --- Step 2: Compute occupancy ---
    print("\n=== Step 2: Computing occupancy ===")
    occupancy = get_occupancy_from_sparse(links, density_data, threshold=0.0)
    occupancy = dilate_3d(occupancy, kernel_size=config.occupancy_dilate)
    occ_pct = 100.0 * occupancy.float().mean().item()
    print(f"Occupancy: {occ_pct:.1f}% of {R}³ volume")

    # --- Step 3: Determine wavelet structure ---
    print(f"\n=== Step 3: Wavelet decomposition structure ===")
    level_sizes = _get_level_sizes(
        config.base_resolution * (2 ** config.decomp_levels),
        config.decomp_levels, config.wavelet,
    )
    print(f"decomp_levels={config.decomp_levels}, base_level={config.base_level}")
    print(f"level_sizes={level_sizes}")

    # Find how many DWT levels the svox2 grid gives us
    # The grid resolution R should match a level_sizes entry.
    # Search from the end: level_sizes can have duplicates (e.g. [40, 40, ...])
    # and we want the highest index (detail level, not approx).
    grid_level_idx = None
    for idx in range(len(level_sizes) - 1, -1, -1):
        if level_sizes[idx] == R:
            grid_level_idx = idx
            break
    if grid_level_idx is None:
        raise ValueError(
            f"svox2 grid resolution {R} not found in level_sizes {level_sizes}. "
            f"Ensure decomp_levels and base_resolution produce a matching size."
        )

    # DWT levels = number of detail levels below the grid position
    # level_sizes[0] = approx, level_sizes[i+1] = detail_i
    # If grid is at index 6, we have details 0..4, so 5 DWT levels
    dwt_levels = grid_level_idx - 1  # -1 because index 0 is approx
    print(f"svox2 grid at level_sizes[{grid_level_idx}] = {R}")
    print(f"Forward DWT: {dwt_levels} levels")

    # --- Step 4: Build occupancy masks for sparse levels ---
    occupancy_masks = {}
    n_dwt_sparse = 0  # sparse levels that get DWT data
    n_empty_sparse = 0  # sparse levels that start empty (trainable)

    for i in range(config.base_level + 1, config.decomp_levels):
        D = level_sizes[i + 1]
        tiles_per_axis = max(1, (D + config.tile_size - 1) // config.tile_size)

        detail_level_idx = i + 1  # position in level_sizes
        if detail_level_idx <= grid_level_idx:
            # DWT-initialized level — use occupancy mask
            occupancy_masks[i] = occupancy_to_tile_mask(occupancy, tiles_per_axis)
            n_dwt_sparse += 1
        else:
            # Fine level beyond the grid — empty (no tiles)
            occupancy_masks[i] = torch.zeros(
                tiles_per_axis, tiles_per_axis, tiles_per_axis, dtype=torch.bool,
            )
            n_empty_sparse += 1

    print(f"Sparse levels: {n_dwt_sparse} DWT-initialized (frozen), "
          f"{n_empty_sparse} empty (trainable)")

    for i, mask in occupancy_masks.items():
        n_occ = mask.sum().item()
        n_total = mask.numel()
        print(f"  Level {i} ({level_sizes[i+1]}³): "
              f"{n_occ}/{n_total} tiles occupied "
              f"({100*n_occ/n_total:.1f}%)")

    # --- Step 5: Create TiledWaveletVolume ---
    print(f"\n=== Step 5: Creating TiledWaveletVolume ===")
    tiled_model = TiledWaveletVolume(
        base_resolution=config.base_resolution,
        decomp_levels=config.decomp_levels,
        num_channels=config.num_channels,
        channels_per_level=config.channels_per_level,
        wavelet=config.wavelet,
        scene_bound=config.scene_bound,
        tile_size=config.tile_size,
        base_level=config.base_level,
        occupancy_masks=occupancy_masks,
    )

    mem = tiled_model.memory_summary()
    print(f"Model: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")

    # --- Step 6: Batched DWT and coefficient population ---
    print(f"\n=== Step 6: Batched DWT ({dwt_levels} levels) ===")
    batch_size = config.dwt_batch_channels
    total_channels = config.num_channels

    for batch_start in range(0, total_channels, batch_size):
        batch_end = min(batch_start + batch_size, total_channels)
        channels = list(range(batch_start, batch_end))
        C_batch = len(channels)

        print(f"  Channels [{batch_start}:{batch_end}] "
              f"({C_batch}ch × {R}³ = {C_batch * R**3 * 4 / 1e9:.1f} GB)...",
              end=" ", flush=True)

        # Expand these channels to dense
        dense_batch = expand_channels_to_dense(
            links, density_data, sh_data, channels,
        )  # (1, C_batch, R, R, R)

        # Forward DWT
        coeffs = ptwt.wavedec3(
            dense_batch, wavelet=config.wavelet, level=dwt_levels,
        )

        # Copy approx channels
        approx = coeffs[0]  # (1, C_batch, S_approx, S_approx, S_approx)
        C_approx = tiled_model.approx.shape[1]
        ch_end_approx = min(batch_end, C_approx)
        if batch_start < C_approx:
            tiled_model.approx.data[
                :, batch_start:ch_end_approx
            ] = approx[:, :ch_end_approx - batch_start]

        # Copy dense detail levels (0..base_level)
        for dwt_idx in range(min(dwt_levels, config.base_level + 1)):
            detail_dict = coeffs[dwt_idx + 1]
            C_level = tiled_model.dense_details[dwt_idx].shape[1]
            ch_end_level = min(batch_end, C_level)
            if batch_start < C_level:
                stacked = torch.stack(
                    [detail_dict[key].squeeze(0) for key in SUBBAND_KEYS],
                )  # (7, C_batch, D, D, D)
                c_src_end = ch_end_level - batch_start
                tiled_model.dense_details[dwt_idx].data[
                    :, batch_start:ch_end_level
                ] = stacked[:, :c_src_end]

        # Copy sparse DWT-initialized levels (base_level+1 .. last DWT level)
        for dwt_idx in range(config.base_level + 1, dwt_levels):
            detail_dict = coeffs[dwt_idx + 1]
            sparse_idx = dwt_idx - (config.base_level + 1)
            sparse_level = tiled_model.sparse_details[sparse_idx]
            C_level = sparse_level.num_channels
            ch_end_level = min(batch_end, C_level)
            if batch_start < C_level and sparse_level.n_occupied > 0:
                stacked = torch.stack(
                    [detail_dict[key].squeeze(0) for key in SUBBAND_KEYS],
                )  # (7, C_batch, D, D, D)
                c_src_end = ch_end_level - batch_start
                sparse_level.set_channels_from_dense(
                    stacked[:, :c_src_end], channel_offset=batch_start,
                )

        del dense_batch, coeffs
        print("done")

    # --- Step 7: Verify dense round-trip ---
    print(f"\n=== Step 7: Verifying DWT round-trip ===")
    with torch.no_grad():
        # Reconstruct base volume from dense levels
        base_vol = tiled_model.reconstruct_base()
        base_res = base_vol.shape[2]

        # Expand the same channels from svox2 to compare
        # (only check first 4 channels to save memory)
        check_channels = list(range(min(4, total_channels)))
        dense_check = expand_channels_to_dense(
            links, density_data, sh_data, check_channels,
        )

        # Downsample the svox2 grid to base volume resolution for comparison
        dense_downsampled = F.interpolate(
            dense_check, size=(base_res, base_res, base_res),
            mode="trilinear", align_corners=True,
        )

        # Compare (not exact due to downsampling, but should be close)
        max_err = (base_vol[:, :len(check_channels)] - dense_downsampled).abs().max().item()
        mean_err = (base_vol[:, :len(check_channels)] - dense_downsampled).abs().mean().item()
        print(f"Base volume ({base_res}³) vs downsampled svox2: "
              f"max_err={max_err:.4f}, mean_err={mean_err:.6f}")
        del dense_check, dense_downsampled, base_vol

    # Free sparse svox2 data
    del links, density_data, sh_data

    print("\nDWT conversion complete.")
    return tiled_model, n_dwt_sparse


def train_svox2_to_wavelet(config: Config):
    """Full pipeline: svox2 → DWT → TiledWaveletVolume → fine-level training."""
    device = torch.device(config.device)

    # --- Conversion ---
    tiled_model, n_dwt_sparse = convert_svox2_to_wavelet(config)
    tiled_model = tiled_model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Freeze base + DWT-initialized sparse levels ---
    tiled_model.approx.requires_grad_(False)
    for d in tiled_model.dense_details:
        d.requires_grad_(False)

    # Freeze the first n_dwt_sparse sparse levels (populated from DWT)
    frozen_sparse = n_dwt_sparse
    for i in range(frozen_sparse):
        for p in tiled_model.sparse_details[i].parameters():
            p.requires_grad_(False)

    frozen_params = (
        tiled_model.approx.numel()
        + sum(d.numel() for d in tiled_model.dense_details)
        + sum(
            p.numel() for sl in tiled_model.sparse_details[:frozen_sparse]
            for p in sl.parameters()
        )
    )
    print(f"\nFrozen: {frozen_params/1e6:.1f}M params "
          f"(base + {frozen_sparse} DWT sparse levels)")

    # --- Set up trainable fine levels with occupancy ---
    # Re-estimate occupancy from the base volume for trainable levels
    print("\nEstimating occupancy for trainable fine levels...")
    tiled_model.eval()

    # Use base volume density to estimate occupancy
    with torch.no_grad():
        base_vol = tiled_model.reconstruct_base()
        density = F.relu(base_vol[0, 0])  # (R_base, R_base, R_base)
        occupancy = density > config.occupancy_threshold
        occupancy = dilate_3d(occupancy.cpu(), config.occupancy_dilate).to(device)
        occ_pct = 100.0 * occupancy.float().mean().item()
        print(f"Base volume occupancy: {occ_pct:.1f}%")
        del base_vol, density

    # Update only the trainable (empty) sparse levels with occupancy
    for sparse_idx in range(frozen_sparse, len(tiled_model.sparse_details)):
        level_idx = config.base_level + 1 + sparse_idx
        D = tiled_model.level_sizes[level_idx + 1]
        tiles_per_axis = max(1, (D + config.tile_size - 1) // config.tile_size)
        tile_mask = occupancy_to_tile_mask(occupancy, tiles_per_axis)
        n_occ = tile_mask.sum().item()
        C = config.channels_per_level[level_idx] if config.channels_per_level else config.num_channels
        print(f"  Level {level_idx} ({D}³, {C}ch): "
              f"{n_occ}/{tile_mask.numel()} tiles → "
              f"{n_occ * 7 * C * config.tile_size**3 * 4 / 1e9:.1f} GB")

        new_sparse = SparseDetailLevel(
            full_size=D,
            num_channels=C,
            block_size=min(config.tile_size, D),
            occupancy_mask=tile_mask,
        ).to(device)
        tiled_model.sparse_details[sparse_idx] = new_sparse

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable_params = sum(
        p.numel() for p in tiled_model.parameters() if p.requires_grad
    )
    print(f"Trainable: {trainable_params/1e6:.1f}M params (fine levels)")

    # --- Load data ---
    print(f"\nLoading {config.scene}...")
    train_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "train",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )
    val_data = NerfSyntheticDataset(
        config.data_dir, config.scene, "val",
        resolution=config.train_resolution,
        white_bg=config.white_background,
        device=device,
    )

    out_dir = os.path.join(config.output_dir, config.scene)
    os.makedirs(out_dir, exist_ok=True)

    # --- Memory summary ---
    mem = tiled_model.memory_summary()
    print(f"\nModel: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")
    print(f"  Approx: {mem['approx_mb']:.0f} MB")
    print(f"  Dense details: {mem['dense_mb']:.0f} MB")
    print(f"  Sparse details: {mem['sparse_mb']:.0f} MB")
    for sl in mem['sparse_levels']:
        print(f"    Level {sl['level']}: {sl['occupied']}/{sl['total']} tiles, "
              f"{sl['memory_mb']:.0f} MB")

    # --- Training ---
    n_trainable_levels = len(tiled_model.sparse_details) - frozen_sparse
    fine_iters = config.iterations

    trainable = [p for p in tiled_model.parameters() if p.requires_grad]
    if not trainable:
        print("No trainable parameters — skipping training.")
        return

    optimizer = torch.optim.Adam(trainable, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fine_iters, eta_min=config.lr_min,
    )

    # Progressive activation for trainable fine levels
    fine_starts = [
        int(i * fine_iters / (n_trainable_levels + 1))
        for i in range(n_trainable_levels)
    ]
    print(f"\nFine level activation schedule: {fine_starts}")
    print(f"Training for {fine_iters} iterations...\n")

    best_psnr = 0.0
    t0 = time.time()

    for iteration in range(fine_iters):
        tiled_model.train()

        # Progressive fine level activation (only for trainable levels)
        for lvl_offset in range(n_trainable_levels):
            sparse_idx = frozen_sparse + lvl_offset
            active = iteration >= fine_starts[lvl_offset]
            for p in tiled_model.sparse_details[sparse_idx].parameters():
                p.requires_grad_(active)

        rays_o, rays_d, rgb_gt = train_data.get_random_rays(config.batch_rays)

        base_volume = tiled_model.reconstruct_base()

        result = render_rays(
            tiled_model, rays_o, rays_d, config,
            base_volume=base_volume,
            perturb=True,
        )

        l1_loss = F.l1_loss(result["rgb"], rgb_gt)
        l1_coarse = F.l1_loss(result["rgb_coarse"], rgb_gt)
        sparse_loss = result["density_fine"].mean()
        loss = l1_loss + 0.1 * l1_coarse + config.lambda_sparse * sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration + 1) % config.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{iteration + 1:6d}/{fine_iters}] "
                  f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"lr={lr:.2e} elapsed={elapsed:.0f}s")

        if (iteration + 1) % config.val_every == 0 or iteration == 0:
            tiled_model.eval()
            with torch.no_grad():
                result = render_image(
                    tiled_model, val_data.poses[0],
                    config.train_resolution, config.train_resolution,
                    val_data.focal, config,
                )

            val_psnr = psnr(result["rgb"], val_data.images[0])
            val_ssim = ssim(result["rgb"], val_data.images[0]).item()
            print(f"  → val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(img).save(
                os.path.join(out_dir, f"val_fine_{iteration + 1:06d}.png"),
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": tiled_model.state_dict(),
                    "config": config,
                    "psnr": val_psnr,
                }, os.path.join(out_dir, "best.pt"))
                print(f"  → saved best model (PSNR={best_psnr:.2f})")

        if (iteration + 1) % config.save_every == 0:
            torch.save({
                "iteration": iteration,
                "model_state_dict": tiled_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, os.path.join(out_dir, f"ckpt_fine_{iteration + 1:06d}.pt"))

    # Final save
    torch.save({
        "iteration": fine_iters,
        "model_state_dict": tiled_model.state_dict(),
        "config": config,
        "psnr": best_psnr,
    }, os.path.join(out_dir, "final.pt"))

    total_time = time.time() - t0
    print(f"\nFine-level training done. Best PSNR={best_psnr:.2f} dB. "
          f"Time={total_time / 60:.1f} min")
    mem = tiled_model.memory_summary()
    print(f"Model size: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="svox2 → Wavelet training")
    parser.add_argument("--svox2_ckpt", required=True,
                        help="Path to svox2 .npz checkpoint")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--data_dir", default="data/nerf_synthetic")
    parser.add_argument("--decomp_levels", type=int, default=7)
    parser.add_argument("--base_level", type=int, default=2)
    parser.add_argument("--channels_per_level", type=str,
                        default="28,28,28,28,28,4,4",
                        help="Comma-separated channel counts per level")
    parser.add_argument("--iterations", type=int, default=40000)
    parser.add_argument("--batch_rays", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument("--dwt_batch_channels", type=int, default=4,
                        help="Channels per DWT batch (memory vs speed)")
    parser.add_argument("--convert_only", action="store_true",
                        help="Only convert, don't train fine levels")
    args = parser.parse_args()

    config = Config(
        scene=args.scene,
        data_dir=args.data_dir,
        decomp_levels=args.decomp_levels,
        base_level=args.base_level,
        iterations=args.iterations,
        batch_rays=args.batch_rays,
        lr=args.lr,
        tile_size=args.tile_size,
        svox2_ckpt=args.svox2_ckpt,
        dwt_batch_channels=args.dwt_batch_channels,
        training_mode="svox2_to_wavelet",
    )
    config.channels_per_level = [int(x) for x in args.channels_per_level.split(",")]
    config.progressive_starts = []
    config.__post_init__()

    if args.convert_only:
        model, _ = convert_svox2_to_wavelet(config)
        out_dir = os.path.join(config.output_dir, config.scene)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, os.path.join(out_dir, "wavelet_converted.pt"))
        print(f"Saved to {out_dir}/wavelet_converted.pt")
    else:
        train_svox2_to_wavelet(config)
