"""Quick smoke test: TiledWaveletVolume forward + backward on B200."""
import torch
from tiled_wavelet_volume import TiledWaveletVolume, _get_level_sizes
from renderer import render_rays
from config import Config

device = torch.device("cuda")
config = Config(decomp_levels=5, base_level=2, tile_size=64)
print(f"Config: {config.base_resolution}^3 -> {config.target_resolution}^3")
level_sizes = _get_level_sizes(config.target_resolution, config.decomp_levels, config.wavelet)
print(f"Level sizes: {level_sizes}")

# Create sparse occupancy masks (~10% occupied) for realistic memory usage
occupancy_masks = {}
for i in range(config.base_level + 1, config.decomp_levels):
    D = level_sizes[i + 1]
    block = min(config.tile_size, D)
    tiles_per = max(1, D // block)
    # Only center tile occupied
    occ = torch.zeros(tiles_per, tiles_per, tiles_per, dtype=torch.bool)
    center = tiles_per // 2
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                cx, cy, cz = center+dx, center+dy, center+dz
                if 0 <= cx < tiles_per and 0 <= cy < tiles_per and 0 <= cz < tiles_per:
                    occ[cx, cy, cz] = True
    occupancy_masks[i] = occ
    pct = 100 * occ.float().mean().item()
    print(f"  Level {i}: {tiles_per}^3 tiles, {pct:.0f}% occupied")

model = TiledWaveletVolume(
    base_resolution=config.base_resolution,
    decomp_levels=config.decomp_levels,
    num_channels=config.num_channels,
    wavelet=config.wavelet,
    scene_bound=config.scene_bound,
    tile_size=config.tile_size,
    base_level=config.base_level,
    occupancy_masks=occupancy_masks,
).to(device)

mem = model.memory_summary()
print(f"\nModel: {mem['total_params_M']:.0f}M params, {mem['total_mb']:.0f} MB")
print(f"  Approx: {mem['approx_mb']:.0f} MB, Dense: {mem['dense_mb']:.0f} MB, Sparse: {mem['sparse_mb']:.0f} MB")

# Forward pass
base_vol = model.reconstruct_base()
print(f"\nBase volume: {base_vol.shape}")

N = 512
rays_o = torch.randn(N, 3, device=device) * 0.5
rays_d = torch.randn(N, 3, device=device)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

result = render_rays(model, rays_o, rays_d, config, base_volume=base_vol, perturb=True)
print(f"Forward OK: rgb={result['rgb'].shape}")

# Backward
loss = result['rgb'].mean() + 0.1 * result['rgb_coarse'].mean()
loss.backward()

has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
total = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Backward OK: {has_grad}/{total} params have gradients")

mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"\nGPU: {mem_alloc:.1f} / {mem_total:.1f} GB")
print("\nSMOKE TEST PASSED")
