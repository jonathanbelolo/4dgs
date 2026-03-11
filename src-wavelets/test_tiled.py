import torch, ptwt
from tiled_idwt import tiled_waverec3
from sparse_coefficients import SparseDetailLevel
from tiled_wavelet_volume import _get_level_sizes
from wavelet_volume import SUBBAND_KEYS

torch.manual_seed(42)
wavelet = "bior4.4"
R = 256; decomp_levels = 3; C = 4; base_level = 0; tile_size = 64

level_sizes = _get_level_sizes(R, decomp_levels, wavelet)
approx = torch.randn(1, C, level_sizes[0], level_sizes[0], level_sizes[0])
details = []
for i in range(decomp_levels):
    D = level_sizes[i+1]
    details.append(torch.randn(7, C, D, D, D))

volume = approx
for i in range(decomp_levels):
    d = {}
    for j, key in enumerate(SUBBAND_KEYS):
        d[key] = details[i][j].unsqueeze(0)
    volume = ptwt.waverec3((volume, d), wavelet=wavelet)

base = approx
for i in range(base_level + 1):
    d = {}
    for j, key in enumerate(SUBBAND_KEYS):
        d[key] = details[i][j].unsqueeze(0)
    base = ptwt.waverec3((base, d), wavelet=wavelet)

fine_details = []
for i in range(base_level + 1, decomp_levels):
    D = level_sizes[i+1]
    block = min(tile_size, D)
    tiles_per = max(1, D // block)
    occ = torch.ones(tiles_per, tiles_per, tiles_per, dtype=torch.bool)
    sparse = SparseDetailLevel(D, C, block, occ)
    with torch.no_grad():
        for idx in range(sparse.n_occupied):
            pos = sparse.occupied_positions[idx]
            tx, ty, tz = pos[0].item(), pos[1].item(), pos[2].item()
            sx = slice(tx*block, min((tx+1)*block, D))
            sy = slice(ty*block, min((ty+1)*block, D))
            sz = slice(tz*block, min((tz+1)*block, D))
            actual = details[i][:, :, sx, sy, sz]
            sparse.tiles[idx].data[:, :, :actual.shape[2], :actual.shape[3], :actual.shape[4]].copy_(actual)
    fine_details.append(sparse)

# Boundary tile (3,0,0): x=[192,256)
ts = (192, 0, 0)
ref = volume[:, :, 192:256, 0:64, 0:64]
tile = tiled_waverec3(base, fine_details, ts, tile_size, wavelet, base_level+1, level_sizes[1:])
tile_cropped = tile[:, :, :64, :64, :64]
err = (tile_cropped - ref).abs()
print(f"Boundary tile: tiled shape={tile.shape}, max err={err.max().item():.4f}")
for x in range(0, 64, 8):
    e = err[0, 0, x, :, :].max().item()
    print(f"  x={x:2d}: max_err={e:.6f}")

# Interior tile (1,1,1)
ts2 = (64, 64, 64)
ref2 = volume[:, :, 64:128, 64:128, 64:128]
tile2 = tiled_waverec3(base, fine_details, ts2, tile_size, wavelet, base_level+1, level_sizes[1:])
tile2_cropped = tile2[:, :, :64, :64, :64]
print(f"\nInterior tile: tiled shape={tile2.shape}, max err={(tile2_cropped - ref2).abs().max().item():.6e}")

# Check: what about tile (3,0,0) with extra output -- is the error at the end?
print(f"\nTile output vs ref shapes: tile={tile.shape}, ref={ref.shape}")
if tile.shape[2] > 64:
    print("Tile is LARGER than 64 -- excess was trimmed")
