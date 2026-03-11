# Wavelet Volume Rendering Pipeline

Sparse wavelet voxel grids for high-resolution neural radiance field reconstruction.

## Overview

```
NeRF Synthetic / N3V dataset
    |
    v
[1] data.py                -- Load images + cameras, generate ray batches
    |
    +---> NerfSyntheticDataset (poses, images, focal)
    |
    v
[2] train_frequency_matched.py  -- Multi-stage coarse-to-fine training (svox2 CUDA)
    |   Stages: 136³ -> 264³ -> 520³ -> 520³ (-> 1032³)
    |   svox2 RMSProp optimizer (~40x faster than PyTorch Adam)
    |   Wavelet or trilinear upsampling between stages
    |
    +---> output/{scene}/{mode}/stage{N}.npz     (svox2 native checkpoints)
    +---> output/{scene}/{mode}/stage{N}_analysis.pt  (dense grids + residuals)
    |
    v
[3] wavelet_analysis.py    -- Forward DWT, coefficient statistics, method comparison
    |
    +---> analysis/ plots + metrics
    |
    v
[4] viewer_svox2.py / viser_viewer.py  -- Interactive real-time 3D viewer (browser)
```

### Alternate paths

```
[A] train.py               -- Pure wavelet training (progressive IDWT, no svox2)
    |   WaveletVolume / TiledWaveletVolume
    |   Differentiable IDWT -> trilinear query -> volume rendering
    |
[B] train_svox2.py         -- Convert pretrained svox2 checkpoint to TiledWaveletVolume
    |   load_svox2.py: sparse -> dense -> forward DWT -> wavelet coefficients
    |
[C] direct_grid_volume.py  -- Dense grid (Plenoxels-style), convertible to wavelets via DWT
```

## Scripts

### Stage 1: Data Loading

**`data.py`**
Loads NeRF Synthetic (Blender) or N3V-converted datasets.

- Parses `transforms_{split}.json` (camera intrinsics + per-image c2w extrinsics)
- Resizes images to target resolution, alpha-composites onto white background
- `get_random_rays(batch_size)`: stratified sampling across all training images
- OpenGL camera convention (-Z forward, Y up)

```bash
# Dataset structure:
data/nerf_synthetic/lego/
  transforms_train.json    # 100 views
  transforms_val.json
  transforms_test.json     # 200 views
  train/r_000.png          # 800x800 RGBA
  ...
```

**`prepare_n3v.py`** (planned)
Converts N3V (Neural 3D Video) LLFF format to NeRF Synthetic format.

- Reads `poses_bounds.npy` (N x 17: 3x5 pose [DRB convention] + near/far)
- Extracts frame 0 from 18 synchronized MP4 videos via ffmpeg
- Converts LLFF DRB -> OpenGL c2w, centers and scales scene
- Outputs `transforms_{train,val,test}.json` + images

```bash
python prepare_n3v.py \
  --data_dir data/n3v/coffee_martini \
  --output_dir data/nerf_synthetic/coffee_martini \
  --frame 0
```

### Stage 2: Training

**`train_frequency_matched.py`** (primary)
Multi-stage coarse-to-fine training using svox2's CUDA kernels.

Core insight: match image resolution to volume resolution so the optimizer never
sees frequencies it can't represent. Pixel footprint = 3.0/N, image footprint =
2.16/R, match when R ~ 0.72 * N.

**Four training modes:**

| Mode | Description | Stages |
|------|-------------|--------|
| FM | Frequency-matched: image res scales with volume | (100px, 136³) -> (200px, 264³) -> (400px, 520³) -> (800px, 520³) |
| SP | Standard progressive: full-res images throughout | (800px, 136³) -> (800px, 264³) -> (800px, 520³) -> (800px, 520³) |
| SS | Single-scale: one resolution from scratch | (800px, 520³) |
| Hybrid | Full-res images + wavelet upsampling | (800px, 136³) -> (800px, 264³) -> (800px, 520³) -> (800px, 520³) |

**Per-stage pipeline:**
1. Create svox2 SparseGrid at target resolution (or load previous stage)
2. If resuming: wavelet upsample (IDWT with zero detail) or trilinear interpolate
3. Train with svox2 fused forward+backward CUDA kernel + RMSProp
4. Early stopping on validation PSNR (patience=20 evals, min 22.5K iters)
5. Periodic sparsification: remove voxels with density < 5.0
6. Save svox2 .npz + dense analysis .pt checkpoint

**Wavelet upsampling:** Grid IS the DWT approximation coefficients. IDWT with zero
detail produces the smoothest possible upsample (pure linear operator, no interpolation
artifacts). Natural bior4.4 sizes: 40 -> 72 -> 136 -> 264 -> 520 -> 1032.

**Residual training:** Stages > 0 freeze the upsampled base and train only the detail
(residual). TV regularization applied to residual only, encouraging sparse refinements.

```bash
# Stage 0: from scratch
python train_frequency_matched.py --mode fm --stage 0 --scene lego

# Stage 1: resume from stage 0
python train_frequency_matched.py --mode fm --stage 1 --resume output/fm_poc/lego/fm/stage0.npz

# Stage 2
python train_frequency_matched.py --mode fm --stage 2 --resume output/fm_poc/lego/fm/stage1.npz

# Stage 3 (super-resolution refinement)
python train_frequency_matched.py --mode fm --stage 3 --resume output/fm_poc/lego/fm/stage2.npz
```

| Key parameter | Value | Notes |
|---|---|---|
| lr_sigma (stage 0) | 30.0 -> 0.05 | Exponential decay, 7.5K warm-up |
| lr_sh (stage 0) | 0.01 -> 5e-4 | |
| lr_sigma (stage > 0) | 0.5 -> 0.005 | Low LR for residual refinement |
| lr_sh (stage > 0) | 0.001 -> 0.0001 | |
| batch_rays | 4096 | Random rays per iteration |
| TV density | 1e-5 * scale | Scaled by param/data ratio |
| TV SH | 1e-3 * scale | |
| Sparsify every | 5K iters | sigma_thresh=5.0, dilate=1 |
| val_every | 5000 | Average over 10 views |
| Early stopping | 20 evals | ~100K iters without improvement |

**`train.py`** (alternate)
Pure wavelet training without svox2. Uses WaveletVolume or TiledWaveletVolume with
PyTorch Adam optimizer. Progressive level activation (coarse-to-fine).

- Differentiable IDWT reconstruction -> trilinear grid_sample -> volume rendering
- Coarse+fine hierarchical ray marching (64+64 samples per ray)
- Loss: L1(rgb) + L1(rgb_coarse) + sparsity + TV

```bash
python train.py --mode wavelet --decomp_levels 3 --scene lego
```

### Stage 3: Analysis

**`wavelet_analysis.py`**
Post-training wavelet coefficient analysis.

- Forward DWT (ptwt.wavedec3) on trained grids
- Per-level statistics: energy, sparsity, shape
- Comparative analysis across FM/SP/SS/Hybrid modes
- Rate-distortion via coefficient pruning

```bash
python run_poc.py --scene lego --analysis_only
```

**`compress.py`**
Rate-distortion curves via magnitude-based pruning.

- For each keep_ratio: prune coefficients, render test views, compute PSNR
- LoD visualization at each wavelet level
- Plot PSNR vs model size

### Stage 4: Evaluation

**`eval.py`**
Render test views and compute metrics.

- Dispatches to correct model class (WaveletVolume / TiledWaveletVolume)
- PSNR + SSIM per view, average metrics
- Saves rendered images

**`metrics.py`**
- `psnr(pred, target)`: MSE-based PSNR in dB
- `ssim(pred, target)`: Structural similarity with Gaussian windowing

### Stage 5: Viewing

**`viewer_svox2.py`**
Interactive browser viewer for svox2 checkpoints (viser + svox2 CUDA rendering).

- Camera convention: OpenGL -> OpenCV transform (diag(1,-1,-1,1))
- Scene scale factor: 2/3 (matching svox2's nerf_dataset loader)
- FOV slider, resolution slider, real-time FPS display
- Remote: `ssh -L 7080:localhost:7080 runpod` then http://localhost:7080

```bash
python viewer_svox2.py --ckpt output/fm_poc/lego/fm/stage3.npz --port 7080
```

**`viser_viewer.py`**
Alternate viewer with Viser-to-NeRF coordinate transform.

- VISER_TO_NERF rotation: Viser (+Z up) -> NeRF synthetic (+Y up)
- Orbit controls, FOV 15-90 deg (default 35 ~ 50mm equiv), resolution 128-800px

```bash
python viser_viewer.py output/fm_poc/lego/fm/stage3.npz --port 8080
```

**`render_video.py`**
Orbit video rendering (360 deg, 120 frames, MP4 via ffmpeg).

## Volume Representations

### WaveletVolume (`wavelet_volume.py`)

Multi-resolution wavelet coefficient volume (32³ -> 512³).

```
Learnable coefficients: approx (32³) + detail levels (7 subbands each)
    -> IDWT (ptwt.waverec3) -> dense (1, 28, 256³) feature volume
    -> trilinear grid_sample at query points
    -> decode: density (ReLU) + RGB (SH evaluation)
```

- `reconstruct(max_level)`: Incremental IDWT with level cap for LoD
- `reconstruct_pair(fine, coarse)`: Efficient dual reconstruction (single IDWT pass)
- `prune(keep_ratio)`: Magnitude-based coefficient pruning
- Wavelet: bior4.4 (CDF 9/7, JPEG 2000 standard)

### TiledWaveletVolume (`tiled_wavelet_volume.py`)

High-resolution (1024³+) with tiled IDWT + sparse coefficient storage.

```
Dense base levels (0..2): full IDWT -> base volume (fits in RAM)
Sparse fine levels (3+): tiled IDWT on-demand per query batch
    -> SparseDetailLevel: only occupied tiles have parameters
    -> occupancy mask from coarse density estimation
```

- `query_tiled(xyz, base_volume)`: On-demand tile reconstruction
- `tiled_idwt.py`: Backward trace through levels, halo padding, cascading IDWT
- `sparse_coefficients.py`: SparseDetailLevel with spatial hash lookup
- `occupancy.py`: Density thresholding + dilation -> tile-level mask

### DirectGridVolume (`direct_grid_volume.py`)

Dense voxel grid with direct parameter optimization (Plenoxels-style).

- Separate `density_grid` (1,1,R,R,R) and `sh_grid` (1,27,R,R,R) parameters
- `upsample(new_res)`: Trilinear interpolation to higher resolution
- `prune(max_weights, threshold)`: Weight-based voxel pruning
- `tv_loss()`, `tv_loss_sh()`: L1 total variation regularization
- **ResidualGridVolume**: Frozen base + trainable detail (residual architecture)

### svox2 SparseGrid (external)

CUDA-optimized sparse voxel grid from Plenoxels.

- Sparse storage: `links` (R,R,R int32) + `density_data` (N,1) + `sh_data` (N,27)
- Fused CUDA kernels: volume_render_image, volume_render_fused (forward+backward+loss)
- Custom RMSProp optimizer: sparse updates, ~40x faster than PyTorch Adam
- In-place TV gradient computation
- `load_svox2.py`: Load sparse data, expand channels to dense, extract occupancy

## Rendering Pipeline (`renderer.py`)

Differentiable ray marching through volumes.

1. **Coarse pass**: Stratified sampling (64 samples) on coarse volume
2. **Importance sampling**: Concentrate fine samples where coarse weights are high
3. **Fine pass**: Render fine volume with combined (coarse+fine) samples
4. **Compositing**: alpha = 1 - exp(-sigma * delta_t), transmittance = cumprod(1-alpha)

Color decoding (`sh.py`):
- 28 channels: 1 density + 27 SH coefficients (degree 2, 9 basis x 3 colors)
- `eval_sh_bases(dirs)`: Real SH basis evaluation (degrees 0,1,2)
- `eval_sh_color(coeffs, dirs)`: Dot product + sigmoid -> RGB in [0,1]

## svox2-4d Fork (lost, to be re-implemented)

Custom svox2 fork with 5 CUDA/Python modifications. Detailed spec in
`/Users/jonathanbelolo/.claude/plans/recursive-sparking-marble.md`.

| Mod | Description | Complexity | Status |
|-----|-------------|------------|--------|
| 1 | Higher SH degree (basis_dim 10 -> 25) | HIGH | Lost, needs reimplementation |
| 2 | RMS optimizer state preservation across resample | LOW | Lost, needs reimplementation |
| 3 | Sparse-to-sparse trilinear upsample (OOM fix) | MED | Lost, needs reimplementation |
| 5 | Per-voxel hit count + adaptive LR | MED | Lost, needs reimplementation |
| 6 | Fused anchor gradient CUDA kernel | LOW | Lost, needs reimplementation |

## Training Results

### NeRF Synthetic Lego

| Configuration | Test PSNR | Views | Resolution | Notes |
|---|---|---|---|---|
| Plenoxels (published) | 34.10 dB | 200 | 800px | Reference baseline |
| FM 5-stage (136->1032³) | 33.80 dB | 200 | 800px | Best result before RunPod wipe |
| FM 4-stage (136->520³) | ~33.0 dB | 200 | 800px | Without 1032³ stage |
| Wavelet upsample 520->1032 | 17.02 dB | - | 800px | Zero-detail upsample only (no fine training) |

### N3V Coffee Martini (planned)

- 18 synchronized cameras, 2704x2028 MP4 videos, 300 frames @ 30fps
- Frame 0 extraction -> NeRF Synthetic format -> 3D reconstruction baseline
- Future: 4D temporal extension over full 300-frame sequence

## File Layout

```
src-wavelets/
  # Core
  config.py                  # Hyperparameters (Config dataclass)
  data.py                    # NerfSyntheticDataset loader
  sh.py                      # Spherical harmonics (degree 2)
  rays.py                    # Stratified + importance sampling
  renderer.py                # Differentiable volume rendering
  metrics.py                 # PSNR, SSIM

  # Volume representations
  wavelet_volume.py          # WaveletVolume (dense DWT, <= 512³)
  tiled_wavelet_volume.py    # TiledWaveletVolume (sparse tiled, 1024³+)
  sparse_coefficients.py     # SparseDetailLevel (per-tile storage)
  tiled_idwt.py              # Memory-efficient tiled IDWT
  occupancy.py               # Occupancy estimation + dilation
  direct_grid_volume.py      # DirectGridVolume + ResidualGridVolume
  load_svox2.py              # svox2 sparse -> dense channel expansion

  # Training
  train.py                   # Pure wavelet training (progressive IDWT)
  train_frequency_matched.py # Multi-stage svox2 training (primary)
  train_svox2.py             # Convert svox2 ckpt -> TiledWaveletVolume
  run_poc.py                 # Run all FM/SP/SS/Hybrid + analysis

  # Evaluation + analysis
  eval.py                    # Test set rendering + metrics
  compress.py                # Rate-distortion via pruning
  wavelet_analysis.py        # Coefficient statistics + comparison

  # Visualization
  viewer.py                  # Generic viser viewer
  viewer_svox2.py            # svox2 checkpoint viewer
  viser_viewer.py            # Alternate viewer (VISER->NeRF coords)
  render_video.py            # 360° orbit video rendering
  verify_conversion.py       # svox2 <-> wavelet conversion check
```

## Dependencies

- **Core**: torch >= 2.0, numpy, Pillow
- **Wavelets**: ptwt >= 0.1.9 (PyTorch Wavelet Transform), pywt (filter metadata)
- **Training**: svox2 (CUDA sparse voxel grids, custom RMSProp optimizer)
- **Viewing**: viser (interactive browser viewer)
- **Analysis**: matplotlib (plots)
- **Optional**: lpips (perceptual loss), tqdm (progress bars)

## Infrastructure

- **RunPod**: H100 SXM 80GB (fine_chocolate_chameleon), SSH alias `ssh runpod`
- **Workspace**: `/workspace/src-wavelets/` (uploaded via scp from local)
- **svox2**: `/workspace/svox2/` (vanilla clone, `pip install -e .`)
- **Data**: `/workspace/data/nerf_synthetic/`, `/workspace/data/n3v/`
- **Local code**: `/Users/jonathanbelolo/dev/claude/code/4D/src-wavelets/`

## Key Lessons

1. **Natural wavelet sizes matter**: bior4.4 DWT of 264 -> approx=136, not 128. Power-of-2 grids don't nest. Use the natural sequence: 40, 72, 136, 264, 520, 1032.

2. **Frequency matching prevents aliasing**: Training a coarse grid against high-res images bakes compromise artifacts into the low-frequency structure. Match image bandwidth to volume bandwidth at each stage.

3. **svox2's CUDA optimizer is essential**: ~40x faster than PyTorch Adam for sparse voxel grids. Fused forward+backward+loss in one kernel, sparse RMSProp on touched voxels only.

4. **Residual training preserves coarse quality**: Freeze the upsampled base, train only the detail. Low LR (0.5 vs 30.0) prevents the residual from overwriting the base signal.

5. **Wavelet upsample > trilinear**: IDWT with zero detail is a pure linear operator with no interpolation artifacts. The grid IS the approximation coefficients.

6. **Back up to git**: RunPod volumes can be wiped without warning. All custom code (especially CUDA modifications) must be in a git repo with a remote.
