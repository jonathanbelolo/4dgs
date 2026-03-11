# Frequency-Matched Wavelet Reconstruction: NeRF Synthetic Lego Proof of Concept

## Implementation Plan

---

## 1. Objective

Demonstrate that **frequency-matched coarse-to-fine training** produces superior reconstructions compared to standard single-resolution training. The core claim from `frequency_matched_wavelet_reconstruction.md`:

> When a 64³ volume is trained against 512px images, the optimizer finds the best explanation of the low-frequency structure. There is no high-frequency signal to distort its solution. [...] This is fundamentally different from training a 64³ volume against 4K images, where the volume makes compromises to approximately represent detail it cannot capture.

We validate this on the NeRF Synthetic Lego scene by comparing three training strategies that all produce a 512³ volume, then analyzing the quality and wavelet compressibility of each.

---

## 2. Dataset: NeRF Synthetic Lego

| Parameter | Value |
|-----------|-------|
| Training views | 100 |
| Test views | 200 |
| Native resolution | 800 × 800 px |
| Camera FOV | 39.6° (camera_angle_x = 0.6911 rad) |
| Focal length (at 800px) | 1111.11 px |
| Camera distance | ~4.0 units from origin |
| Scene bound | 1.5 (cube from -1.5 to 1.5, 3 units across) |
| Object extent | ~1.0 unit radius (Lego bulldozer) |
| Background | White (alpha composited) |
| Color depth | 8-bit RGBA |
| Location | `data/nerf_synthetic/lego/` |

---

## 3. Frequency Analysis for Lego

### 3.1 Pixel Footprint Derivation

The focal length in pixels scales linearly with image resolution:

$$f_{\text{px}} = \frac{R}{800} \times 1111.11$$

At camera distance $d \approx 4.0$ units, the pixel footprint on the object surface (at distance $d_{\min} \approx 3.0$ from camera to nearest surface):

$$\Delta = \frac{d_{\min}}{f_{\text{px}}} = \frac{3.0 \times 800}{R \times 1111.11} = \frac{2.16}{R}$$

The matched volume resolution for a 3-unit scene cube:

$$N = \frac{3.0}{\Delta} = \frac{3.0 \times R}{2.16} = 1.389 \times R$$

### 3.2 Resolution Matching Table

| Image resolution $R$ | Focal length $f$ | Pixel footprint $\Delta$ (units) | Matched volume $N$ | Practical $N$ |
|----------------------|-----------------|--------------------------------|-------------------|--------------|
| 100 | 138.9 | 0.0216 | 139 | **128³** |
| 200 | 277.8 | 0.0108 | 278 | **256³** |
| 400 | 555.6 | 0.0054 | 556 | **512³** |
| 800 | 1111.1 | 0.0027 | 1111 | **1024³** |

**Key observation:** The native 800×800 images match a volume of ~1024³, not 512³. This means a 512³ volume at full resolution is already operating below the Nyquist limit of the images — the images contain spatial frequencies the volume cannot represent. Training 512³ against 800px forces the optimizer to approximate information it cannot capture.

### 3.3 Implications for the PoC

- Stages 0–2 (100→200→400 px against 128→256→512³) are **frequency-matched**: the image bandwidth equals the volume bandwidth at each stage.
- Stage 3 (800px against 512³) is **over-resolved**: the images have 2× more spatial frequency content than the volume can hold. The optimizer must make compromises.
- This is exactly the scenario the frequency-matching principle is designed to improve: by training coarse levels against band-limited images first, we avoid baking high-frequency compromise artifacts into the coarse structure.

---

## 4. Training Pipeline

### 4.1 Stage Overview

| Stage | Image res | Volume res | Iterations | Learning rate | Image supervision |
|-------|-----------|-----------|------------|--------------|-------------------|
| 0 | 100 × 100 | 128³ | 10,000 | 0.02 → 0.002 | Lanczos ↓8× |
| 1 | 200 × 200 | 256³ | 15,000 | 0.01 → 0.001 | Lanczos ↓4× |
| 2 | 400 × 400 | 512³ | 25,000 | 0.005 → 0.0005 | Lanczos ↓2× |
| 3 (optional) | 800 × 800 | 512³ | 30,000 | 0.002 → 0.0002 | Native |
| **Total** | | | **80,000** | | |

**Stage 3 is a super-resolution refinement stage.** The volume resolution does not increase (stays at 512³), but the supervision images double in resolution. The optimizer can't add new spatial frequencies to the volume, but the higher-resolution images provide a cleaner, less aliased target signal. The frequency-matching principle says this stage should produce marginal improvement compared to stages 0–2, which is itself a testable prediction.

### 4.2 Anti-Aliased Image Downsampling

**Critical requirement.** The images at each stage must be properly low-pass filtered before downsampling. Naive subsampling (skipping pixels) leaks high-frequency energy into the low-resolution images and defeats the purpose of frequency matching.

The existing `NerfSyntheticDataset` in `data.py` already uses `PIL.Image.LANCZOS` for resizing, which is a high-quality sinc-approximating filter. No changes needed — we simply pass the `resolution` parameter to the dataset constructor at each stage:

```python
dataset = NerfSyntheticDataset(
    data_dir, "lego", "train",
    resolution=image_res,  # 100, 200, 400, or 800
    white_bg=True,
    device=device,
)
```

### 4.3 Volume Initialization

**Stage 0 (128³):** Initialize from scratch.
- Density: near-zero (init with small random values or constant -1 so ReLU gives 0)
- SH coefficients: zero

**Stages 1–3 (256³, 512³):** Initialize from the converged previous stage via **wavelet upsampling**.

### 4.4 Wavelet Upsampling Between Stages

The transition from volume resolution $N$ to $2N$ uses wavelet synthesis:

1. Take the converged $N^3 \times C$ grid as an approximation signal
2. Compute the required approximation size for the target resolution using `ptwt`
3. If sizes don't exactly match (due to wavelet filter boundary effects), resize via trilinear interpolation to match
4. Create zero-valued detail coefficients at the matching size
5. Apply inverse DWT (`ptwt.waverec3`) → produces the $(2N)^3$ upsampled grid

```python
def wavelet_upsample(grid, target_size, wavelet="bior4.4"):
    """Upsample grid from N³ to ~2N³ via wavelet synthesis.

    The grid is treated as the low-frequency approximation.
    New high-frequency detail coefficients are initialized to zero.
    This produces a smooth upsampling — equivalent to ideal low-pass
    interpolation in the wavelet basis.

    Args:
        grid: (1, C, N, N, N) dense volume
        target_size: desired output size (approximately 2N)
        wavelet: wavelet family for synthesis

    Returns:
        (1, C, target_size, target_size, target_size) upsampled volume
    """
    import ptwt

    # Forward DWT of a dummy target-sized volume to determine exact coefficient sizes
    dummy = torch.zeros(1, 1, target_size, target_size, target_size)
    dummy_coeffs = ptwt.wavedec3(dummy, wavelet, level=1)
    approx_size = dummy_coeffs[0].shape[-1]  # size of approximation at this level

    # Resize grid to match the required approximation size
    approx = F.interpolate(grid, size=approx_size, mode="trilinear", align_corners=True)

    # Zero detail coefficients at the correct size
    detail_size = list(dummy_coeffs[1].values())[0].shape[-1]
    C = grid.shape[1]
    zero_details = {
        key: torch.zeros(1, C, detail_size, detail_size, detail_size, device=grid.device)
        for key in ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]
    }

    # Inverse DWT: smooth approximation + zero details → upsampled grid
    upsampled = ptwt.waverec3([approx, zero_details], wavelet)

    # Crop/pad to exact target size if needed
    upsampled = upsampled[:, :, :target_size, :target_size, :target_size]

    return upsampled
```

**Why wavelet upsampling instead of trilinear?** Wavelet synthesis uses the matched reconstruction filter, producing an output that is exactly band-limited to the approximation's frequency content. Trilinear interpolation has a triangular frequency response that attenuates high frequencies within the passband and leaks some energy beyond it. For the frequency-matching principle, wavelet synthesis gives a cleaner separation.

In practice for the PoC, the difference is likely small. Both approaches should be tested.

### 4.5 Per-Stage Training Details

Each stage trains a dense `DirectGridVolume` (or equivalent `nn.Parameter` grid) using volume rendering with hierarchical sampling:

**Optimizer:** Adam
- Separate parameter groups for density (channel 0) and SH coefficients (channels 1–27)
- Density learning rate: 2× the SH learning rate
- Learning rate schedule: cosine decay from `lr_start` to `lr_end` over the stage's iterations

**Loss function:**
$$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}$$

- $\mathcal{L}_{\text{color}}$: L1 photometric loss (more robust to outliers than L2)
- $\mathcal{L}_{\text{TV}}$: Total variation on the density field (smoothness prior)
  - Decay schedule: $\lambda_{\text{TV}}$ starts at $10^{-4}$ and decays to $10^{-6}$ over the stage
  - TV is essential at coarse stages to prevent noise in empty regions
- $\mathcal{L}_{\text{sparse}}$: L1 penalty on raw density values (encourages empty space to stay empty)
  - Weight: $10^{-5}$, constant

**Ray sampling:**
- Batch size: 4,096 rays per iteration
- Coarse samples: 64 stratified samples per ray in $[t_{\text{near}}, t_{\text{far}}]$
- Fine samples: 64 importance-sampled from coarse weights
- Near/far: 2.0 / 6.0 (standard for NeRF Synthetic)
- Perturbation: on during training, off during eval

**Density activation:** ReLU (matching svox2 convention)

**Voxel pruning (stages 1+):**
- Every 5,000 iterations, zero out voxels with $\text{ReLU}(\sigma) < \epsilon$
- This keeps the effective parameter count manageable

**Validation:**
- Every 5,000 iterations: render 5 test views at the stage's image resolution
- Log PSNR and SSIM
- Also render at 800px for cross-resolution comparison

---

## 5. Baseline Comparisons

Three configurations, all producing a final 512³ volume with 80,000 total iterations:

### Baseline A: Single-Scale (SS)

Train a 512³ grid from scratch against 800×800 images for 80,000 iterations. This is the standard Plenoxels approach — no multi-resolution, no frequency matching.

| Stage | Image | Volume | Iterations |
|-------|-------|--------|------------|
| 0 | 800 | 512³ | 80,000 |

### Baseline B: Standard Progressive (SP)

Progressive resolution schedule (128→256→512) but always using full 800×800 images. This is the approach used by `train_direct()` in the current codebase — it benefits from coarse-to-fine initialization but suffers from the coarse commitment problem because coarse volumes see high-frequency image content they cannot represent.

| Stage | Image | Volume | Iterations |
|-------|-------|--------|------------|
| 0 | 800 | 128³ | 10,000 |
| 1 | 800 | 256³ | 15,000 |
| 2 | 800 | 512³ | 25,000 |
| 3 | 800 | 512³ | 30,000 |

Upsampling between stages: trilinear interpolation (current behavior).

### Proposed: Frequency-Matched (FM)

The pipeline from Section 4:

| Stage | Image | Volume | Iterations |
|-------|-------|--------|------------|
| 0 | 100 | 128³ | 10,000 |
| 1 | 200 | 256³ | 15,000 |
| 2 | 400 | 512³ | 25,000 |
| 3 | 800 | 512³ | 30,000 |

Upsampling between stages: wavelet synthesis (Section 4.4).

### Hybrid Variant: FM + Wavelet Upsample applied to SP

To isolate the effect of frequency matching from the effect of wavelet upsampling, also test:

| Stage | Image | Volume | Iterations | Upsampling |
|-------|-------|--------|------------|------------|
| 0 | 800 | 128³ | 10,000 | — |
| 1 | 800 | 256³ | 15,000 | Wavelet |
| 2 | 800 | 512³ | 25,000 | Wavelet |
| 3 | 800 | 512³ | 30,000 | — |

This tests whether wavelet upsampling alone (without frequency matching) provides a benefit over trilinear upsampling.

---

## 6. Evaluation

### 6.1 Quality Metrics

After each stage and at the end of training:

1. **PSNR** on the 200 test views rendered at 800×800 (native resolution)
2. **SSIM** on the same views
3. **LPIPS** (perceptual quality, using AlexNet backbone)
4. **Per-stage convergence curves**: PSNR vs. iteration at each stage

### 6.2 Wavelet Compression Analysis

After training, apply a full multi-level DWT to the final 512³ grid:

```python
coeffs = ptwt.wavedec3(grid, wavelet="bior4.4", level=3)
# coeffs[0]: approx at ~72³
# coeffs[1]: detail_0 at ~72³ (7 subbands)
# coeffs[2]: detail_1 at ~136³
# coeffs[3]: detail_2 at ~264³
```

Analyze the wavelet coefficient distributions:

1. **Coefficient magnitude histograms** at each level, for each method (FM, SP, SS)
2. **Energy concentration**: what fraction of total energy is in the top-$k$ coefficients?
3. **Rate-distortion curves**: threshold coefficients by magnitude, reconstruct via IDWT, render test views, measure PSNR
   - Plot: number of retained coefficients vs. PSNR
   - The frequency-matched result should have a **higher** (better) curve — same quality with fewer coefficients

### 6.3 Coarse-Level Quality Analysis

To directly test the coarse commitment hypothesis, compare the quality of the coarse reconstruction (approximation coefficients only, no detail) between methods:

```python
# Reconstruct from approximation only (zero all details)
coarse_grid = ptwt.waverec3([coeffs[0], zeros, zeros, zeros], wavelet)
# Render test views from coarse_grid
coarse_psnr = evaluate(coarse_grid, test_data)
```

**Prediction:** The FM approach produces a coarse reconstruction that is closer to optimal for its resolution band — it should have higher PSNR when rendered at a low resolution (e.g., 100px test images). The SP/SS approaches produce coarse reconstructions contaminated by high-frequency compromise artifacts.

### 6.4 Qualitative Comparisons

Render side-by-side comparison images:

1. **Full resolution**: FM vs. SP vs. SS at 800px — differences should be subtle
2. **Coarse reconstruction only**: FM vs. SP vs. SS approximation-only renders — differences should be dramatic
3. **Error maps**: absolute per-pixel error heatmaps highlighting where each method fails
4. **Detail crops**: zoom into fine features (Lego studs, text, thin elements) where the coarse commitment problem manifests

---

## 7. Implementation

### 7.1 What Already Exists

| Component | File | Status |
|-----------|------|--------|
| Dataset loading with arbitrary resolution | `data.py` | Ready (Lanczos resize) |
| Direct grid volume training | `train.py` (`train_direct()`) | Needs modification |
| Volume rendering (hierarchical sampling) | `renderer.py` | Ready |
| SH evaluation and color decoding | `sh.py`, `wavelet_volume.py` | Ready |
| Metrics (PSNR, SSIM) | `metrics.py` | Ready |
| Forward/inverse DWT | `ptwt` (external) | Ready |
| Occupancy estimation | `occupancy.py` | Ready |
| Config system | `config.py` | Needs new fields |

### 7.2 What Needs to Be Written

#### New file: `train_frequency_matched.py` (~300 lines)

The core training script implementing the frequency-matched pipeline. Orchestrates the multi-stage training loop with per-stage dataset resolution, wavelet upsampling transitions, and per-stage validation.

```
train_frequency_matched(config)
├── for each stage in config.fm_stages:
│   ├── Load dataset at stage.image_resolution
│   ├── Initialize or upsample grid
│   ├── Create optimizer with stage-specific learning rates
│   ├── Training loop (stage.iterations)
│   │   ├── Sample random rays
│   │   ├── Volume render (coarse + fine pass)
│   │   ├── Compute L1 + TV + sparsity loss
│   │   ├── Backward + optimizer step
│   │   ├── Periodic pruning
│   │   └── Periodic validation (at stage res + 800px)
│   ├── Log stage metrics
│   └── Wavelet upsample grid for next stage
├── Final evaluation on full test set at 800px
├── Wavelet analysis (DWT, coefficient statistics)
└── Save checkpoint + wavelet coefficients
```

Key functions:
- `wavelet_upsample(grid, target_size, wavelet)` — as described in Section 4.4
- `train_stage(grid, dataset, config, stage_config)` — single-stage training loop
- `wavelet_analysis(grid, wavelet, levels)` — DWT + coefficient statistics
- `rate_distortion_curve(coeffs, test_data, thresholds)` — compression analysis

#### Modifications to `config.py`

Add frequency-matched configuration:

```python
# Frequency-matched training
fm_stages: list = None  # [(image_res, volume_res, iterations), ...]
fm_wavelet: str = "bior4.4"  # Wavelet for upsampling transitions
fm_lr_density_mult: float = 2.0  # Density LR multiplier relative to SH
fm_tv_start: float = 1e-4  # TV regularization start weight
fm_tv_end: float = 1e-6  # TV regularization end weight
fm_prune_every: int = 5000  # Voxel pruning interval
fm_prune_threshold: float = 0.01  # Density threshold for pruning
```

#### New file: `run_poc.py` (~100 lines)

Convenience script that runs all four configurations (FM, SP, SS, hybrid) and produces comparison outputs:

```bash
python run_poc.py --scene lego --output_dir output/fm_poc
```

Produces:
```
output/fm_poc/
├── frequency_matched/     # FM results
├── standard_progressive/  # SP results
├── single_scale/          # SS results
├── hybrid/                # Wavelet-upsample + full-res images
├── comparison/            # Side-by-side renders
├── wavelet_analysis/      # DWT coefficient plots
└── report.json            # All metrics in one file
```

#### New file: `wavelet_analysis.py` (~150 lines)

Post-training analysis tools:
- `analyze_coefficients(grid, wavelet, levels)` → histogram, energy distribution
- `rate_distortion(grid, wavelet, levels, test_dataset, thresholds)` → R-D curve data
- `render_coarse_only(grid, wavelet, levels, test_dataset)` → coarse-only renders
- `plot_comparisons(results_dict)` → matplotlib comparison figures

### 7.3 What Does NOT Need to Change

- `renderer.py` — the volume rendering code handles any dense grid, no changes needed
- `data.py` — already supports arbitrary resolution via Lanczos
- `metrics.py` — standard PSNR/SSIM computation
- `wavelet_volume.py` — not used for the PoC (we train a raw grid, DWT is post-hoc)
- `tiled_wavelet_volume.py` — not needed for 512³ (dense fits in memory)

---

## 8. Memory Budget

### 8.1 Per-Stage Memory (28 channels, float32)

| Stage | Grid size | Grid memory | Adam states | Gradients | Ray buffers | **Peak** |
|-------|-----------|-------------|-------------|-----------|-------------|----------|
| 0 (128³) | 128³ × 28 × 4 | 0.23 GB | 0.47 GB | 0.23 GB | ~0.5 GB | **~1.5 GB** |
| 1 (256³) | 256³ × 28 × 4 | 1.88 GB | 3.75 GB | 1.88 GB | ~0.5 GB | **~8 GB** |
| 2 (512³) | 512³ × 28 × 4 | 15.0 GB | 30.0 GB | 15.0 GB | ~1 GB | **~61 GB** |
| 3 (512³, fine-tune) | same | 15.0 GB | 30.0 GB | 15.0 GB | ~2 GB | **~62 GB** |

### 8.2 Hardware Requirements

| GPU | Max stage | Notes |
|-----|-----------|-------|
| 24 GB (RTX 4090) | Stage 1 (256³) | Stages 0–1 only |
| 48 GB (RTX 6000 Ada) | Stage 2 (512³) | Tight with 28 channels |
| 80 GB (A100/H100) | Stage 2 (512³) | Comfortable |
| 192 GB (B200) | Stage 2 (512³) | Abundant headroom |
| 288 GB (B300) | Stage 3+ (1024³) | Future extension |

**For the PoC on B200:** All stages fit comfortably on a single GPU.

**For development/debugging on a consumer GPU:** Use 4 channels (density + RGB, no SH) to reduce memory 7×. Stage 2 at 512³ × 4ch requires ~8.7 GB — fits on any modern GPU. Quality will be lower (no view dependence) but sufficient to validate the frequency-matching principle.

---

## 9. Expected Results

### 9.1 Quality Predictions

| Method | Expected final PSNR (800px test) | Rationale |
|--------|--------------------------------|-----------|
| SS (512³ @ 800px) | ~31–32 dB | Standard result at this resolution |
| SP (128→512³ @ 800px) | ~32–33 dB | Progressive init helps |
| **FM (128→512³ @ 100→400px)** | **~33–34 dB** | Clean coarse levels, no compromise artifacts |
| FM + stage 3 (800px fine-tune) | ~33.5–34.5 dB | Super-resolution refinement adds polish |

For reference: svox2 at 256³ achieves ~33.2 dB on lego. At 512³ with proper training, we should match or exceed this.

### 9.2 Wavelet Sparsity Predictions

After applying 3-level DWT to the final 512³ grid:

| Method | Fraction of coefficients needed for 30 dB | For 32 dB |
|--------|------------------------------------------|-----------|
| SS | ~15% | ~40% |
| SP | ~12% | ~35% |
| **FM** | **~8%** | **~25%** |

The FM approach should produce sparser wavelet coefficients because each frequency band was trained cleanly against matched supervision, avoiding cross-band contamination.

### 9.3 Coarse Reconstruction Predictions

Rendering from approximation coefficients only (zeroing all detail levels):

| Method | Coarse-only PSNR (at 100px test) |
|--------|--------------------------------|
| SS | ~22–24 dB (compromised coarse) |
| SP | ~24–26 dB (slightly better, still compromised) |
| **FM** | **~28–30 dB** (optimal for its band) |

This is the most dramatic predicted difference. The FM coarse should look clean and correct at low resolution, while SS/SP coarse levels should show visible artifacts from trying to approximate high-frequency content.

### 9.4 What Success Looks Like

The PoC succeeds if:

1. **FM final PSNR ≥ SP final PSNR** (equal or better quality)
2. **FM coarse PSNR >> SP coarse PSNR** (dramatically better coarse reconstruction)
3. **FM rate-distortion curve above SP curve** (better compression efficiency)
4. **FM wavelet coefficients are measurably sparser** at each detail level

Even if (1) shows only marginal improvement at full resolution, results (2)–(4) would validate the core principle: frequency-matched training produces a fundamentally cleaner multi-resolution decomposition.

---

## 10. Execution Plan

### Phase 1: Infrastructure (1 day)

1. Write `train_frequency_matched.py` with the multi-stage training loop
2. Implement `wavelet_upsample()` for inter-stage transitions
3. Add FM configuration fields to `config.py`
4. Test on a single stage (128³ @ 100px) to verify the pipeline works
5. Download the lego dataset to B200 if not already present

### Phase 2: Training Runs (1 day)

Run all four configurations on B200:

```bash
# Frequency-matched
python train_frequency_matched.py --mode fm --scene lego

# Standard progressive (same script, full-res images)
python train_frequency_matched.py --mode sp --scene lego

# Single-scale
python train_frequency_matched.py --mode ss --scene lego

# Hybrid (wavelet upsample + full-res)
python train_frequency_matched.py --mode hybrid --scene lego
```

Each run: ~1–2 hours on B200. Total: ~6 hours (can run in parallel if memory allows — stage 0–1 of one config while stage 2 of another).

### Phase 3: Analysis (half day)

1. Run `wavelet_analysis.py` on all four final grids
2. Generate rate-distortion curves
3. Render coarse-only reconstructions
4. Create comparison images and plots
5. Write up results

### Phase 4: Extension (optional)

If results are positive:
1. Add stage 3 at 1024³ using `TiledWaveletVolume` (sparse storage)
2. Run on other NeRF Synthetic scenes (chair, drums, hotdog) to verify generality
3. Compare wavelet families (Haar vs. bior4.4 vs. db4) for the transitions
4. Measure training time per stage — FM may converge faster at each stage

---

## 11. Stretch Goal: 1024³ with Tiled Wavelet Volume

If the 512³ PoC validates the approach, the natural next step is stage 4 at 1024³ against 800px images — a true frequency-matched stage where the volume resolution matches the image resolution.

This requires sparse storage (1024³ × 28ch × 4B = 120 GB dense, but only ~3% occupied = 3.6 GB sparse). The existing `TiledWaveletVolume` with `SparseDetailLevel` handles this:

1. After stage 2 (512³), compute occupancy mask from the density field
2. Apply 1-level DWT to the 512³ grid → approx (~260³) + detail (~260³)
3. Create `TiledWaveletVolume` with the 512³ wavelet coefficients as frozen base
4. Allocate sparse detail level at 1024³ resolution with occupancy mask
5. Train the sparse fine level against 800px images

This connects the PoC directly to the architecture we've designed for the 8192³ volumetric capture system — validating the pipeline end-to-end on a manageable scene before scaling up.

---

## 12. Connection to the Full System

The NeRF Synthetic Lego PoC validates the foundational claim of the frequency-matched approach at a scale that can be run in hours. The same principles apply directly to the volumetric capture stage:

| PoC (Lego) | Production (Capture Stage) |
|------------|--------------------------|
| 100 views, 800px | 120 cameras, 5136px |
| 3-unit scene cube | 2m performer cube |
| 128³ → 256³ → 512³ → 1024³ | 264³ → 520³ → 1031³ → 2054³ → 4100³ → 8192³ |
| ~80K iterations, 1 hour | ~300K iterations, several hours |
| Single B200 GPU | Single B300 GPU |
| bior4.4 wavelet | bior4.4 wavelet |
| Dense + optional sparse | Dense base + sparse fine levels |
| White background | Gray cyclorama + SAM segmentation |

The key algorithmic components are identical: Lanczos image downsampling, wavelet upsampling transitions, per-stage frequency-matched training, and post-hoc DWT analysis for compression. If the lego PoC shows the predicted improvements in coarse quality and wavelet sparsity, the same approach will produce cleaner, more compressible reconstructions of human performers at 8192³.
