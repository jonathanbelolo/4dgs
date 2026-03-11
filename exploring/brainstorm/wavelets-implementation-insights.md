# Wavelet Volume: Implementation Insights & Design Evolution

Findings from the first PoC implementation (static 3D, NeRF Synthetic lego scene).
Builds on the original proposal in `wavelets.md`.

---

## 1. Tiled IDWT Reconstruction

**Problem:** A full dense volume reconstruction at high resolution is prohibitive.
512³ × 28ch = 14 GB fp32. 1024³ = 112 GB. 2048³ = 896 GB.

**Solution:** The IDWT is a convolution with finite filter support. CDF 9/7 has ~4 taps
per axis. This means we can reconstruct the volume in independent spatial tiles with
a small overlap border (~4 voxels per face) and get identical results to a full
reconstruction.

**How it works:**
1. Always reconstruct coarse levels fully (they're tiny — 39³ for a 256³ target).
2. Use the coarse pass to identify which spatial regions contain density (occupancy mask).
3. At the finest detail level(s), only reconstruct tiles that intersect occupied regions.
4. For training: only reconstruct tiles intersected by the current ray batch.
5. For inference: only reconstruct tiles visible in the current camera frustum.

**Numbers for 2048³ (5 decomp levels, 128³ tiles):**
- Total tiles: 16³ = 4096
- Typical occupancy for a head: ~15-20% → 600-800 active tiles
- Per tile: 128³ × 28ch × 2 bytes (fp16) = 115 MB
- Active tile memory: 6-12 GB (vs 896 GB dense)
- Backprop works per-tile — gradients accumulate to the shared wavelet coefficients.

**This unlocks sub-millimeter resolution on a single GPU.**

---

## 2. Runtime Memory Budget (Inference)

For a trained, pruned model at inference time (no optimizer state, no gradients):

| Component | 2048³ full | 2048³ pruned + tiled |
|-----------|-----------|---------------------|
| Wavelet coefficients | 58 GB fp16 | ~6 GB (90% pruned) |
| Dense volume | 896 GB | 6-12 GB (active tiles only) |
| Render buffers | 1-2 GB | 1-2 GB |
| **Total** | ~955 GB | **13-20 GB** |

**Fits on an RTX 5090 (32 GB).** Tile eviction/loading as the camera moves, like
virtual texturing.

At 2048³ over a ~1m bounding box: **~0.5mm per voxel** — enough for individual skin
pores, fine wrinkles, peach fuzz follicle roots.

---

## 3. 4D Temporal Extension: Warm-Start Optimization

**Key insight:** For a video sequence, once frame 0 is fully optimized, subsequent
frames don't start from scratch. They initialize from the previous frame's converged
wavelet coefficients.

**Why this works exceptionally well for wavelets:**
- Wavelet coefficients are spatially localized at each detail level.
- Between frames, only coefficients in motion regions need to change.
- Coarse levels (global structure) barely change at all.
- Fine-level changes are sparse — small perturbations on an otherwise stable surface.

**Estimated per-frame optimization:**
- Frame 0: full optimization (50K iters, hours)
- Frame 1+: warm-start, ~2K-5K iters (~10-20 min per frame)
- With selective region optimization (freeze static regions): even faster

**Selective optimization by region:**
- Detect motion regions from capture data (optical flow, motion masks) or by monitoring
  which coefficients changed most during warm-start iterations.
- Freeze static-region coefficients — no gradient computation, no optimizer state needed.
- Only backpropagate through tiles intersecting motion regions.
- This cuts compute proportionally to the motion area fraction.

---

## 4. Temporal Smoothness Regularization

Add a penalty on coefficient changes between adjacent frames:

```
loss_temporal = lambda_t * ||coeffs_t - coeffs_{t-1}||_1
```

**Per-level weighting:** Stronger smoothness on coarse levels (global structure
shouldn't flicker between frames), weaker on fine levels (allow sharp motion detail).

```
loss_temporal = sum over levels:
    lambda_coarse * ||approx_t - approx_{t-1}||_1
  + lambda_fine   * ||detail_i_t - detail_i_{t-1}||_1
```

This prevents temporal flickering and encourages the optimizer to find minimal
coefficient changes, which also improves compression.

---

## 5. 4D Compression: I-Frame / P-Frame Analogy

For storage and streaming, the wavelet representation naturally supports a
video-codec-like structure:

- **I-frames:** Full wavelet coefficient set (e.g., every 10th frame). After pruning,
  ~50 MB per frame for a head capture.
- **P-frames:** Only the coefficient deltas from the previous frame. For typical
  motion (talking head), maybe ~2-5 MB per frame.
- **GOP (Group of Pictures):** 1 I-frame + 9 P-frames = ~70-100 MB per GOP
  → 10 seconds at 30fps ≈ **2-3 GB** for a full volumetric video clip.

This is orders of magnitude better than storing independent per-frame representations,
and the deltas are themselves sparse and wavelet-structured (compressible further with
entropy coding).

---

## 6. True 4D Wavelet Decomposition (Research Frontier)

Beyond frame-by-frame warm-start, the deeper opportunity is adding a temporal wavelet
axis. Instead of optimizing frame-by-frame:

- Group frames into GOPs (e.g., 16 frames)
- Apply a 4D wavelet decomposition: 3 spatial + 1 temporal
- Temporal low-frequency components = static background / slow motion
- Temporal high-frequency components = fast motion / transients
- Optimize the entire GOP's 4D wavelet coefficients jointly

**Advantages:**
- Temporal redundancy is captured in the representation itself, not just via
  warm-starting
- Pruning temporal high-frequency coefficients naturally compresses slow-moving regions
- One optimization pass per GOP instead of N passes per frame
- The rate-distortion framework extends naturally to the temporal axis

**This is the path forward.** The current static 3D PoC validates the spatial wavelet
representation. The next step is true 4D — no intermediate frame-by-frame warm-start
hack. We want the theoretically optimal solution: a single 4D wavelet decomposition
over space and time, optimized jointly.

The warm-start approach (section 3) and temporal regularization (section 4) are
documented for completeness but are NOT the plan. They are shortcuts that sacrifice
the formal compression-theoretic properties that make wavelets worth pursuing in the
first place. A true 4D decomposition captures temporal redundancy in the
representation itself — the temporal wavelet coefficients ARE the motion model.

---

## 7. PoC Findings So Far

**Architecture validated:**
- Learnable wavelet coefficients → IDWT (ptwt) → dense volume → trilinear query → SH
  decode → volume rendering. No MLP.
- CDF 9/7 (`bior4.4`) works with ptwt's incremental IDWT. Coefficient sizes are
  irregular (not exact powers of 2), resolved via dry-run `wavedec3`.
- Single-pass `reconstruct_pair()` avoids redundant IDWT computation for
  coarse + fine volumes.
- Gradients flow correctly to all wavelet coefficient levels.

**Progressive training works:**
- Start with coarse levels, activate finer levels over time.
- Loss jumps when a new level activates (expected — new coefficients near zero),
  then recovers.

**Resolution scaling (memory, single GPU):**

| Target | Params | Training mem | GPU |
|--------|--------|-------------|-----|
| 256³ | 531M | ~12 GB | Any |
| 512³ | 4B | ~89 GB | H100/B200 |
| 1024³ | 31B | ~17 GB (tiled) | B200 |
| 2048³ | - | ~20 GB (tiled) | B200/5090 |

**Training speed (H100, no tiling):**
- Level 0 (64³): ~3 iter/s
- Level 1 (128³): ~6 iter/100s
- Level 2 (256³): ~1 iter/s
- Compute-bound, not memory-bound (96% GPU util, 10/80 GB used)

**First run in progress:** NeRF Synthetic lego, 256³, 50K iterations.
PSNR at 25K: 11.27 dB (level 2 just activated, still converging).
