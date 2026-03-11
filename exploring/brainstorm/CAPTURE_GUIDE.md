# 4D Gaussian Splatting: Capture Configuration Guide

Recommendations for volumetric capture optimized for 4D Gaussian Splatting reconstruction.

---

## Camera Count

The key metric is not total camera count but **angular density relative to the scene**, and specifically how many cameras have an **unoccluded view** of each surface point.

The Gaussian model uses SH degree 3 (16 coefficients per color channel). Each point needs ~16 distinct viewpoints for proper SH fitting. Beyond that, additional views provide diminishing returns.

### Recommendations by Scene Type

| Scene | Cameras | Rationale |
|-------|---------|-----------|
| Single person, static pose | 30-50 | Minimal occlusion, simple geometry |
| Single person, moderate motion | 60-80 | Some self-occlusion from limb crossings |
| **Single performer, dancing/singing** | **80-90** | **Flying hair, fabric, extreme poses create heavy self-occlusion** |
| 2-3 performers, moderate stage | 90-120 | Inter-performer occlusion |
| 8+ performers, large stage | 120-150+ | Scale with ~20-30 cameras per independently-moving subject |

### Diminishing Returns

| Cameras | Angular spacing (half-sphere) | Quality gain per added camera |
|---------|------------------------------|------------------------------|
| 30 | ~35° | High |
| 60 | ~17° | High |
| 80-90 | ~11-13° | Moderate |
| 120 | ~10° | Low |
| 150+ | ~8° | Very low — exceeds SH3 angular resolution (~15°) |

Beyond ~90 cameras for a single subject, additional cameras mainly provide insurance against occlusion rather than angular detail. The SH degree 3 representation itself becomes the quality bottleneck.

---

## Resolution

### 4K is the Sweet Spot for Training

The Gaussian model's spatial detail is limited by Gaussian count and scale, not pixel count. The published literature trains at 1352×1014 (half of native 2704×2028) and achieves 32+ dB PSNR. Full 4K training is already above what has been demonstrated.

- **4K (3840×2160)**: Recommended training resolution. Sufficient supervision for Gaussian fitting.
- **8K (7680×4320)**: Not recommended as training resolution. 4× more pixels, ~10-15× slower per training step, marginal quality improvement. The model can't exploit the extra detail.

### 8K as a Noise Reduction Tool

Filming in 8K and downsampling to 4K for training is valuable — not for resolution, but for **noise and aliasing reduction**:

- Each output pixel averages 4 input pixels
- Sensor noise (independent per pixel) reduces by sqrt(4) = **2× (6 dB SNR improvement)**
- Eliminates Moire patterns and aliasing artifacts on fine textures
- Gives cleaner training data for Gaussian fitting

This is particularly useful when combined with high frame rates (see below).

---

## Frame Rate

### The Temporal Model Constraint

Our 4DGS uses a linear velocity model: `position(t) = mean + velocity × (mu_t - t)`. Between frames, any non-linear motion (rotation, acceleration, deformation) creates approximation error. Faster sampling = better linear approximation.

| FPS | Frame gap | Hand motion in gap (fast dancer) | Linear approx quality |
|-----|-----------|--------------------------------|----------------------|
| 30 | 33ms | ~30 cm | Poor for fast motion |
| 60 | 16.7ms | ~15 cm | Good for most motion |
| 120 | 8.3ms | ~7 cm | Excellent |

### The Noise Tradeoff

Higher FPS = shorter exposure = less light per frame = more noise.

Going from 30fps to 120fps means 4× less light per frame (+12 dB noise). To compensate:
- 4× more light intensity (heat, cost, power)
- Wider aperture (shallower DOF — bad for reconstruction)
- Higher ISO (more sensor noise — bad for Gaussian fitting)

**Sensor noise directly degrades 3DGS quality.** The optimizer creates spurious tiny Gaussians to fit noisy pixels, producing floater artifacts. Clean images matter more than high resolution or high frame rate.

### The 8K + High FPS Strategy

Filming in 8K and downsampling to 4K recovers 6 dB of SNR, which offsets the noise penalty of higher frame rates:

| Setup | Noise vs 4K@30fps | Temporal quality | Data volume (5 min) |
|-------|-------------------|-----------------|-------------------|
| 4K @ 30fps | Baseline | Low | 9,000 frames |
| 4K @ 60fps | +6 dB worse | Good | 18,000 frames |
| 4K @ 120fps | +12 dB worse | Excellent | 36,000 frames |
| 8K→4K @ 60fps | −6 dB better | Good | 18,000 frames |
| **8K→4K @ 120fps** | **Baseline** | **Excellent** | **36,000 frames** |

**8K capture at 120fps, downsampled to 4K** gives the same noise as native 4K at 30fps, but with 4× better temporal resolution. The extra pixels are used purely to offset the higher shutter speed.

### Training at Lower FPS Than Capture

You don't need to train at the capture frame rate:

1. **Capture at 120fps** for maximum temporal fidelity
2. **Train at 60fps** (every other frame) as the default — halves training time
3. For sequences with very fast motion (hair whips, fabric snaps), retrain those segments at full 120fps
4. The trained model renders at **any timestamp** — the temporal Gaussians interpolate smoothly between trained frames

---

## Priority Order

When optimizing a capture setup for 4DGS quality, invest in this order:

1. **Lighting** — the single biggest factor. Clean, well-lit images matter more than anything else. Low noise eliminates floater artifacts and produces sharp Gaussian boundaries.

2. **Camera count** — sufficient unoccluded views of every surface point. ~80-90 for a single dynamic performer, scale up with scene complexity and number of subjects.

3. **Frame rate** — highest FPS where images remain clean. 60fps is the practical target for most production stages. 120fps if lighting supports it.

4. **Lens quality** — sharp optics with minimal distortion. Soft or distorted images degrade multi-view consistency.

5. **Resolution** — 4K is sufficient. 8K is only useful as a noise reduction tool (downsample to 4K) to enable higher frame rates.

---

## Recommended Configurations

### Budget-Conscious (Single Performer)

- 60 cameras, 4K, 60fps
- Good studio lighting
- Training: ~60fps, all frames
- Quality: Good for moderate motion

### Production (Single Performer, High Dynamic)

- 90 cameras, 4K, 60fps
- Professional LED lighting
- Training: 60fps, 4K native
- Quality: High quality for dancing, singing, athletic performance

### Premium (Single Performer, Maximum Quality)

- 90 cameras, 8K sensors, 120fps capture
- High-intensity professional lighting (compensate for short exposure)
- Training: downsample to 4K, train at 60-120fps depending on motion
- Quality: Best achievable — low noise + high temporal resolution

### Multi-Performer (Large Stage)

- 120-150 cameras, 4K, 60fps
- Distributed lighting covering full stage
- Training: 60fps, scale GPU count with data volume
- Quality: Good coverage despite inter-performer occlusion
