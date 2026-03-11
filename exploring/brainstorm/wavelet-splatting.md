# Wavelet Splatting: Difference-of-Gaussians as Projectable Wavelet Primitives

**Date:** March 2026
**Context:** Exploring whether wavelet compression benefits can be combined with splatting-based rendering, avoiding the volume rendering bottleneck documented in `wavelets.md` and `wavelets-implementation-insights.md`.

---

## 1. The Question

Can we build a splattable primitive that behaves like a wavelet — with zero mean, vanishing moments, and multi-resolution structure — while retaining the projection and rasterization efficiency of Gaussian splatting?

The answer is yes, with caveats. The primitive is a **Difference of Gaussians (DoG)**, and the caveat is that the rendering model must change from multiplicative alpha compositing to a hybrid base+detail architecture.

---

## 2. The Difference of Gaussians as a 3D Wavelet

### 2.1 Definition

A 3D DoG wavelet centered at position μ with orientation R and scale σ:

```
ψ(x) = G(x; μ, σ²Σ) − G(x; μ, kσ²Σ)
```

where:
- `G(x; μ, Σ)` is a 3D Gaussian with mean μ and covariance Σ
- `Σ = R · diag(s²) · Rᵀ` defines the shape and orientation (anisotropic)
- `k > 1` is the scale ratio between the two Gaussians (typically k = 2 or k = √2)
- The narrow Gaussian `G(σ)` forms the positive center lobe
- The broad Gaussian `G(kσ)` forms the negative surround

### 2.2 Wavelet Properties

**Zero mean:**
```
∫ψ(x)dx = ∫G(x; σ)dx − ∫G(x; kσ)dx = 1 − 1 = 0   ✓
```
The DoG integrates to zero — it is a proper wavelet, not a scaling function. It adds detail without changing the global average.

**Band-pass behavior:**
In Fourier space, the DoG is the difference of two low-pass filters, giving a band-pass response centered at spatial frequency ~1/σ. Each DoG wavelet captures detail at a specific spatial scale.

**Vanishing moments:**
The DoG has one vanishing moment (the zeroth — zero mean). True Daubechies/CDF wavelets have more vanishing moments, which means they annihilate higher-order polynomials and compress smooth signals more efficiently. The DoG is equivalent to a Ricker wavelet (Mexican hat) approximation — not optimal in a formal wavelet theory sense, but practically effective. It's the same approximation used in SIFT for blob detection, validated across decades of computer vision.

**Localized support:**
Both component Gaussians decay exponentially. The DoG is effectively zero beyond ~3kσ from the center. Support is compact in practice.

### 2.3 Relationship to the Laplacian of Gaussian

The DoG is a well-known approximation to the **Laplacian of Gaussian** (LoG / Mexican hat / Ricker wavelet):

```
∇²G(x; σ) ≈ (1/σ²) · [G(x; σ) − G(x; kσ)]    for k close to 1
```

The LoG is a true second-derivative wavelet with two vanishing moments. As k → 1, the DoG converges to the LoG. For practical k values (√2 or 2), the DoG is a coarser approximation but retains the essential properties: zero mean, band-pass, localized.

---

## 3. Projection: 3D DoG → 2D DoG

This is why the DoG is special among wavelets: **both component Gaussians project cleanly to 2D**.

### 3.1 Gaussian Projection (Review)

A 3D Gaussian `G(x; μ, Σ)` viewed from a camera with projection matrix P and local Jacobian J (at the Gaussian's center) projects to a 2D Gaussian:

```
G_2D(u; Pμ, JΣJᵀ)
```

This is the standard EWA splatting result used in 3DGS.

### 3.2 DoG Projection

Since projection is a linear operation on each Gaussian independently:

```
ψ_2D(u) = G_2D(u; Pμ, σ²JΣJᵀ) − G_2D(u; Pμ, kσ²JΣJᵀ)
```

The 3D DoG projects to a **2D DoG** — a narrow positive Gaussian minus a broad negative Gaussian, both centered at the same projected point.

This is exact (up to the local-affine approximation at the center, same as standard 3DGS). No new approximations are introduced.

### 3.3 What the 2D DoG Footprint Looks Like

On the image plane, a projected DoG wavelet looks like a bright center surrounded by a dark ring (or vice versa, depending on the sign of the coefficient). It's the same pattern as a Mexican hat wavelet in 2D — the fundamental edge/blob detector in image processing.

```
               ┌─────────────────────┐
               │     − − − − −       │
               │   − − − − − − −     │
               │  − − + + + + − −    │    + = positive contribution
               │  − − + + + + − −    │    − = negative contribution
               │  − − + + + + − −    │
               │   − − − − − − −     │    Zero-crossing ring between
               │     − − − − −       │    the two Gaussians
               └─────────────────────┘
```

The positive center has amplitude A₊ and width ~σ. The negative ring has amplitude A₋ and extends to ~kσ. The total integral over the 2D footprint is zero.

---

## 4. The Rendering Model

### 4.1 Why Standard Alpha Compositing Breaks

Standard 3DGS compositing:

```
C(pixel) = Σᵢ cᵢ · αᵢ · Πⱼ<ᵢ (1 − αⱼ)
```

This requires αᵢ ∈ [0, 1]. A DoG wavelet has negative lobes — it cannot serve as an opacity value. Forcing it through sigmoid/clamp destroys the zero-mean property that makes it a wavelet.

### 4.2 The Hybrid Base+Detail Model

The solution is to split the representation into two layers with different compositing rules:

**Base layer (scaling function splats):**
- Standard 3D Gaussians (or Beta kernels)
- Non-negative opacity, full alpha compositing
- Handles occlusion, depth ordering, global appearance
- Represents the coarse, low-frequency scene content
- This is exactly what current 3DGS/Beta Splatting already does

**Detail layers (wavelet splats):**
- DoG wavelet primitives at multiple scales
- Signed color contributions, additive compositing
- No depth ordering needed (additive is commutative)
- Gated by base layer visibility (only contribute where base layer has accumulated alpha)
- Represent high-frequency edges, texture, fine detail

**Compositing equation:**

```
C(pixel) = C_base(pixel) + α_base(pixel) · C_detail(pixel)
```

where:

```
C_base(pixel) = Σᵢ cᵢ · αᵢ · Πⱼ<ᵢ (1 − αⱼ)              [standard alpha compositing]
α_base(pixel) = 1 − Πᵢ (1 − αᵢ)                            [accumulated base alpha]
C_detail(pixel) = Σₗ Σₖ dₖₗ · ψₖₗ(pixel)                   [additive, per detail level l]
```

The `α_base` gating ensures detail contributions only appear where the base layer has rendered a surface. This prevents wavelet artifacts in empty/background regions.

### 4.3 Why Additive Compositing Works for Detail

The detail layer doesn't need depth ordering because it represents **corrections to an already-rendered surface**, not independent surfaces. Consider what the detail splats are doing:

- A positive DoG on the forehead brightens the center and darkens the surround → sharpens a highlight
- A negative DoG on a fabric fold darkens the center and brightens the surround → sharpens a shadow edge
- Multiple DoGs at different scales and positions build up a Laplacian pyramid of corrections

This is mathematically equivalent to a **Laplacian pyramid** applied to the rendered image — the base layer is the coarsest Gaussian pyramid level, and each DoG detail level adds one octave of spatial frequency. Laplacian pyramids are purely additive in reconstruction.

### 4.4 Depth-Aware Detail Gating (Handling Multiple Surfaces)

The simple `α_base` gating above works for a single surface per pixel. For scenes with multiple overlapping surfaces (a hand in front of a torso), each detail splat should only correct the surface it belongs to, not surfaces behind it.

**Solution:** Associate each detail splat with a depth range (inherited from its parent base splat or spatial neighborhood). During rasterization:

```
C_detail(pixel) = Σₖ dₖ · ψₖ(pixel) · w(zₖ, z_base(pixel))
```

where `w(zₖ, z_base)` is a soft depth gating function that weights the detail contribution by proximity to the visible base surface depth at that pixel. A Gaussian falloff in depth works:

```
w(zₖ, z_base) = exp(−(zₖ − z_base)² / (2σ_z²))
```

This ensures a detail splat on the hand doesn't also sharpen the torso behind it.

---

## 5. Multi-Resolution Hierarchy

### 5.1 Scale Levels

Organize DoG wavelet splats into discrete scale levels, analogous to wavelet decomposition levels:

```
Level 0 (base):    Standard Gaussians/Beta kernels, scale ~5-20mm
                   Full alpha compositing, handles occlusion
                   ~50K-200K splats (like a coarse 3DGS)

Level 1 (detail):  DoG wavelets, σ₁ ~ 5-10mm
                   Captures features at ~1cm scale
                   ~200K-500K splats

Level 2 (detail):  DoG wavelets, σ₂ ~ 2-5mm
                   Captures features at ~5mm scale (wrinkles, fabric weave)
                   ~500K-2M splats

Level 3 (detail):  DoG wavelets, σ₃ ~ 1-2mm
                   Captures features at ~2mm scale (pores, fine texture)
                   ~1M-4M splats
```

At each level, the DoG scale ratio k defines the octave bandwidth. With k = 2, each level covers one octave of spatial frequency — standard wavelet practice.

### 5.2 Spawning Detail Splats

Detail splats are spawned from base splats (or coarser detail splats) during training, analogous to 3DGS densification:

1. After Stage 1 (base layer converges), compute per-pixel residual: `R = I_gt − I_base`
2. The residual `R` is the signal that detail splats must reconstruct
3. Spawn Level 1 DoG splats at locations where the residual has high energy (large gradients in the base-layer rendered image)
4. Train Level 1 to minimize ||R − I_detail_1||
5. Compute new residual: `R₂ = R − I_detail_1`
6. Spawn Level 2 DoG splats where R₂ has high energy
7. Repeat

This is exactly **Laplacian pyramid construction**, implemented with splattable primitives.

### 5.3 Rate-Distortion Control

To hit a target memory budget:

1. Train all levels
2. Rank all detail splats by |dₖ| · energy(ψₖ) — the magnitude of the coefficient times the basis function's energy (L2 norm of the projected 2D footprint)
3. Prune the lowest-ranked splats until the memory budget is met
4. This is wavelet coefficient thresholding — provably optimal for the class of piecewise smooth signals

**Graceful degradation:** Dropping the finest detail level reduces memory by ~50% while only removing the highest-frequency detail (pores, fine texture). The image becomes slightly softer but maintains correct edges and mid-frequency structure. This is fundamentally smoother degradation than pruning Gaussians, where removing any primitive can create holes.

---

## 6. Training / Optimization

### 6.1 Progressive Training Schedule

```
Phase 1:  Train base layer only (standard 3DGS/Beta Splatting)
          → converge to coarse appearance
          → 10K-30K iterations

Phase 2:  Freeze base, spawn + train Level 1 detail
          → learn largest-scale corrections (broad shadows, large highlights)
          → 5K-10K iterations

Phase 3:  Freeze base + Level 1, spawn + train Level 2 detail
          → learn medium-frequency detail (wrinkles, fabric folds)
          → 5K-10K iterations

Phase 4:  Unfreeze all levels, joint fine-tuning
          → allow all levels to adjust together
          → 5K-10K iterations
          → apply coefficient thresholding / pruning
```

### 6.2 Gradient Flow

The rendering equation is fully differentiable:

```
C(pixel) = C_base(pixel) + α_base(pixel) · Σₖ dₖ · ψₖ(pixel)
```

Gradients with respect to:

- **dₖ (DoG coefficient/color):** ∂C/∂dₖ = α_base · ψₖ(pixel). Linear, clean gradient.
- **μₖ (DoG position):** ∂C/∂μₖ = α_base · dₖ · ∂ψₖ/∂μₖ. The spatial derivative of the DoG is a third-derivative-of-Gaussian — well-defined, localized.
- **σₖ (DoG scale):** ∂C/∂σₖ involves derivatives of both component Gaussians with respect to scale — standard, same as 3DGS scale gradients but applied twice (once per Gaussian in the DoG).
- **Base layer parameters:** Gradients flow through both C_base and α_base. The α_base gating means the base layer receives gradient signal from both its own photometric loss *and* from enabling/disabling detail contributions.

No new mathematical difficulties vs. standard 3DGS backpropagation. The DoG is a linear combination of two Gaussians, so all derivatives decompose into standard Gaussian derivatives.

### 6.3 The Detail Color Space

Each DoG splat carries a signed color value `dₖ ∈ ℝ³` (not clamped to [0,1]). This represents the color correction at that location:

- `dₖ = (+0.1, +0.05, +0.02)` → adds a warm highlight
- `dₖ = (−0.05, −0.05, −0.03)` → darkens (shadow edge sharpening)
- `dₖ = (+0.1, −0.05, −0.05)` → shifts color toward red (e.g., subsurface scattering edge)

The final pixel color `C_base + α_base · C_detail` is clamped to [0,1] only at the end, after all layers are composited. The intermediate detail signal is unclamped.

---

## 7. Connection to Beta Splatting

### 7.1 DoG with Beta Kernels

The DoG doesn't have to use Gaussians — it can use **any kernel that projects cleanly to 2D**. Beta kernels from Beta Splatting qualify:

```
ψ_beta(x) = B(x; μ, Σ, b₁) − B(x; μ, kΣ, b₂)
```

where B is the Beta kernel with shape parameter b. This gives additional control:

- b₁ > 0 (peaked narrow kernel) minus b₂ < 0 (flat broad kernel) → very sharp central peak with a soft negative surround
- b₁ < 0 (flat narrow kernel) minus b₂ < 0 (flat broad kernel) → plateau-like center with sharp negative edges

The Beta shape parameter lets the DoG wavelet adapt its frequency response beyond what a pure Gaussian DoG can achieve. A peaked narrow kernel produces a tighter band-pass — more selective frequency capture. A flat narrow kernel produces a broader band-pass — captures a wider range of spatial frequencies in a single splat.

### 7.2 Fitting Into the Existing Pipeline

The base layer of wavelet splatting IS standard Beta Splatting. The detail layers are an addition on top. This means the existing pipeline (Stages 1-3 in the optimization protocol) trains the base layer exactly as designed. Wavelet detail levels are an optional enhancement that can be added after Stage 3 converges:

```
Existing pipeline:
  Stage 1 (geometry) → Stage 2 (shape) → Stage 3 (appearance)
                                                    ↓
                                          Wavelet enhancement (optional):
                                            Detail Level 1 → Level 2 → Level 3
                                                    ↓
                                          Joint fine-tuning (all levels)
```

This is non-disruptive — the wavelet enhancement is additive to an already-working pipeline.

---

## 8. What This Actually Buys You

### 8.1 vs. Standard Beta Splatting (More Splats)

The naive alternative to wavelet detail splats is just adding more Beta splats during densification. Why would DoG wavelets be better?

**Edge representation efficiency:**
A sharp edge in the rendered image (e.g., the boundary between skin and hair) requires many overlapping Gaussians/Beta kernels to approximate — each one is smooth, and their sum must produce a sharp transition. A single DoG wavelet at the right scale naturally produces a sharp transition (positive on one side, negative on the other). Fewer primitives for the same edge quality.

**Formal pruning criterion:**
In standard 3DGS, the pruning criterion (opacity < threshold) is heuristic. A low-opacity splat might be critically important for a subtle detail or completely irrelevant — opacity alone can't distinguish. DoG coefficient magnitude `|dₖ|` directly measures the splat's contribution to image quality, giving principled pruning.

**Scale separation:**
Standard densification creates splats at whatever scale the gradient pressure dictates, mixing coarse and fine splats with no organization. The multi-resolution DoG hierarchy explicitly separates scales, enabling:
- Level-of-detail rendering (drop fine levels for distant views)
- Targeted quality: allocate more detail levels to the face, fewer to the shoes
- Clean rate-distortion tradeoff

**Quantitative expectation:**
For equivalent image quality (PSNR), a wavelet-enhanced model should require **30-50% fewer total primitives** than a flat splat model, based on the known compression efficiency of wavelet vs. direct representations in 2D image coding. The savings come primarily from edges and texture regions where the multi-resolution structure avoids redundant overlapping splats.

### 8.2 vs. Full Wavelet Volume Rendering (wavelets.md approach)

The wavelet volume approach (documented in `wavelets.md` and `wavelets-implementation-insights.md`) provides better theoretical compression and true rate-distortion optimality. Wavelet splatting trades some of that optimality for:

**Rendering speed:**
Splatting is fundamentally faster than volume rendering. No ray marching, no per-sample volume queries, no scattered coefficient gathering. The DoG rasterization adds ~2× the cost of standard Gaussian rasterization (two Gaussians per DoG splat), but this is still order-of-magnitude faster than ray marching through a wavelet volume.

**Existing infrastructure:**
The 3DGS CUDA rasterizer is mature, optimized, and widely available. Wavelet splatting extends it (signed compositing, multi-pass rendering) rather than replacing it. The wavelet volume renderer requires building the entire rendering pipeline from scratch (the tiled IDWT + ray marcher documented in the implementation insights).

**Compatibility with body model anchoring:**
The LBS-anchored position parameterization (`Position_world(t) = LBS(θ_t, β, w_i) · (v_canonical_i + Δ_i)`) applies directly to DoG splats — each DoG is positioned and deformed just like a regular splat. The wavelet volume approach has no natural way to integrate body model deformation (the volume grid doesn't deform with LBS; you'd need to warp the coordinate system, which introduces resampling artifacts).

**What you give up:**
- Formal rate-distortion optimality (the DoG is an approximate wavelet — only 1 vanishing moment vs. CDF 9/7's 4)
- The spatial organization of the wavelet volume (hash-map coefficient lookup is theoretically more cache-friendly than scattered splat evaluation)
- True multi-resolution volume queries (the volume approach naturally handles coarse-to-fine evaluation; the splat approach must explicitly manage the level hierarchy)

---

## 9. Rasterizer Modifications

### 9.1 What Changes in the CUDA Rasterizer

The standard 3DGS CUDA rasterizer (diff-gaussian-rasterization) needs these modifications:

**Pass 1: Base layer (unchanged)**
- Standard tile-based sorting, front-to-back alpha compositing
- Output: `C_base` (RGB), `α_base` (accumulated alpha), `z_base` (depth of primary surface)

**Pass 2: Detail layer (new)**
- For each DoG wavelet splat: evaluate both component Gaussians, subtract, multiply by signed color
- Additive accumulation (no depth sorting needed — commutative)
- Gate by `α_base` (skip pixels where base alpha is near zero)
- Optionally gate by `z_base` proximity (for multi-surface scenes)
- Output: `C_detail` (RGB, signed, unclamped)

**Final compositing:**
```
C_final = clamp(C_base + α_base · C_detail, 0, 1)
```

**The additive detail pass is simpler than the base pass** — no sorting, no transmittance tracking. The main cost is evaluating two Gaussians per DoG splat instead of one. Tile-based culling still applies (use the broader Gaussian's extent for the tile intersection test).

### 9.2 Performance Estimate

| Component | Relative Cost |
|---|---|
| Base pass (standard 3DGS) | 1.0× |
| Detail pass (DoG, additive, no sorting) | 0.5-0.8× per detail level |
| With 2 detail levels | Total: ~2.0-2.6× standard 3DGS |
| With 3 detail levels | Total: ~2.5-3.4× standard 3DGS |

At 3× the cost of standard 3DGS, this is still real-time (standard 3DGS renders at 100+ FPS for typical scenes; 3× cost → 30-40 FPS, still interactive).

---

## 10. Temporal Extension (4D)

### 10.1 Temporal DoG Wavelets

For dynamic scenes, the same DoG principle extends to time. A temporal DoG at time t with temporal scale τ:

```
ψ_temporal(t) = G(t; t₀, τ²) − G(t; t₀, kτ²)
```

This captures temporal detail at scale τ — a brief flash, a fast gesture, a momentary expression. Static regions produce zero temporal DoG response (zero coefficient).

### 10.2 4D DoG Splat

A full 4D DoG wavelet splat has:
- Spatial position μ(x,y,z) — or LBS-anchored canonical position + offset
- Temporal center t₀ and temporal scale τ
- Spatial DoG scale σ and ratio k
- Signed color d ∈ ℝ³

The 4D DoG is separable as spatial × temporal, or can be a full 4D anisotropic DoG for coupled spatiotemporal detail (e.g., a wrinkle that appears during a frown — spatial detail correlated with a temporal event).

### 10.3 Temporal Compression Behavior

| Scene Region | Temporal Behavior | Temporal DoG Coefficients |
|---|---|---|
| Static background | Constant | Zero (only DC in base layer) |
| Slow smooth motion | Low frequency | Few coarse temporal DoGs |
| Fast periodic motion (walking) | Band-limited | Moderate temporal DoGs at the motion frequency |
| Transient event (blink, clap) | Impulse-like | Many fine temporal DoGs, but temporally localized |

The temporal DoG hierarchy provides the same rate-distortion benefits in time as the spatial DoGs provide in space: allocate bits where the temporal complexity demands it, save bits on static or smoothly moving regions.

---

## 11. Open Questions

### 11.1 Optimal Scale Ratio k

The ratio k between the two component Gaussians controls the wavelet's bandwidth:
- k = √2: narrow bandwidth, many levels needed, closer to a true continuous wavelet
- k = 2: one-octave bandwidth, standard dyadic wavelet, fewer levels needed
- k = 4: very broad bandwidth, coarse approximation, fewest levels

Empirical testing needed. Start with k = 2 (standard practice in scale-space theory).

### 11.2 Base/Detail Split Point

How many splats go in the base layer vs. detail layers? Too few base splats → poor occlusion handling, detail corrections must be large. Too many base splats → the detail layers add little value.

Hypothesis: the base layer should converge to roughly the same quality as a standard 3DGS at ~0.5-1× the usual splat count. Detail layers then add the final quality push more efficiently than adding more base splats would.

### 11.3 Negative Color Clamping

The final clamp `C_final = clamp(C_base + detail, 0, 1)` can create gradient issues where the detail signal pushes below 0 or above 1. Smooth approximations (softplus, smooth clamp) may help training stability.

### 11.4 Interaction with View-Dependent Color (SH)

Base splats use spherical harmonics for view-dependent appearance. Should detail splats also be view-dependent? Arguments both ways:
- **Yes:** Specular highlights are high-frequency, view-dependent features — exactly what detail splats should capture
- **No:** Adding SH coefficients to every detail splat increases memory significantly. The base layer's SH already handles gross view-dependence; detail splats could be view-independent (diffuse corrections only)

Practical compromise: Level 1 detail splats get low-order SH (band 1, 4 coefficients). Level 2+ detail splats are view-independent (3 RGB values only).

### 11.5 Does This Compose with the Wavelet Volume?

In principle, the two approaches are not mutually exclusive. The wavelet volume could provide the base layer (reconstructed to a coarse resolution, then splatted via marching cubes → mesh → rasterization), with DoG detail splats adding high-frequency corrections. This combines the volume's formal compression with the splats' rendering speed. But the engineering complexity is high and the benefit unclear — this is a much later exploration.

---

## 12. Implementation Path

### 12.1 Minimal Viable Experiment

1. Take a converged standard 3DGS scene (static, e.g., NeRF Synthetic or a single frame from PKU-DyMVHumans)
2. Render all training views → compute residuals R = I_gt − I_3DGS
3. Spawn DoG splats at high-residual locations
4. Train only the DoG detail layer (base frozen) to minimize ||R − I_detail||
5. Measure: PSNR improvement per additional MB of DoG parameters vs. per additional MB of standard splats (from further densification)

This directly tests the core hypothesis: are DoG wavelets more parameter-efficient than standard splats for high-frequency detail?

### 12.2 Full Integration

If the MVE validates the hypothesis:
1. Modify the diff-gaussian-rasterization CUDA kernel to support a two-pass base+detail pipeline
2. Implement DoG splat spawning during training (residual-driven, not gradient-driven like standard densification)
3. Add multi-level DoG hierarchy management
4. Integrate with the Beta Splatting + LBS pipeline for dynamic humans

### 12.3 What to Measure

| Metric | What It Tests |
|---|---|
| PSNR per MB | Core efficiency: wavelet detail vs. more standard splats |
| Edge sharpness (gradient magnitude at boundaries) | DoG advantage on edges |
| Pruning curve (quality vs. % detail splats retained) | Rate-distortion behavior |
| Render FPS (base only vs. base+1 detail vs. base+2 detail) | Performance cost |
| Training convergence (iterations to target PSNR) | Whether progressive training helps |
