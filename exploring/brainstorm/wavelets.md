# Wavelet-Based 4D Scene Representation: A Vision for Compressed, Photorealistic Novel View Synthesis

## Motivation: The Compression Gap in Current Methods

### What 4D Gaussian Splatting Does Today

Current 4D Gaussian splatting represents dynamic scenes using a set of 3D Gaussian primitives, each parameterized by a mean (position), covariance matrix (shape and orientation), opacity, and spherical harmonic coefficients (view-dependent color). The temporal dimension is handled by learning deformation fields or per-primitive trajectories that warp a canonical set of Gaussians through time, with densification and pruning adjusting the number of primitives during optimization.

This works well in practice, converging to 1–5 million Gaussians for typical benchmark scenes and fitting comfortably in GPU memory. But the representation is heuristic — there is no formal framework governing how parameters are allocated across the scene, leading to well-known failure modes: floater artifacts, over-smoothed fine detail, and inefficient parameter usage where featureless walls consume as many bytes per primitive as complex textured regions.

### The Pixel-Level Baseline: How Much Data Are We Actually Compressing?

To understand the magnitude of compression these methods provide, consider what a full, uncompressed pixel-level 4D representation would require:

**Per-point storage (full parameterization):** ~200–250 bytes (position, scale, rotation, SH coefficients, opacity).

**Static scene point count:** For a room-scale benchmark scene (~5–10 m² of visible surfaces, captured at 2K resolution from 1–3m distance), pixel-level coverage requires roughly 10–100 million points at ~1mm spacing.

**Static scene memory:** 100M points × 250 bytes ≈ **25 GB** (full params) or ~4 GB (minimal RGB + position).

**4D extension (10 seconds at 30fps, 300 frames):**

| Approach | Estimate |
|---|---|
| Independent point cloud per frame | 100M × 300 frames × 250 bytes ≈ **7.5 TB** |
| Canonical + per-frame position deltas | Base 25 GB + 100M × 12 bytes × 300 ≈ **400–700 GB** |
| With topology changes | Even larger due to partial inter-frame independence |

**Current 4D Gaussian splatting compresses this by 2–3 orders of magnitude** (down to a few GB), exploiting surface smoothness, motion coherence, and smooth appearance variation. The question is whether we can achieve better compression — and better quality — using a representation with formal compression-theoretic foundations.

## The Core Idea: 3D+1D Wavelet Decomposition of the Scene

### Why Wavelets

Wavelets are the gold standard for compressing signals with spatial coherence and multi-scale structure — exactly the properties of 3D scenes. They provide:

- **Energy compaction:** Decompose the signal into a multi-resolution pyramid of coefficients. For piecewise smooth signals (a good model of 3D scenes), most coefficients are near-zero. Retain only the significant ones.
- **Provable rate-distortion optimality:** For a given bit budget, wavelet thresholding gives the best possible reconstruction among a broad class of representations.
- **Multi-resolution structure:** Coarse coefficients capture global shape; fine coefficients capture detail and edges. This naturally provides level-of-detail.
- **Localized support:** Each wavelet basis function is compact in both space and frequency, meaning coefficient updates during optimization are local and parallelizable.

In 2D image compression (JPEG 2000), wavelets routinely achieve 10–100× compression with excellent quality. The hypothesis is that extending this to 3D+1D yields similar or better gains for dynamic scene representation.

### Connection to Existing Work

This idea is not entirely without precedent — but it has not been pursued in its full form:

- **Octree representations** (PlenOctrees, etc.) are implicitly performing a Haar wavelet decomposition. Octree subdivision corresponds to scale levels, and storing a value at a coarse node instead of subdividing is equivalent to zeroing detail coefficients. But they use only the simplest possible wavelet (Haar) without the full machinery of optimal coefficient selection and quantization.
- **Multi-resolution hash grids** (Instant-NGP) have a similar multi-resolution structure, but the hash function destroys the spatial coherence that wavelets exploit, and the learned features at each level are not organized as proper wavelet coefficients.
- **Video codecs** have used 3D wavelet transforms for decades, proving the viability of the temporal extension.

The proposed approach takes the formal wavelet framework seriously — using proper wavelet families (Daubechies, CDF 9/7), principled coefficient thresholding, and rate-distortion optimal bit allocation — applied to a sparse 3D+1D scene representation.

## Representation Architecture

### Spatial Structure: Sparse Multi-Resolution Wavelet Coefficient Volume

Rather than a dense 3D grid (which is impossibly large at fine resolution), the representation uses a **sparse multi-resolution voxel structure** where wavelet coefficients exist only near surfaces.

- **Resolution levels:** 6–8 levels, from ~16³ (coarsest) up to 2048³–4096³ (finest, ~1mm voxels for a room-scale scene).
- **Occupancy:** At the finest level, only ~0.5–2% of voxels are occupied (surface regions). This means ~10–30 million active coefficients at the finest level out of 1–1.5 billion potential voxels.
- **Per-coefficient data:** Each coefficient encodes density and a compact appearance descriptor — approximately 16 bytes per coefficient (1 float density, 3 floats or quantized color/feature).
- **Data structure:** Hash map from (level, x, y, z) to coefficient values, following the Instant-NGP pattern for GPU-efficient scattered access.

**Static spatial memory:** ~1–2 GB for a room-scale scene.

### Temporal Structure: 1D Wavelet Along Time

For each spatial coefficient, a 1D wavelet transform is applied along the time axis (300 frames for a 10-second clip). This is where the representation achieves its most dramatic compression:

- **Static regions** (the majority of most scenes) produce essentially one nonzero temporal coefficient — the DC/average. The entire temporal history of a static surface point costs no more than a single frame.
- **Moving regions** produce temporal detail coefficients proportional to the complexity of their motion, but smooth motions compress extremely well under wavelet decomposition.

**Estimated temporal memory:** With ~10% of the scene in motion and conservative 10:1 temporal compression: 30M spatial coefficients × 10% × 30 surviving temporal coefficients × 16 bytes ≈ **1.5 GB**.

### Total 4D Representation Size

| Component | Memory |
|---|---|
| Spatial wavelet coefficients (canonical) | 1–2 GB |
| Temporal wavelet coefficients (dynamics) | 1–1.5 GB |
| Index structures and metadata | 0.5–1 GB |
| **Total model** | **3–5 GB** |

This fits comfortably on a high-end GPU (e.g., NVIDIA RTX 5090 with 32 GB VRAM), leaving 27+ GB for rendering buffers, training state, and intermediate computation.

## Rendering: Volume Rendering with Multi-Resolution Wavelet Evaluation

### Why Volume Rendering Over Splatting

For maximum reconstruction quality, the rendering approach should be **differentiable volume rendering** (ray marching with emission-absorption integration) rather than splatting. The reasons:

- Splatting introduces approximation errors from depth sorting and discrete alpha compositing of primitives. Volume rendering gives exact evaluation of the continuous field defined by the wavelets.
- The multi-resolution wavelet structure enables **coarse-to-fine ray evaluation**: start with low-frequency coefficients, add detail only where rays hit surfaces. This provides inherent level-of-detail without additional machinery.
- Wavelet basis functions have compact, localized support, so evaluating the field at any point requires gathering only the coefficients whose support overlaps that point — a bounded, efficient operation.

### Rendering Pipeline

1. **Cast rays** from the virtual camera through each pixel.
2. **March along each ray**, sampling the volume at regular or adaptive intervals.
3. **At each sample point**, gather relevant wavelet coefficients across all resolution levels (spatial locality limits the number of contributing coefficients).
4. **Reconstruct density and appearance** via wavelet synthesis (weighted sum of basis functions evaluated at the sample point).
5. **Integrate** using standard emission-absorption quadrature to produce the final pixel color.

### Computational Feasibility on a 5090

At ~105 TFLOPS (FP32) and ~1.8 TB/s memory bandwidth:

- The primary bottleneck is the scattered memory access pattern during coefficient gathering. This is the same pattern as Instant-NGP's hash grid lookups, which achieves real-time rendering on current hardware — validating feasibility.
- Wavelet evaluation at each sample point involves a small, fixed number of multiply-adds per level (determined by the wavelet filter length), making it arithmetically cheap.
- The coarse-to-fine structure allows early ray termination and adaptive sampling, reducing the number of evaluations in empty or low-detail regions.

## Training: End-to-End Differentiable Optimization

### Optimization Procedure

1. **Initialize** the wavelet coefficient volume (e.g., from a coarse SfM point cloud or random initialization at coarse levels).
2. **Render** training views via differentiable volume rendering.
3. **Compute loss** against ground truth images (L1/L2 + perceptual loss like LPIPS).
4. **Backpropagate** through the rendering and wavelet reconstruction. The wavelet transform is linear and trivially differentiable. Gradients flow directly to the wavelet coefficients.
5. **Prune coefficients** whose magnitude falls below a threshold — this is the natural sparsification mechanism, analogous to Gaussian densification/pruning but with theoretical grounding.
6. **Progressively refine** from coarse to fine levels during training, similar to coarse-to-fine NeRF training schedules.

### Optimizer Memory During Training

Adam optimizer stores two moment estimates per parameter, so training requires ~3× the model size in memory: 5 GB of coefficients → ~15 GB total training state. This leaves ~17 GB on a 32 GB GPU for frame buffers, batch rendering, and gradient computation.

### Principled Bit Allocation

Unlike Gaussian splatting, where the densification heuristic decides where to add capacity, wavelets enable **rate-distortion optimal coefficient selection**. Given a target memory budget:

- Rank all coefficients by their contribution to reconstruction quality (magnitude × basis function energy).
- Retain the top-N coefficients that maximize quality within the budget.
- This is provably optimal for the class of piecewise smooth signals and provides a formal guarantee that the representation is spending its parameters where they matter most.

## Expected Advantages Over Gaussian Splatting

### Quality

- **Sharper edges and fine detail:** Wavelets with proper vanishing moments (e.g., CDF 9/7) represent edges efficiently. Gaussians, being inherently smooth, require many overlapping primitives to approximate sharp features.
- **Fewer floater artifacts:** Floaters in Gaussian splatting arise from primitives attempting to compensate for limited resolution. Wavelet coefficient pruning removes low-energy coefficients cleanly, without the geometric ambiguity of free-floating Gaussians.
- **Better parameter efficiency:** The provable rate-distortion optimality of wavelet thresholding means that for a given memory budget, the wavelet representation should extract more visual quality than any heuristic allocation of Gaussian primitives.

### Compression

- **Temporal compression is principled:** Instead of learning a deformation MLP (whose capacity must be set by hand and whose generalization is unpredictable), the temporal wavelet transform compresses motion with well-understood, tunable rate-distortion behavior.
- **Graceful quality scaling:** Adjusting the coefficient threshold smoothly trades quality for memory, unlike Gaussian splatting where reducing primitive count causes abrupt quality degradation.

### Theoretical Foundation

- **Formal compression guarantees** replace heuristic densification/pruning.
- **Multi-resolution structure** is intrinsic to the representation, not bolted on.
- **Direct connection to established compression theory** enables borrowing decades of results from signal processing.

## Open Challenges and Research Questions

1. **GPU memory access patterns:** Scattered coefficient lookups across wavelet levels must be made cache-efficient. The Instant-NGP hash grid approach provides a template, but optimal data layout for wavelet structures on GPU is an open engineering problem.

2. **Wavelet family selection:** Haar (equivalent to octrees) is simple but produces blocky artifacts. Higher-order wavelets (Daubechies, CDF) give better compression but have wider support, increasing the cost of point evaluation. The optimal tradeoff for 3D scenes needs empirical investigation.

3. **Handling non-grid geometries:** Standard wavelets assume a regular grid. Scenes with complex topology may benefit from second-generation wavelets (lifting scheme) that can be defined on irregular meshes or point sets.

4. **Negative values in wavelet reconstruction:** Wavelet basis functions have negative lobes, but density and opacity must be non-negative. A nonlinear activation (sigmoid, softplus) after reconstruction resolves this but slightly complicates the optimization landscape.

5. **View-dependent appearance:** The current design stores compact appearance descriptors per coefficient. Handling strong view-dependent effects (specularities) may require augmenting coefficients with directional features or coupling the wavelet volume with a small appearance-decoding network.

6. **Training convergence speed:** Coarse-to-fine wavelet optimization should converge faster than flat representations (the coarse levels provide a good initialization for fine levels), but this needs to be validated empirically against Gaussian splatting's typically fast convergence.

## Summary

The proposed representation replaces the unstructured set of Gaussian primitives in 4D Gaussian splatting with a **sparse 4D wavelet coefficient volume** — a multi-resolution decomposition of the scene's density and appearance in space and time. This brings the formal machinery of wavelet compression theory to bear on dynamic scene representation, promising better quality per parameter, principled temporal compression, and graceful quality-memory scaling, while fitting within the 32 GB VRAM budget of a high-end consumer GPU like the NVIDIA RTX 5090.

The rendering pipeline uses differentiable volume rendering with multi-resolution wavelet evaluation, leveraging the localized support and coarse-to-fine structure of wavelets for efficient ray marching. Training proceeds by end-to-end gradient-based optimization of wavelet coefficients, with magnitude-based pruning providing theoretically grounded sparsification.

The key bet is that the formal compression advantages of wavelets — provable rate-distortion optimality, natural multi-resolution structure, and decades of engineering knowledge from signal processing — translate into measurable quality and efficiency gains over the heuristic representations currently dominant in neural scene reconstruction.