# Frequency-Matched Multi-Resolution Wavelet Scene Reconstruction

## A Residual Network Architecture for 3D/4D Scene Representation

---

## 1. The Core Insight

Current approaches to multi-resolution 3D scene optimization — including coarse-to-fine voxel training and progressive Gaussian densification — suffer from a fundamental asymmetry: the training supervision (full-resolution images) contains information at all spatial frequencies simultaneously, while the representation at coarse stages can only express low frequencies. This mismatch forces the optimizer to make compromises. The coarse representation tries to approximate high-frequency detail it cannot faithfully capture, producing systematic errors that become frozen biases when finer levels are added on top.

The insight is to eliminate this mismatch entirely. Instead of training a coarse volume against high-resolution images and hoping it focuses on low-frequency structure, we train it against low-resolution images where high-frequency structure literally does not exist in the supervision. Each resolution level of the volume is trained against images downsampled to the corresponding spatial frequency band. The volume at each level converges to the genuinely optimal representation for the information available — not a compromised approximation of information it cannot represent.

This is not merely a curriculum learning trick. It establishes an exact correspondence between the frequency content of the 2D supervision and the 3D spatial frequency band each wavelet level represents. Doubling the image resolution reveals one octave of spatial frequency in image space, which maps — through the camera geometry — to one octave of spatial frequency in 3D, which corresponds precisely to one level of the wavelet pyramid in the volume representation.

## 2. The Frequency Correspondence

### Image Resolution to 3D Spatial Frequency

The relationship between image pixel size and 3D spatial resolution is governed by the camera geometry. For a pinhole camera at distance $d$ from a surface, with focal length $f$ (in pixels), a single pixel subtends an angular extent of $1/f$ radians, corresponding to a surface patch of size approximately $d/f$ meters.

For a typical benchmark setup — cameras at roughly 2 meters from the scene, focal length around 1000 pixels (standard for 2K capture):

| Image resolution | Pixel size (m) at surface | Equivalent 3D frequency | Matched volume resolution (0.5m scene) |
|---|---|---|---|
| 512 × 512 | ~8 mm | ~125 cycles/m | ~64³ |
| 1024 × 1024 | ~4 mm | ~250 cycles/m | ~128³ |
| 2048 × 2048 | ~2 mm | ~500 cycles/m | ~256³ |
| 4096 × 4096 | ~1 mm | ~1000 cycles/m | ~512³ |

This mapping is not arbitrary — it follows from the Nyquist sampling theorem applied through perspective projection. A volume at resolution $N^3$ can represent spatial frequencies up to $N/2$ cycles per scene extent. Supervision images at resolution $R$ contain spatial frequencies up to $R/2$ cycles per image extent. The camera projection maps between the two.

When these are matched, each optimization stage is solving a well-posed inverse problem: the information bandwidth of the supervision equals the representational bandwidth of the volume. There is no missing information (under-fitting) and no information the volume cannot capture (over-fitting to unrepresentable frequencies).

### Why This Solves the Coarse Commitment Problem

When a 64³ volume is trained against 512px images, the optimizer finds the best explanation of the low-frequency structure of the scene. There is no high-frequency signal to distort its solution. Surface positions are correct to the precision that the supervision can measure (~8mm). Average colors and broad shading are correct.

When moving to 128³ against 1K images, the new information in the supervision is precisely the frequency band between 512px and 1K — one octave of spatial detail. The new wavelet detail coefficients are optimized to explain this new information, on top of a coarse reconstruction that is already optimal for its band. There is no error to compensate for, only new detail to add.

This is fundamentally different from training a 64³ volume against 4K images (where the volume makes compromises to approximately represent 1mm detail with 8mm voxels) and then trying to refine.

## 3. The Residual Network Interpretation

### Architecture

The structure of this system is not merely analogous to a residual deep network — it is mathematically equivalent to one, with specific architectural properties that arise from the wavelet framework.

Consider the forward pass. The scene is represented as a set of wavelet coefficients at $L$ levels: a base approximation $A_0$ and detail coefficients $D_1, D_2, \ldots, D_L$. The full-resolution volume is reconstructed by wavelet synthesis:

$$V = \text{Synthesis}(A_0, D_1, D_2, \ldots, D_L)$$

For orthogonal wavelets, synthesis is linear, and the contribution of each level is additive in the appropriate basis. The volume at level $\ell$ is:

$$V_\ell = V_{\ell-1} + \text{Upsample}(D_\ell)$$

where $\text{Upsample}(D_\ell)$ places the detail coefficients into the appropriate spatial locations and convolves with the wavelet synthesis filter. This is exactly a residual block: the input is the coarse reconstruction $V_{\ell-1}$, the skip connection passes it through unchanged, and the learned detail $D_\ell$ is added.

### Layer-by-Layer Training

In this framework, each level is trained sequentially against supervision at the matched resolution:

**Level 0 (base):** Train $A_0$ (the coarsest approximation) against the lowest-resolution images. This is the first "layer" of the network. It learns the DC component and very low-frequency structure of the scene — overall geometry, average albedo, broad illumination.

**Level 1 (first detail):** Freeze or slow-update $A_0$. Initialize $D_1 = 0$ (the residual default — "nothing to add"). Train $D_1$ against the next-higher image resolution. The loss measures the difference between the level-1 reconstruction $V_1 = V_0 + \text{Upsample}(D_1)$ rendered from the training viewpoints, and the training images at this resolution. $D_1$ learns the first octave of detail: edge sharpening, fine geometry, texture onset.

**Levels 2 through L:** Repeat. Each level adds one octave of spatial frequency, supervised by the corresponding image resolution.

The training cost at each level is modest: only the detail coefficients at that level are the primary trainable parameters, and these are sparse (most of the scene is already well-explained by coarser levels). Surfaces are already located; only their fine-scale properties need refinement.

### The Depth of the Network

For a pipeline from 512px to 4K images (three doublings), the "network" has three residual blocks plus a base layer — four effective layers. Each layer has a clear, physically interpretable role:

- Layer 0 (64³, 512px): Scene-scale geometry, room layout, average colors
- Layer 1 (128³, 1K): Object-level shape refinement, broad texture
- Layer 2 (256³, 2K): Surface detail, texture patterns, fine geometry
- Layer 3 (512³, 4K): Sub-centimeter detail, sharp edges, material microstructure

### Differences from Standard Neural Networks

Several properties distinguish this from a generic ResNet:

**Fixed architecture from physics.** The number of levels, their resolution, and the filter structure are determined by the wavelet family and the camera geometry, not by hyperparameter search. The architecture is grounded in signal processing theory.

**Interpretable parameters.** Every wavelet coefficient has a specific spatial location, scale, and orientation. There are no opaque learned features — the representation is fully interpretable.

**Provable properties.** The wavelet framework provides rate-distortion guarantees: for piecewise smooth signals, retaining the top-N coefficients by magnitude gives the optimal N-term approximation. No generic neural architecture provides this guarantee.

**Layer-wise optimality.** Because each level is trained against band-limited supervision, each layer's parameters are optimal for their frequency band — not a compromise across all frequencies. This is a property the residual network analogy reveals but that arises from the frequency-matched training, not from the network structure per se.

## 4. Practical Training Pipeline

### Stage-by-Stage Procedure

The implementation leverages existing optimized infrastructure — specifically, Plenoxels-style CUDA ray marching and direct voxel optimization — at each stage, with wavelet analysis applied between stages.

**Preprocessing:** Generate an image pyramid from the training views. If the originals are 4K, produce 2K, 1K, and 512px versions using a proper anti-aliased downsampling filter (Lanczos or similar — this matters, because a bad downsampling filter leaks high-frequency energy into the low-resolution images and defeats the purpose of frequency matching).

**Stage 0 — Base (64³ against 512px images):**
Initialize a 64³ voxel grid. Run Plenoxels optimization against the 512px images until convergence. This should take seconds to a few minutes. The result is a coarse but clean representation of the scene's low-frequency structure.

**Transition 0→1:**
Apply a one-level 3D DWT to the 64³ grid. This produces 64³ worth of approximation coefficients (which encode the 32³ content) and detail coefficients. Threshold the detail coefficients if desired. Perform wavelet synthesis to produce a 128³ grid, with the new (level-1) detail coefficients initialized to zero. This 128³ grid is exactly the coarse reconstruction upsampled — smooth, with no fine detail.

**Stage 1 — First Detail (128³ against 1K images):**
Run Plenoxels optimization on the 128³ grid against 1K images. The optimizer starts from a good initialization (the upsampled coarse result) and needs only to learn the residual detail. Allow all voxels to optimize, but the coarse-level information is already embedded in the initialization, so the optimizer naturally focuses on adding high-frequency corrections. Converges quickly because the bulk of the structure is already correct.

**Transition 1→2:**
Apply DWT to the converged 128³ grid. This decomposes it into the refined coarse content plus level-1 detail. Synthesize to 256³ with zero level-2 detail coefficients.

**Stage 2 — Medium Detail (256³ against 2K images):**
Optimize the 256³ grid against 2K images. Starting from Plenoxels' preferred base resolution, this stage should be fast and produce high-quality results — Plenoxels already works well at 256³.

**Transition 2→3 and Stage 3 — Fine Detail (512³ against 4K images):**
Same pattern. This final stage captures sub-centimeter detail, sharp edges, and high-frequency texture.

### Key Implementation Details

**Occupancy masking across stages.** At each transition, the coarse volume provides an occupancy mask — regions where density is effectively zero need no detail coefficients at finer levels. This keeps memory growth proportional to surface complexity, not volume size. At 512³ with 1-2% surface occupancy, the active voxel count stays manageable (~1-3 million active fine-level coefficients instead of 134 million).

**Anti-aliased image downsampling is critical.** If the low-resolution images contain aliased high-frequency content (from naive subsampling), the coarse volume will try to represent that content and the frequency-matching principle breaks down. Use a proper low-pass filter before downsampling.

**Wavelet family selection.** Haar is simplest (averaging and differencing) and maps cleanly to octree-like structures. CDF 9/7 (used in JPEG 2000) gives better compression and fewer artifacts but has wider support, meaning each coefficient influences a larger region. For the initial implementation, Haar is the pragmatic choice — it's trivially parallelizable on GPU and the Plenoxels grid structure naturally supports it. Higher-order wavelets can be explored once the pipeline is validated.

**Learning rate strategy.** At each stage, the initial learning rate can be relatively high because the optimization target is clear (match the residual between current reconstruction and the new-resolution images). A cosine or exponential decay schedule within each stage is reasonable. If coarse coefficients are kept trainable (recommended, with reduced learning rate), use a 10-100× lower learning rate for them compared to the new detail coefficients.

### Memory Budget Across Stages (Targeting 32 GB VRAM)

| Stage | Volume resolution | Active voxels | Model memory | Training overhead (Adam) | Image batch | Available |
|---|---|---|---|---|---|---|
| 0 | 64³ | 262K | ~10 MB | ~30 MB | 512px: tiny | ~31 GB free |
| 1 | 128³ | ~2M | ~80 MB | ~240 MB | 1K: small | ~31 GB free |
| 2 | 256³ | ~16M | ~640 MB | ~1.9 GB | 2K: moderate | ~29 GB free |
| 3 | 512³ | ~30-50M (sparse) | ~2 GB | ~6 GB | 4K: ~200MB/batch | ~23 GB free |

Comfortably within the 32 GB budget at every stage, with substantial headroom for rendering buffers and gradient computation.

## 5. The Final Representation: Compressed Wavelet Volume

### What You Have After Training

After all stages complete, you have a converged 512³ volume. Apply the full multi-level 3D DWT. The result is a pyramid of wavelet coefficients at levels corresponding to resolutions 64³, 128³, 256³, and 512³. Because each level was trained against frequency-matched supervision, the coefficients at each level are already clean — they represent genuine signal at their frequency band, not noise or compromise artifacts.

### Compression

Threshold coefficients by magnitude. The rate-distortion theory of wavelets guarantees that for piecewise smooth signals (which 3D scenes approximated as density fields are), retaining the $N$ largest coefficients gives the optimal $N$-term approximation. The thresholding curve (PSNR vs. number of retained coefficients) tells you exactly how much you can compress for any target quality.

Expected compression from the raw 512³ grid: 10-50× depending on scene complexity and quality target, reducing the representation from ~2 GB to 40-200 MB.

### Rendering the Compressed Representation

The compressed wavelet volume can be rendered via:

**Wavelet synthesis followed by ray marching.** Reconstruct the full (or partial) 512³ grid from the sparse coefficients, then render as a standard voxel grid. Simple but memory-intensive — requires the full grid in memory.

**Direct ray marching in wavelet space.** At each sample point along a ray, evaluate the wavelet expansion by gathering coefficients whose support overlaps the sample point. More complex but avoids reconstructing the full grid. The localized support of wavelets makes this efficient — only a bounded number of coefficients contribute at any point.

**Adaptive resolution rendering.** For distant surfaces or peripheral regions, evaluate only coarse wavelet levels. For close-up or foveal regions, evaluate all levels. The multi-resolution structure provides built-in level-of-detail at zero additional cost.

## 6. Extension to 4D: Temporal Wavelet Decomposition

### Temporal Frequency Matching

The same principle extends to time. Instead of training all 300 frames at full temporal resolution, decompose the training video temporally:

- Temporal level 0: Every 8th frame (low temporal frequency — slow motions, scene averages)
- Temporal level 1: Every 4th frame (medium motions, rhythmic patterns)
- Temporal level 2: Every 2nd frame (fast motions, transient events)
- Temporal level 3: Every frame (the finest temporal detail)

At each temporal level, train the temporal wavelet coefficients to explain the residual between the current temporal reconstruction and the supervision at that temporal sampling rate. Static regions produce zero temporal detail coefficients at all levels. Slowly moving regions produce coefficients only at the coarsest temporal level. Only fast, complex motions require coefficients at the finest temporal level.

### 4D Memory Estimate

The temporal compression compounds with spatial compression:

| Component | Raw size | After wavelet compression |
|---|---|---|
| Spatial (512³, one frame) | ~2 GB | 40-200 MB |
| Temporal (300 frames, per-coefficient) | ×300 = 600 GB raw | 90% static: ~4 GB; 10% moving at 10:1: ~6 GB |
| **Total 4D representation** | ~600 GB | **~10-15 GB** |

This fits in a single 32 GB GPU. Further quantization of coefficients (16-bit, 8-bit) could reduce this by another 2-4×.

## 7. Toward Learned Cross-Scene Priors

### The Generalization Opportunity

The residual network interpretation opens a path beyond per-scene optimization. If the detail coefficients at each level are not raw learned parameters but the output of a small prediction network, the architecture becomes:

$$D_\ell = f_\ell(V_{\ell-1}, I_\ell)$$

where $f_\ell$ is a learned function (a small convolutional network) that takes the coarse reconstruction $V_{\ell-1}$ and the training images at resolution level $\ell$ as input, and predicts the detail coefficients $D_\ell$.

If this function is shared across scenes and pre-trained on a large dataset:

**At training time on a new scene**, $f_\ell$ can be initialized from the pre-trained weights and fine-tuned, dramatically reducing per-scene optimization time. The pre-trained network has learned a prior over what kind of fine-scale detail is consistent with a given coarse structure — essentially a 3D super-resolution prior.

**At inference time**, given only a few views of a new scene, the network could predict plausible fine detail without any per-scene optimization at the finest levels. The coarse levels (which require less data to constrain) are optimized, and the fine levels are hallucinated by the learned prior. This is a principled approach to few-shot novel view synthesis grounded in wavelet theory.

### Why Wavelets Provide the Right Inductive Bias

A generic neural network predicting voxel values has no structural reason to decompose the problem by frequency. It could learn to mix frequencies arbitrarily across layers, making generalization harder. The wavelet framework forces each "layer" to operate at a specific frequency band, which is a strong inductive bias that matches the actual structure of the problem. Fine-scale detail (texture, edges, surface roughness) is statistically similar across scenes in ways that coarse structure (room layout, object placement) is not. By separating these into distinct layers with distinct learned functions, the architecture can exploit this statistical regularity.

### Connection to Existing Work

This framework unifies several existing threads:

- **Progressive neural networks** (progressive GAN training, progressive NeRF): same coarse-to-fine philosophy, but without wavelet structure or frequency-matched supervision.
- **Laplacian pyramid networks** (LapGAN, Laplacian pyramid reconstruction): operate on the Laplacian pyramid, which is closely related to wavelets but without the orthogonality and rate-distortion properties.
- **Neural radiance field priors** (pixelNeRF, MVSNeRF, etc.): learn cross-scene priors but use generic network architectures without multi-resolution decomposition.
- **Image codec neural networks** (learned JPEG 2000-like codecs): use wavelet structure for 2D image compression, demonstrating that the wavelet inductive bias works in learned systems.

The proposed framework could be seen as the convergence of all of these: a wavelet-structured residual network for 3D/4D scene representation, with frequency-matched training and the possibility of learned cross-scene priors.

## 8. Summary: What Makes This Approach Different

The approach rests on three pillars, each reinforcing the others:

**Frequency-matched supervision.** Each resolution level of the 3D wavelet volume is trained against images downsampled to the corresponding spatial frequency band. This eliminates the coarse-commitment problem that plagues multi-resolution optimization, because each level is trained against supervision that exactly matches its representational capacity.

**Residual wavelet architecture.** The coarse-to-fine structure is mathematically equivalent to a residual deep network, where each layer adds one octave of spatial detail. The wavelet framework provides guarantees (rate-distortion optimality, interpretability, natural compression) that generic network architectures lack.

**Practical compatibility.** The training at each stage is a standard sparse voxel grid optimization (Plenoxels-style), leveraging existing highly optimized CUDA infrastructure. The wavelet analysis and synthesis happen between stages, not during the inner loop. The entire pipeline fits within 32 GB VRAM (NVIDIA RTX 5090 class hardware) at all stages with substantial headroom.

The resulting representation is a sparse set of 3D (or 4D) wavelet coefficients that compresses room-scale dynamic scenes to 10-15 GB (4D) or 40-200 MB (static 3D), with formal guarantees on compression optimality, inherent level-of-detail for rendering, and a natural path toward learned cross-scene priors through the residual network interpretation.
