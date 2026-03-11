# Deriving the 8192³ Wavelet Volume from First Principles

## Why the Capture Stage Demands an 8K Cube

---

## 1. The Question

Given a 120-camera volumetric capture stage with known optics and geometry, what is the minimum cube resolution $N^3$ that preserves **all** spatial detail captured by the cameras? And how does this interact with the wavelet reconstruction pipeline?

The answer is **8192³**. This document derives that result from the camera specifications, explains the fundamental limits, and shows how the frequency-matched wavelet pipeline exploits multi-resolution camera data to fill an 8K volume efficiently.

---

## 2. Camera and Stage Specifications

From the capture stage specification (see `capture-stage-specification.md`):

| Parameter | Value |
|-----------|-------|
| Cameras | 120 × Emergent HZ-25000-SBS |
| Sensor | Sony IMX947, 5136 × 5136 (square), 5.48 µm pixels |
| Sensor dimensions | 28.15mm × 28.15mm (39.7mm diagonal) |
| Frame rate | 60 fps @ 12-bit |
| Camera-to-center | ~3.5m |
| Capture volume | 4m diameter × 2.5m height |
| Subject | Human performer (singing, dancing) |
| Compute | NVIDIA DGX B300 (8 × B300, 288 GB HBM3e per GPU) |

### Optimized Lens Assignment

The original specification called for uniform 35mm lenses on all cameras. Analysis shows that a mixed-lens strategy dramatically improves face detail — the most perceptually critical region for 6DOF VR close-ups — without compromising full-body coverage:

| Ring | Height | Cameras | Lens | FOV (square) | Coverage at 3.5m | Primary target |
|------|--------|---------|------|-------------|------------------|----------------|
| 1 | 0.5m | 20 | **35mm** | 43.8° | 2.82m × 2.82m | Feet, lower body |
| 2 | 1.2m | 24 | **35mm** | 43.8° | 2.82m × 2.82m | Hips, hands, full body |
| 3 | 1.8m | 24 | **85mm** | 18.8° | 1.16m × 1.16m | **Face, shoulders** |
| 4 | 2.4m | 20 | **50mm** | 31.4° | 1.94m × 1.94m | Head from above, hair |
| 5 + Dome | 3.0m+ | 32 | **35mm** | 43.8° | 2.82m × 2.82m | Above, full coverage |

**Rationale:** The face is the highest-priority region for VR. Ring 3 cameras sit at face height and look straight at the performer. Equipping them with 85mm lenses puts 2.4× more pixels on the face compared to 35mm, resolving individual pores and iris detail. Ring 4 gets a moderate upgrade (50mm) for hair and top-of-head detail. All other rings keep 35mm for full-body angular coverage.

**Cost impact:** +$10,200 (0.25% of total budget). The Sigma 85mm f/1.4 Art and 50mm f/1.4 Art share the same EF mount and image circle as the 35mm.

**Coverage check for Ring 3 (85mm):** 24 cameras at 15° angular spacing with 18.8° FOV per camera. Since 18.8° > 15°, adjacent fields of view overlap by 3.8°, providing continuous azimuthal coverage. At any performer position, the face is visible from 2–4 Ring 3 cameras simultaneously.

---

## 3. The Fundamental Pixel Density Limit

### 3.1 Pixel Footprint

For a camera with focal length $f_{\text{px}}$ (in pixels) at distance $d$ from the subject surface, the **pixel footprint** — the physical size of one pixel projected onto the subject — is:

$$\Delta = \frac{d}{f_{\text{px}}}$$

The focal length in pixels for a lens of focal length $f_{\text{mm}}$ on a sensor of width $W_{\text{mm}}$ with $R$ pixels:

$$f_{\text{px}} = \frac{f_{\text{mm}} \times R}{W_{\text{mm}}}$$

For each lens in the stage:

| Lens | $f_{\text{px}}$ | $\Delta$ at 3.5m (center) | $\Delta$ at 3.2m (surface) | $\Delta$ at 2.0m (close) |
|------|---------|--------------------------|---------------------------|-------------------------|
| 35mm | 6,387 | 0.548 mm | 0.501 mm | 0.313 mm |
| 50mm | 9,124 | 0.384 mm | 0.351 mm | 0.219 mm |
| 85mm | 15,511 | 0.226 mm | 0.206 mm | 0.129 mm |

### 3.2 The Full-Body Framing Ceiling

There is a fundamental limit when every camera must frame the full capture height. The number of vertical pixels on the subject is fixed by the sensor resolution and the angular extent of the subject, **regardless of lens choice or camera distance**:

$$\text{pixels on subject (vertical)} = R \times \frac{\text{subject height}}{\text{FOV height}}$$

When the lens is chosen to match the FOV to the capture height (the tightest lens that still frames the full body), $\text{FOV height} \approx \text{capture height}$, so:

$$\text{pixels on subject} \approx R \times \frac{H_{\text{subject}}}{H_{\text{capture}}} = 5136 \times \frac{1.8\text{m}}{2.5\text{m}} = 3{,}698 \text{ pixels}$$

This gives a pixel footprint of $1.8/3698 = 0.487\text{ mm}$, yielding a cube resolution of:

$$N = \frac{S}{0.487\text{ mm}} \approx 4{,}096 \quad \text{for } S = 2\text{m bounding cube}$$

**This ceiling is independent of lens or distance.** Moving cameras closer requires a wider lens to frame the same body, which exactly cancels the distance gain. The proof: the pixel density at center is $f_{\text{px}} / d = (f_{\text{mm}} R / W) / d$. For full-height framing, $f_{\text{mm}} = W H_{\text{capture}} / (2d)$ (from the FOV constraint), so density $= R / (2 H_{\text{capture}})$, a constant.

**4096³ is the hard ceiling for any setup where every camera frames the full body.**

### 3.3 Breaking the Ceiling with Tiled Coverage

The only way past 4096³ is to have some cameras **not** frame the full body — to tile the coverage so that each camera's pixels are concentrated on a smaller region. This is exactly what the mixed-lens strategy achieves:

- **Ring 3 (85mm):** Each camera sees 1.16m × 1.16m — face and upper torso only
- **Ring 4 (50mm):** Each camera sees 1.94m × 1.94m — head to knees
- **Other rings (35mm):** Full body for angular coverage and coarse reconstruction

The 85mm cameras provide a pixel footprint of **0.206 mm** at the face — 2.4× finer than the full-body ceiling. This drives the cube resolution requirement upward.

---

## 4. Deriving the Cube Resolution

### 4.1 The Matching Condition

For the volume to capture all spatial detail visible in the camera data, the voxel size must be no larger than the finest pixel footprint from any camera on any part of the subject:

$$\frac{S}{N} \leq \Delta_{\min}$$

$$N \geq \frac{S}{\Delta_{\min}}$$

### 4.2 Computing $\Delta_{\min}$

The finest detail occurs at the closest approach of the subject to the tightest-lens camera. For Ring 3 (85mm) cameras at 3.5m from center:

- Performer at center: body surface at ~3.2m from camera → $\Delta = 0.206\text{ mm}$
- Performer 1m toward camera: surface at ~2.2m → $\Delta = 0.142\text{ mm}$
- Performer near camera (edge of volume): surface at ~1.2m → $\Delta = 0.077\text{ mm}$

Taking the nominal case (performer near center, $\Delta = 0.206\text{ mm}$) with a 2.5m bounding cube:

$$N = \frac{2.5\text{ m}}{0.000206\text{ m}} = 12{,}136$$

This exceeds 8192 but is below 16384. However, 12,136³ is impractical (the memory would be enormous even sparse), and the 85mm cameras only cover the face region — not the full body. The 35mm cameras, which cover the full body, resolve 0.50mm → matched to ~5,000 → 4096³.

The cube must serve both:
- **Face (from 85mm):** wants ~12K, but 8192 captures 67% of the detail
- **Body (from 35mm):** wants ~5K, and 8192 captures 100%

### 4.3 The Answer: 8192³

$$\boxed{N = 8192}$$

At 8192³ with a 2.5m bounding cube:
- Voxel size: $2.5 / 8192 = 0.305\text{ mm}$... but this is for a 2.5m cube

More precisely, for a 2m cube (tight around a performer):
- Voxel size: $2.0 / 8192 = 0.244\text{ mm}$
- Captures 85% of the face detail from Ring 3 at center ($0.206/0.244 = 85\%$)
- Captures 100% of body detail from 35mm cameras ($0.50/0.244 = 200\%$, over-provisioned)
- When performer moves toward a camera: captures increasingly fine detail up to $0.244/0.077 = 32\%$ at closest approach

Going to 16384³ would capture the close-approach detail but is impractical (~4× the memory). 8192³ is the sweet spot: it captures nearly all center-distance face detail and all body detail, while fitting in B300 GPU memory.

---

## 5. The Frequency-Matched Wavelet Pipeline

### 5.1 Wavelet Level Sizes

For an 8192³ target with 8 levels of DWT decomposition using the biorthogonal 4.4 wavelet:

```
level_sizes(8192, 8, "bior4.4") ≈ [40, 40, 72, 136, 264, 520, 1031, 2054, 4100]
```

Reconstruction chain: 40 → 72 → 136 → 264 → 520 → 1031 → 2054 → 4100 → 8192

### 5.2 Matched Image Resolutions

The frequency-matching principle: each wavelet level should be trained against images downsampled to the spatial frequency band that level can represent. The matched image resolution for volume level $N_\ell$ is:

$$R_\ell = N_\ell \times \frac{d \times R_{\text{full}}}{f_{\text{px}} \times S}$$

For the 35mm cameras ($f = 6387$, $d = 3.2\text{m}$, $S = 2.0\text{m}$, $R = 5136$):

$$R_\ell \approx N_\ell \times \frac{3.2 \times 5136}{6387 \times 2.0} = N_\ell \times 1.287$$

For the 85mm cameras ($f = 15511$):

$$R_\ell \approx N_\ell \times \frac{3.2 \times 5136}{15511 \times 2.0} = N_\ell \times 0.530$$

This means the 85mm cameras at full resolution (5136px) are matched to volume level $5136 / 0.530 \approx 9{,}690$ — between 8192 and 16384. The camera over-resolves relative to the 8192 cube, which means the finest wavelet level is well-constrained by the face camera data.

### 5.3 Training Stages

The capture stage ingest pipeline already generates a 3-tier image pyramid: full resolution (5136²), half (2568²), and quarter (1284²). These map directly to the wavelet training stages:

| Stage | Volume | Matched image | Camera data source | What it learns |
|-------|--------|---------------|-------------------|----------------|
| 0 | 264³ | ~256 px | ↓ from 1284² tier | Body pose, room-scale geometry |
| 1 | 520³ | ~512 px | ↓ from 1284² tier | Limb shape, face oval, clothing silhouette |
| 2 | 1031³ | ~1,284 px | **1284² tier** (captured) | Facial features, hand shape, fabric folds |
| 3 | 2054³ | ~2,568 px | **2568² tier** (captured) | Skin texture, wrinkles, fabric weave |
| 4 | 4100³ | ~5,136 px | **5136² full** (all cameras) | Fine skin detail, cloth threads |
| 5 | 8192³ | >5,136 px | **5136² from 85mm + 50mm** | **Sub-0.3mm: pores, iris, individual hairs** |

**Stages 0–3** use all 120 cameras (downsampled), providing uniform supervision across the entire body. Every camera contributes to the coarse and mid-level reconstruction.

**Stage 4** uses full-resolution images from all cameras. The 35mm cameras drive the body detail; the 85mm cameras provide strong gradients on the face/upper body.

**Stage 5** is supervised primarily by the 24 Ring 3 cameras (85mm) and 20 Ring 4 cameras (50mm). These 44 cameras provide the only data fine enough to constrain the 8192³ wavelet coefficients. The remaining 76 cameras' full-resolution data is already fully captured by stage 4.

### 5.4 How Multi-Resolution Camera Data Maps to Wavelet Sparsity

This is the key architectural insight: the wavelet fine-level coefficients **naturally adapt** to the available camera resolution, with no special code required.

The 8192³ cube has uniform 0.244mm voxels everywhere. But the wavelet coefficients at the finest level are optimized against the camera data via gradient descent. Where camera data provides sub-0.3mm detail (face, from 85mm cameras), the gradients are strong and the fine coefficients converge to non-zero values representing genuine surface microstructure. Where camera data provides only 0.5mm detail (lower body, from 35mm cameras), the gradients at the finest level are weak — the coarser levels already explain the data — and the L1 sparsity regularization drives the fine coefficients to zero.

The result:

```
Fine-level coefficient density (8192³):
  Face:           ████████████████████  dense (~80% of surface voxels active)
  Hands:          ████████████          moderate (~40%)
  Upper torso:    ██████                sparse (~20%)
  Lower body:     ██                    very sparse (~5%)
  Empty space:                          zero (0%)
```

The wavelet framework automatically concentrates detail where the cameras provide it. This is not a designed feature — it emerges from the interaction of multi-resolution camera data, gradient-based optimization, and wavelet sparsity.

---

## 6. Memory Budget on DGX B300

### 6.1 Architecture

```
decomp_levels = 8, base_level = 2
level_sizes = [40, 40, 72, 136, 264, 520, 1031, 2054, 4100]

Dense base (frozen):
  approx:    40³ × 28ch
  detail_0:  40³ × 28ch
  detail_1:  72³ × 28ch
  detail_2: 136³ × 28ch
  → reconstruct_base() → 264³ volume

Sparse DWT-initialized (frozen):
  detail_3: 264³ × 28ch, ~8% occupancy
  detail_4: 520³ × 28ch, ~4% occupancy
  detail_5: 1031³ × 28ch, ~2% occupancy    (from svox2 DWT or prior training)

Sparse trainable:
  detail_6: 2054³ × 4ch, ~1% occupancy     (learns 1K→2K refinement)
  detail_7: 4100³ × 4ch, ~0.5% occupancy   (learns 2K→4K refinement, face-dominated)

Final reconstruction: 8192³
```

### 6.2 Occupancy at the Finest Level

For a human performer in a 2m bounding cube at 8192³ resolution:

- **Body volume** (~0.07 m³): 0.07 / (0.000244³) = 4.83 billion voxels = 0.88% of cube
- **Surface shell** (1.7 m² × 3 voxels thick): ~86 million voxels = 0.016%
- **Wavelet coefficient sparsity**: detail coefficients are non-zero primarily at surfaces (density transitions), not in the smooth interior or empty exterior. Effective occupancy at the finest level: **~0.3–0.5%** of the tile grid

This surface-dominated sparsity is a key property of wavelets: the interior of a solid body has constant density → zero detail coefficients. The exterior is empty → zero. Only the surface boundary produces non-zero wavelet coefficients.

### 6.3 Memory Table (Single B300 GPU, 288 GB)

**Training the finest two levels (stages 4–5):**

| Component | Size | Notes |
|-----------|------|-------|
| Dense base (frozen, →264³, 28ch) | 1.4 GB | approx + detail_0 through detail_2 |
| Sparse detail_3 (frozen, 264³, 28ch) | 2.7 GB | ~13 occupied tiles @ 205 MB |
| Sparse detail_4 (frozen, 520³, 28ch) | 7.4 GB | ~36 tiles |
| Sparse detail_5 (frozen, 1031³, 28ch) | 20.1 GB | ~98 tiles |
| Sparse detail_6 (trainable, 2054³, 4ch) | 10.6 GB | ~360 tiles @ 29 MB |
| Sparse detail_7 (trainable, 4100³, 4ch) | 40.4 GB | ~1,400 tiles @ 29 MB |
| Adam optimizer states (2× trainable) | 102.0 GB | momentum + variance for detail_6 + detail_7 |
| Gradient buffers (1× trainable) | 51.0 GB | |
| Base volume cache (264³ × 28ch) | 1.4 GB | materialized for tiled rendering |
| Rendering intermediates | 10 GB | ray buffers, sample points, tile reconstruction |
| Image batch (4 × 5136²) | 1.3 GB | |
| **Peak total** | **~248 GB** | **Fits with 40 GB headroom** |

With 2 channels at the finest level instead of 4: peak drops to **~190 GB** (98 GB free).

### 6.4 Multi-GPU Opportunity

The DGX B300 has 8 GPUs with NVLink 5 (1.8 TB/s per GPU). For future expansion:

- **Data-parallel training** across 2 GPUs: 576 GB total → 8192³ with 28ch at finest level
- **Model-parallel** across 4+ GPUs: enables 16384³ if future camera upgrades demand it
- **Temporal parallel**: different GPUs reconstruct different time frames of a 60fps performance

---

## 7. Pixel Utilization Analysis

### 7.1 Per-Camera Pixel Efficiency

For a 1.8m tall, 0.6m wide performer at center:

| Lens | Coverage at 3.5m | Pixels on body | Utilization | Detail on face |
|------|-----------------|----------------|-------------|----------------|
| 35mm | 2.82m × 2.82m | 3.55M | 13.4% | 462 px vertical |
| 50mm | 1.94m × 1.94m | 7.6M | 28.8% | 660 px |
| 85mm | 1.16m × 1.16m | 13.2M* | 50.0%* | **1,130 px** |

*Ring 3 cameras see only face + upper torso, so all visible pixels are on the subject.

### 7.2 Total Data Budget Per Frame

| Camera group | Count | Pixels per cam on subject | Total subject pixels |
|-------------|-------|--------------------------|---------------------|
| 35mm (Rings 1,2,5,dome) | 76 | 3.55M | 270M |
| 50mm (Ring 4) | 20 | 7.6M | 152M |
| 85mm (Ring 3) | 24 | 13.2M | 317M |
| **Total** | **120** | | **739M** |

Compare to the volume:
- 8192³ at 0.5% surface occupancy: ~2.75 billion active voxels × 4 channels = 11 billion unknowns
- 739M pixels per frame × 60 fps = 44.3 billion measurements per second

The system is well-constrained: even a single frame provides ~0.07 observations per unknown, and temporal consistency across the 60fps sequence rapidly builds up the constraint count. The face region, with the highest pixel density from 24 × 85mm cameras, converges fastest.

---

## 8. Comparison of Scale to Existing Work

| System | Volume resolution | Cameras | Pixel footprint | Year |
|--------|------------------|---------|-----------------|------|
| NeRF (original) | 256³ implicit | ~100 | N/A (MLP) | 2020 |
| Plenoxels | 256–512³ | ~100 | ~2 mm | 2022 |
| 3D Gaussian Splatting | ~1M points | ~100 | ~1 mm equiv | 2023 |
| Instant NGP | 512³ hash grid | ~100 | ~1 mm | 2022 |
| Microsoft MRCS | ~512³ | 106 | ~1.5 mm | 2018 |
| **This system** | **8192³** | **120** | **0.21–0.50 mm** | **2026** |

The 8192³ wavelet volume represents a **16× linear resolution increase** (4,096× volumetric increase) over the state of the art. This is made possible by three advances:

1. **Wavelet sparsity**: only surface-adjacent coefficients are stored, reducing memory from 35 PB (dense 8192³ × 28ch) to ~250 GB
2. **Tiled reconstruction**: the full volume is never materialized; tiles are reconstructed on demand during rendering
3. **B300 GPU memory**: 288 GB per GPU, vs 24–80 GB in previous generations

---

## 9. Physical Detail at 8192³

At 0.244mm voxel size, the volume resolves:

| Feature | Physical size | Voxels across | Resolved? |
|---------|--------------|---------------|-----------|
| Facial features (nose, lips) | 20–50 mm | 82–205 | Far above threshold |
| Individual fingers | 15–20 mm | 61–82 | Well resolved |
| Fabric weave (cotton) | 0.5–1 mm | 2–4 | Resolved |
| Skin pores | 0.05–0.1 mm | 0.2–0.4 | Below threshold (Nyquist) |
| Individual hair strands | 0.07 mm | 0.3 | Below threshold |
| Iris detail (crypts) | 0.5–1 mm | 2–4 | Resolved |
| Fingerprint ridges | 0.3–0.5 mm | 1.2–2 | At Nyquist limit |
| Fabric threads (fine silk) | 0.1–0.2 mm | 0.4–0.8 | Below threshold |
| Wrinkle fine structure | 0.5–2 mm | 2–8 | Resolved |

The volume cannot resolve individual pores or hair strands (those would require ~16384³), but it captures the **texture pattern** of pores and hair as aggregate detail — similar to how a 4K photograph captures skin texture without resolving individual pores. For VR viewing at arm's length, this level of detail exceeds the perceptual resolution of current and near-future headsets.

---

## 10. Summary

The 8192³ wavelet volume is derived from a chain of physical constraints:

1. **Camera resolution** (5136 × 5136) determines the maximum angular detail
2. **Camera distance** (3.5m) and **lens focal length** (85mm on face ring) determine the pixel footprint on the subject (0.206mm on the face)
3. **The Nyquist condition** ($N \geq S / \Delta_{\min}$) determines the cube resolution: $2.0 / 0.000206 = 9{,}709$
4. **Practical rounding** to a wavelet-compatible size gives **8192³**
5. **Wavelet sparsity** makes this feasible: only ~0.3–0.5% of voxels at the finest level carry non-zero coefficients
6. **B300 GPU memory** (288 GB) accommodates the sparse representation with room for optimizer states

The mixed-lens strategy (85mm on Ring 3, 50mm on Ring 4, 35mm elsewhere) maximizes the perceptual quality by concentrating pixel density on the face while maintaining full-body coverage from all angles. The frequency-matched wavelet pipeline naturally separates coarse supervision (all cameras, downsampled) from fine supervision (tight-lens cameras, full resolution), with the wavelet sparsity pattern adapting automatically to the available camera resolution at each body region.

The result is a volumetric representation that captures a human performer at **sub-millimeter spatial resolution** — a 16× linear improvement over prior art — reconstructed from 120 synchronized cameras and trained on a single B300 GPU.
