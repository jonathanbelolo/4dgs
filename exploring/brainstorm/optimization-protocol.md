# Capture & Optimization Protocol
## Dynamic Beta Splatting of Human Performances

**Version:** 1.0
**Date:** March 2026
**Prerequisite:** Capture stage built per `capture-stage-specification.md`

---

## Overview

This protocol transforms raw multi-view video from a 120-camera rig into a relightable, dynamic Beta Splatting representation of a human performer, suitable for compositing into CGI environments and streaming to 6DOF VR headsets.

**Critical context:** No public codebase combines Beta Splatting with human body models or inverse rendering for dynamic humans. This protocol describes novel R&D that synthesizes components from multiple existing codebases.

### Pipeline Summary

```
CAPTURE                          PRE-PROCESSING              OPTIMIZATION                    DELIVERY
───────                          ──────────────              ────────────                    ────────
A: Ref spheres (static)      ──► Segmentation (SAM 2)    ──► Stage 1: Geometry (2K)     ──► Export
B: A-pose OLAT (static)          Body estimation (SMPL-X)    Stage 2: Shape (4K)            Octane composite
C: Performance (non-pol)         Point triangulation         Stage 3: Appearance (full-res) VR streaming
   120 cam @ 60fps strobed       Color correction            Stage 4: Material decomp
                                                                (from A-pose OLAT data)
```

---

## Phase 1: Pre-Capture Calibration

### 1.1 Intrinsic Calibration (per camera)

**Frequency:** Once at rig assembly, re-check monthly

1. Display a large ChArUco calibration board (OpenCV `cv2.aruco`) in the capture volume
2. Capture ~50 frames with the board at different positions and orientations
3. Extract intrinsics per camera using OpenCV `calibrateCamera()`:
   - Focal length (fx, fy)
   - Principal point (cx, cy)
   - Distortion coefficients (k1, k2, p1, p2, k3)
4. Validate: re-projection error should be < 0.5 px

**Output:** `intrinsics.json` - per-camera intrinsic parameters

### 1.2 Extrinsic Calibration (camera-to-camera)

**Frequency:** Once at rig assembly, verify before each capture session

1. Place a rigid calibration structure with known fiducial markers (ChArUco or AprilTag grid) in the center of the capture volume
2. All 120 cameras capture simultaneously (single PTP-synced frame)
3. Compute extrinsics using one of:
   - **COLMAP `point_triangulator`**: Provide known intrinsics, let COLMAP find feature matches and triangulate. Fast since camera positions are known approximately from rig geometry.
   - **Custom calibration**: Detect fiducial markers in all views, solve PnP per camera, then bundle adjust.
4. Refine with COLMAP's bundle adjuster holding intrinsics fixed

**Output:** COLMAP-format files:
- `cameras.txt` (intrinsics per camera, PINHOLE model)
- `images.txt` (extrinsics: quaternion + translation per camera)
- `points3D.txt` (sparse calibration points - can be empty for training)

### 1.3 Color Calibration

1. Place X-Rite ColorChecker Classic in capture volume
2. Capture under strobed lighting (same strobe parameters as production)
3. For each camera, compute a 3x3 color correction matrix (CCM) that maps observed ColorChecker patches to their known sRGB values
4. Apply per-camera CCM during ingest to ensure consistent color across all 120 views
5. Validate: ΔE2000 < 2.0 across all patches on all cameras after correction

**Output:** `color_correction.json` - per-camera 3x3 CCM + white balance gains

---

## Phase 2: Capture Session

### 2.1 Recording Parameters

| Parameter | Value |
|-----------|-------|
| Frame rate | 60 fps |
| Bit depth | 12-bit |
| Shutter (exposure) | 500 µs (1/2000s) |
| Strobe pulse | 500 µs, synchronized to exposure window |
| Strobe overdrive | 10x SafeSense (Gardasoft) |
| Sync | IEEE 1588 PTPv2 (< 1 µs across all cameras) |
| Polarization | Non-polarized for dynamic capture (Bank B); cross-polarized for A-pose reference (Bank A) |

### 2.2 Capture Sequence

The capture session has three phases, performed in order without moving cameras or lighting.

#### Phase A: Reference Spheres (no performer, ~1 minute)

1. Place calibration spheres in the center of the capture volume:
   - **Matte gray sphere:** Validates diffuse lighting uniformity across all views
   - **Chrome sphere:** Captures environment light map for inverse rendering validation
2. Capture a few seconds under cross-polarized strobes
3. Capture a few seconds under non-polarized strobes (remove CPL filters or rotate to pass-through)
4. Remove spheres from volume

#### Phase B: A-Pose Material Reference (performer static, ~2-3 minutes)

The performer stands in an A-pose (arms at ~45°, legs slightly apart, facing forward) and holds still. This static reference is critical for material decomposition — it provides ground-truth diffuse/specular separation that cannot be captured during dynamic performance.

**Step 1: Cross-polarized capture (~2 seconds)**
- Linear polarizing film on all LEDs, CPL filters on all cameras
- Fire strobes for ~120 frames (2s at 60fps) — provides temporal averaging to reduce noise
- Output: clean diffuse-only reference of the full body from all 120 views

**Step 2: Non-polarized capture (~2 seconds)**
- Remove CPL filters from cameras (or rotate to pass-through)
- Same uniform lighting, same pose
- Fire strobes for ~120 frames
- Output: full appearance (diffuse + specular) of the full body from all 120 views
- **Specular isolation:** `I_specular = I_non_polarized - I_cross_polarized`

**Step 3: OLAT — One Light At A Time (~10-15 seconds)**
- Re-engage CPL filters on cameras
- Gardasoft controllers fire each LED fixture individually in sequence
- With ~150-200 LED fixtures, each firing for 1 frame at 60fps: ~2.5-3.3 seconds per pass
- Capture 2-3 OLAT passes (cross-polarized and non-polarized) for redundancy
- Each OLAT frame records the performer's appearance under a single known light direction from all 120 camera views simultaneously
- **Output:** Near-complete light transport matrix of the performer's body
  - 150-200 light positions × 120 camera views = 18,000-24,000 unique light-view pairs
  - This massively over-constrains the inverse rendering problem in Stage 4
  - Enables per-kernel BRDF fitting (roughness, metallic, anisotropy) with high confidence
  - Provides ground-truth self-shadowing data (which body parts shadow which under each light)

**Why this works:** Material properties (albedo, roughness, metallic) are intrinsic to the surface — they don't change with pose. Skin remains skin, leather remains leather, sequins remain sequins whether the arm is raised or lowered. The A-pose captures these properties cleanly with no motion blur, no occlusion ambiguity, and no temporal interpolation needed. Material attributes assigned to Beta kernels on the canonical A-pose mesh travel naturally with those kernels through LBS deformation during the entire dynamic performance.

#### Phase C: Dynamic Performance (performer moving, full takes)

- **Polarization:** Non-polarized (Bank B fires, unpolarized LEDs; CPL on cameras acts as ~1 stop ND only)
- Full specular appearance is captured — hair highlights, skin sheen, fabric glints, metallic reflections all present in training data
- Material decomposition is anchored from the A-pose OLAT reference (Phase B) — no need for cross-polarization during performance
- Performer executes the full performance (singing, dancing, etc.)
- All 120 cameras capture at 60fps with synchronized strobes
- Multiple takes as needed
- **Bank switch:** Operator switches from Bank A to Bank B via Gardasoft software between Phase B and C — no physical changes, no crew touches cameras

### 2.3 Lighting Configuration Summary

CPL filters remain permanently mounted on all 120 cameras throughout the entire session. Polarization mode is controlled by switching between LED banks via Gardasoft software.

| Capture Phase | LED Bank | Camera CPL | What It Captures |
|--------------|----------|-----------|-----------------|
| A: Ref spheres (cross-pol) | Bank A (polarized) | On (permanent) | Diffuse lighting validation |
| A: Ref spheres (non-pol) | Bank B (unpolarized) | On (permanent, ~1 stop ND) | Environment map reference |
| B: A-pose (cross-pol) | Bank A (polarized) | On (permanent) | Diffuse-only body appearance |
| B: A-pose (non-pol) | Bank B (unpolarized) | On (permanent, ~1 stop ND) | Full appearance (diffuse + specular) |
| B: A-pose OLAT | Bank A individual lights | On (permanent) | Per-light transport (BRDF/BCSDF data) |
| C: Dynamic performance | Bank B (unpolarized) | On (permanent, ~1 stop ND) | **Full appearance** (specular included) |

**Zero manual intervention:** All bank switching is electronic via the Gardasoft Ethernet interface. No crew member touches cameras or lights between phases. The performer transitions from A-pose to performance without waiting.

### 2.3 Data Flow During Capture

```
120 cameras (100GigE fiber)
    │
    ▼
3x Spectrum-4 switches (aggregation)
    │
    ▼
Dark fiber link (400GigE)
    │
    ▼
DGX B300 (ConnectX NICs, RDMA)
    │
    ▼
GPU VRAM (via GPUDirect)
    │
    ├──► Full-res write (26MP, 12-bit) ──► NVMe array
    ├──► 4K downscale ──► NVMe array
    └──► 2K downscale ──► NVMe array
```

**Data rates:**
- Per frame: 26MP × 1.5 bytes (12-bit) × 120 cameras ≈ 5.6 GB
- Per second: ~336 GB/s
- Per minute: ~20 TB
- 5-minute take: ~100 TB (raw, before compression)

---

## Phase 3: Pre-Processing

All pre-processing runs on the DGX B300. Steps 1-2 can be parallelized across GPUs.

### 3.1 Segmentation

**Tool:** Grounded SAM 2 (Grounding DINO-X + SAM 2)
**Input:** 2K downscaled footage
**GPU allocation:** 1-2 B300 GPUs

**Procedure:**
1. Select ~10 "anchor" cameras distributed evenly around the hemisphere
2. On frame 0 of each anchor camera, run Grounding DINO with text prompts:
   - `"person"` → full performer mask (foreground/background separation)
   - `"face"`, `"skin"`, `"hair"`, `"clothing"`, `"shoes"` → semantic part labels
3. SAM 2 propagates masks temporally through the full video sequence using memory attention
4. Propagate masks spatially from anchor cameras to neighboring cameras using epipolar consistency
5. Quality check: manually verify masks on 5-10 random frames across different cameras

**Output per frame per camera:**
- `mask_fg.png` - binary foreground mask (performer = 1, background = 0)
- `mask_semantic.png` - multi-class semantic labels (skin=1, hair=2, clothing=3, shoes=4)

**Estimated time:** ~2-4 hours for a 5-minute take across 120 cameras

### 3.2 Body Model Estimation

**Tool:** PyMAF-X (or SMPLify-X with multi-view extension)
**Body model:** SMPL-X (10,475 vertices, 54 joints, expression + hand parameters)
**Input:** 2K masked footage from selected cameras
**GPU allocation:** 2-4 B300 GPUs

**Procedure:**
1. Select 8-16 cameras with good coverage (front, back, sides, 45° angles)
2. Run PyMAF-X on each selected camera view independently to get initial per-view SMPL-X estimates
3. Multi-view fusion: optimize a single set of SMPL-X parameters per frame that minimizes reprojection error across all selected views
4. Temporal smoothing: apply joint angle velocity constraints to prevent frame-to-frame jitter

**Output per frame:**
- SMPL-X parameters: body pose θ (55×3 axis-angle), shape β (10 dims), expression ψ (10 dims), jaw/hand poses
- Fitted mesh vertices (10,475 × 3)
- Per-vertex skinning weights (from SMPL-X model, fixed)

**Note on future body models:**
- **SKEL** (SIGGRAPH Asia 2023, skel.is.tue.mpg.de): Biomechanically accurate skeleton with scapula sliding and radius/ulna rotation. Superior joint behavior for dancers. No Gaussian splatting integration exists yet. Plan to migrate when integration is built.
- **SUPR** (ECCV 2022): Same topology as SMPL-X (10,475 verts) but sparse deformations and 75 joints. Could be drop-in replacement. Not widely adopted in GS community.

### 3.3 Sparse Point Cloud (Optional)

**Tool:** COLMAP `point_triangulator`
**Input:** 2K masked footage + known camera poses from calibration

**Procedure:**
1. Extract SIFT features from masked 2K frames (only within foreground mask)
2. Match features across views using exhaustive or spatial matching
3. Triangulate points using known camera extrinsics
4. Filter: remove any points outside the capture volume bounding box

**Output:** Sparse 3D point cloud (typically 50K-200K points)
- Used as sanity check for calibration accuracy
- Can supplement kernel initialization in areas where SMPL-X mesh has gaps (hair, loose clothing)

### 3.4 Color Correction

Apply per-camera CCM (from calibration) to all footage at all resolutions. Ensures consistent color appearance across all 120 views before optimization begins.

---

## Phase 4: Optimization

This is the core R&D work. We build a novel pipeline combining:
- **Universal Beta Splatting** (github.com/RongLiu-Leo/universal-beta-splatting) - rendering primitive
- **GauHuman-style** (github.com/skhu101/GauHuman) LBS anchoring - motion prior
- **GS-IR** (github.com/lzhnb/GS-IR) inverse rendering - material decomposition

**GPU allocation:** All 8 B300 GPUs (distributed data parallel)

### Stage 1: Geometry Foundation

**Input:** 2K footage (all 120 views) + SMPL-X parameters + foreground masks
**Goal:** Establish temporally stable body geometry anchored to SMPL-X
**Duration:** ~10K-30K iterations, estimated 2-4 hours

#### Initialization

1. Sample 200K-500K seed points on the SMPL-X mesh surface:
   - Every mesh vertex (10,475 points)
   - Additional points at triangle face centers and edge midpoints
   - Denser sampling on face, hands, and clothing regions (guided by semantic masks)
2. For each seed point, create a Beta kernel with:
   - **Position:** On the SMPL-X mesh surface
   - **Orientation:** Aligned to mesh face normal
   - **Scale:** Initial isotropic scale based on local vertex density
   - **Opacity:** 1.0
   - **Beta shape (b):** 0.0 (Gaussian-equivalent, frozen in this stage)
   - **Color:** Initialized from nearest pixel in 2K footage (using known camera projection)
   - **LBS weights:** Copied from nearest SMPL-X vertex

#### Position Parameterization

Each kernel's world-space position is computed as:

```
Position_world(t) = LBS(θ_t, β, w_i) · (v_canonical_i + Δ_i)
```

Where:
- `θ_t` = SMPL-X pose parameters at frame t
- `β` = SMPL-X shape parameters (fixed across frames)
- `w_i` = LBS skinning weights for kernel i (fixed, from SMPL-X)
- `v_canonical_i` = canonical-space position of kernel i (on the T-pose mesh)
- `Δ_i` = learnable offset (initialized to zero)

The optimizer learns `Δ_i` (the offset from the mesh surface) while the body model provides the motion.

#### Loss Functions

```
L_total = L_photo + λ_surf · L_surface + λ_arap · L_ARAP + λ_norm · L_normal + λ_mask · L_mask
```

| Loss | Formula | Purpose | λ |
|------|---------|---------|---|
| Photometric | L1 + 0.2 × (1 - SSIM) | Match rendered image to ground truth | 1.0 |
| Surface proximity | ‖Δ_i‖² | Keep kernels near SMPL-X surface | 0.1 |
| ARAP | Σ_neighbors ‖(p_i - p_j) - R_i(p_i⁰ - p_j⁰)‖² | Preserve local rigidity | 0.01 |
| Normal alignment | 1 - (n_kernel · n_mesh) | Align kernel normals to mesh | 0.05 |
| Mask | BCE(rendered_alpha, SAM2_mask) | Prevent kernels outside performer silhouette | 0.1 |

#### Training Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 2K (1284x1284) |
| Views per iteration | 4-8 (randomly sampled from 120) |
| Learning rate (position Δ) | 1.6e-4, exponential decay |
| Learning rate (color SH) | 2.5e-3 |
| Learning rate (opacity) | 5e-2 |
| Densification interval | Every 100 iterations, from iter 500 to 15000 |
| Densification gradient threshold | 2e-4 |
| Opacity reset | Every 3000 iterations |
| Constraint strength | HIGH (λ_surf = 0.1) |
| Beta shape (b) | Frozen at 0.0 |

#### Densification

Follow the standard adaptive density control from 3DGS:
- **Split:** Kernels with high positional gradient and large scale → split into 2 smaller kernels
- **Clone:** Kernels with high positional gradient and small scale → clone with small offset
- **Prune:** Kernels with opacity < 0.005 → remove
- **Additional prune:** Kernels with ‖Δ_i‖ > 0.15m (15cm from mesh) → remove (prevents floaters)

### Stage 2: Shape Refinement

**Input:** 4K footage (2568x2568) + Stage 1 trained model
**Goal:** Refine geometry, activate Beta kernel shapes, handle clothing/hair
**Duration:** ~20K-50K iterations, estimated 4-8 hours

#### Key Changes from Stage 1

1. **Unfreeze Beta shape parameter (b):**
   - b < 0: flat/box-like kernels for skin, flat surfaces
   - b > 0: peaked kernels for fine details, hair tips, fabric edges
   - Learning rate for b: 1e-3

2. **Semantic-aware constraint relaxation:**

   | Semantic label | Surface loss (λ_surf) | Constraint type |
   |---------------|----------------------|----------------|
   | Skin | 0.05 (moderate) | L2 distance to mesh |
   | Clothing | 0.01 (relaxed) | Laplacian smoothness |
   | Hair | 0.005 (minimal) | Laplacian smoothness |
   | Shoes | 0.05 (moderate) | L2 distance to mesh |

3. **Semantic sorting loss:**
   - If a kernel labeled "skin" (from initialization) renders onto a pixel masked as "hair" → penalty
   - Prevents label confusion at boundaries
   - Weight: λ_sem = 0.05

4. **Temporal splitting (from UBS paper):**
   - If a kernel's motion residual exceeds a threshold across frames → split temporally
   - Creates frame-range-specific kernels for complex motion (e.g., flowing fabric that is only visible during a spin)

5. **Expanded pruning radius:** Increase Δ_i limit from 15cm to 30cm to allow clothing volume

#### Resolution Schedule Within Stage 2

- Iterations 0-10K: 4K (2568x2568) - establish medium-frequency detail
- Iterations 10K-30K: 4K with random crop augmentation (1024x1024 patches)
- Iterations 30K-50K: Full 4K frames, tighten normal alignment

### Stage 3: Appearance Mastering

**Input:** Full-resolution footage (5136x5136, 26MP) + Stage 2 trained model
**Goal:** Bake high-frequency color and texture detail from the full-resolution capture
**Duration:** ~10K-20K iterations, estimated 3-6 hours

#### Key Changes from Stage 2

1. **Freeze geometry:**
   - Position offsets (Δ_i): frozen
   - Beta shape (b): frozen
   - Scale: frozen
   - Rotation: frozen

2. **Optimize only:**
   - Spherical Beta color coefficients (view-dependent appearance)
   - Opacity (minor refinement)

3. **Patch-based training:**
   - Rendering full 5136x5136 images would exhaust VRAM
   - Instead: randomly sample 512×512 pixel patches from the full-res ground truth
   - Render only the corresponding patch region
   - Each iteration uses patches from 2-4 cameras

4. **Normal alignment tightening:**
   - Increase λ_norm to 0.1 (prepare kernels for correct specular behavior in Stage 4)

5. **No densification:** Kernel count is fixed from Stage 2

### Stage 4: Inverse Rendering / Material Decomposition

**Input:** Stage 3 model + A-pose reference data (cross-pol, non-pol, OLAT) + dynamic cross-pol footage
**Goal:** Decompose appearance into PBR material attributes for relighting
**Duration:** ~10K-20K iterations, estimated 4-8 hours

This stage adapts the methodology from GS-IR and Relightable 3DGS to Beta kernels. The A-pose OLAT reference capture (Phase 2B) provides the primary material signal; the dynamic cross-polarized footage provides temporal consistency validation.

#### Per-Kernel Material Attributes

Extend each Beta kernel with material attributes. Non-hair and hair kernels use different shading models:

**Non-hair kernels (skin, clothing, shoes, metal):**

| Attribute | Dimensions | Range | Initialization |
|-----------|-----------|-------|---------------|
| Albedo (ρ) | 3 (RGB) | [0, 1] | From Stage 3 color (desaturated) |
| Roughness (r) | 1 | [0, 1] | 0.5 (medium) |
| Metallic (m) | 1 | [0, 1] | 0.0 (dielectric) |
| Normal offset (Δn) | 3 | [-1, 1] | [0, 0, 0] (use kernel normal) |

**Hair kernels (BCSDF model):**

| Attribute | Dimensions | Range | Initialization |
|-----------|-----------|-------|---------------|
| Tangent (t) | 3 | Unit vector | From multi-view hair orientation estimation |
| Absorption (σ_a) | 3 (RGB) | [0, ∞) | From hair color observation |
| Roughness R lobe (α_R) | 1 | [5°, 15°] | 10° |
| Roughness TRT lobe (α_TRT) | 1 | [10°, 25°] | 15° |
| Cuticle tilt | 1 | [1°, 5°] | 3° |
| Transmission exponent | 1 | [2, 10] | 5 |

#### Decomposition Approach

**Step 1: Albedo extraction from A-pose cross-polarized data**

The A-pose cross-polarized capture (Phase 2B Step 1) eliminates specular highlights, giving a clean diffuse-only image of the static performer from all 120 views:

```
I_cross_pol = albedo × diffuse_shading
```

Since the studio lighting is known (uniform, 5600K):
1. Estimate the environment light as a low-frequency spherical function (order 2-3 SH) — trivial since lighting is uniform and captured by the reference sphere
2. For each kernel on the canonical A-pose mesh, compute diffuse shading from the estimated light and kernel normal
3. Solve for albedo: `albedo = I_cross_pol / diffuse_shading`
4. Validate against X-Rite ColorChecker ground truth

**Step 2: Specular isolation from A-pose dual-capture**

Using the A-pose non-polarized capture (Phase 2B Step 2):

```
I_full = albedo × diffuse_shading + specular_term
I_specular = I_full - I_cross_pol
```

Since both captures are of the exact same static pose from the same cameras, the subtraction is pixel-perfect — no motion compensation needed. The isolated specular component constrains roughness:
- Broad, soft specular → high roughness (matte skin, fabric)
- Tight, bright specular → low roughness (wet surfaces, leather, sequins)

**Step 3: Per-kernel BRDF fitting from OLAT data**

The OLAT sequence (Phase 2B Step 3) provides the richest material signal. For each LED fixture `l` and camera `c`:

```
I_OLAT(l, c) = albedo × V(l) × max(0, n · ω_l) × (1/r²) + specular(ω_l, ω_c, roughness, metallic)
```

Where `V(l)` is the binary visibility (self-shadowing) from light `l`.

With 150-200 light positions × 120 camera views = 18,000-24,000 observations per surface point:
1. For each non-hair kernel, collect all OLAT observations where the kernel is visible
2. Fit a microfacet BRDF (GGX model) to the observations:
   - Roughness (α): controls specular lobe width
   - Metallic (m): controls Fresnel behavior (conductor vs dielectric)
3. Self-shadowing data from OLAT validates kernel normals and geometry:
   - If kernel `i` is lit by light `l` but appears dark → it's shadowed by another body part
   - This provides ground-truth occlusion data that constrains the geometry

**Step 3b: Hair-specific BCSDF fitting from OLAT data**

Hair is not a surface — it's a volume of semi-transparent fibers. Standard PBR (GGX microfacet BRDF) produces matte, lifeless hair. Hair-labeled kernels require a specialized scattering model.

**Tangent direction estimation:**

Before fitting the hair scattering model, estimate the local hair flow direction for each hair kernel:
1. From the 120-view A-pose images, extract hair orientation fields using multi-view orientation estimation (prior work: CT2Hair, NeuralHaircut, HAAR)
2. Assign a tangent vector `t_i` (3D unit vector along local hair flow) to each hair-labeled kernel
3. The Beta kernel's anisotropic scale axis should align with the estimated tangent — enforce this as a constraint
4. Store `t_i` as an additional per-kernel attribute

**Hair scattering model (Marschner/Kajiya-Kay hybrid):**

Each hair kernel is shaded with a dual-lobe model evaluated against its tangent direction:

```
R lobe:   highlight_R   = pow(sin(angle(t, H)), 1/α_R) × fresnel          # primary white highlight
TRT lobe: highlight_TRT = pow(sin(angle(t, H) + tilt), 1/α_TRT) × color_a  # secondary colored highlight
TT lobe:  transmission  = pow(cos(angle(t, ω_l)), exp_TT) × melanin_color  # backlit glow/transmission
```

Per-kernel hair parameters (fitted from OLAT):

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `t_i` | Tangent direction (hair flow) | Unit vector |
| `α_R` | Longitudinal roughness (R lobe) | 5°-15° |
| `α_TRT` | Longitudinal roughness (TRT lobe) | 10°-25° |
| `tilt` | Cuticle tilt angle | 2°-4° |
| `σ_a` | Absorption coefficient (melanin) | Depends on hair color |
| `exp_TT` | Transmission falloff exponent | 2-10 |

The 18,000+ OLAT observations massively over-constrain this 6-parameter fit. Hair kernels observed under backlighting (light behind performer relative to camera) are especially valuable for constraining the TT transmission lobe — this is what makes hair glow when backlit.

**Expected quality per viewing distance:**

| Viewing Distance | Quality | Notes |
|-----------------|---------|-------|
| 1-2m (typical VR) | Convincing | Highlights respond correctly to light changes, backlit glow present |
| 30cm-1m (close-up) | Good | Aggregate highlight behavior correct, but no individual strand detail |
| < 30cm (extreme) | Approximate | Looks like a painted volume, not individual strands |

This is a fundamental limitation of splat-based hair — no splat representation models individual strands. For VR at typical viewing distances, the tangent-based scattering model produces convincing results.

**Step 4: Transfer to dynamic frames**

Material properties are intrinsic and pose-invariant. The per-kernel attributes (PBR for non-hair, BCSDF for hair) fitted on the A-pose canonical mesh carry directly to all deformed poses through LBS:
1. Each kernel retains its material attributes across all frames
2. Hair tangent vectors `t_i` deform with LBS rotation (tangent transforms with the bone rotation, not just position)
3. Validate by re-rendering dynamic non-polarized frames with the decomposed materials under known studio lighting — the full-appearance dynamic capture serves as ground truth
4. If residuals exceed threshold on specific kernels during motion → flag for per-frame material refinement (rare, typically only for stretched/compressed fabric or extreme hair deformation)

**Semantic-guided material priors:**

| Semantic label | Roughness prior | Metallic prior | Special |
|---------------|----------------|---------------|---------|
| Skin | 0.4-0.7 | 0.0 | Subsurface scattering flag |
| Hair | N/A | N/A | Uses BCSDF model (see Step 3b), not PBR |
| Clothing (cotton) | 0.7-0.9 | 0.0 | Diffuse dominant |
| Clothing (silk) | 0.2-0.4 | 0.0 | Anisotropic specular |
| Clothing (leather) | 0.3-0.6 | 0.0 | High specular |
| Shoes (leather) | 0.3-0.5 | 0.0 | -- |
| Metal (jewelry) | 0.1-0.3 | 0.8-1.0 | Conductor response |

These priors are soft constraints (regularization), not hard limits. The OLAT data will override them where observations disagree.

#### Loss Functions for Stage 4

```
L_total = L_render + λ_albedo · L_albedo_smooth + λ_mat · L_material_prior + λ_physics · L_energy_conservation + λ_olat · L_OLAT
```

| Loss | Purpose |
|------|---------|
| L_render | Re-render with decomposed materials under known studio lighting; match dynamic non-polarized ground truth (full appearance including specular) |
| L_OLAT | Re-render A-pose under each individual OLAT light; match OLAT observations |
| L_albedo_smooth | Albedo should be smooth within semantic regions (no baked shadows) |
| L_material_prior | Soft constraint from semantic-guided priors table above |
| L_energy_conservation | diffuse + specular ≤ incoming light (physical plausibility) |

---

## Phase 5: Export & Delivery

### 5.1 Export Format

**Per-frame output (or per-keyframe with interpolation):**

```
canonical_model.spz          # Canonical Beta kernels (T-pose)
├── positions (N × 3)        # Canonical space positions
├── scales (N × 3)           # Per-axis scale
├── rotations (N × 4)        # Quaternion orientation
├── beta_shape (N × 1)       # Beta kernel shape parameter (b)
├── opacity (N × 1)          # Alpha
├── sh_coefficients (N × K)  # Spherical Beta color (K coefficients)
├── albedo (N × 3)           # PBR albedo (non-hair kernels)
├── roughness (N × 1)        # PBR roughness (non-hair kernels)
├── metallic (N × 1)         # PBR metallic (non-hair kernels)
├── normal_offset (N × 3)    # Normal perturbation
├── hair_tangent (N × 3)     # Hair flow direction (hair kernels only, zero for non-hair)
├── hair_params (N × 6)      # BCSDF params: α_R, α_TRT, tilt, σ_a(rgb), exp_TT (hair only)
├── skinning_weights (N × J) # LBS weights for J joints
└── semantic_label (N × 1)   # Skin/hair/clothing/shoes

deformation_sequence.bin      # Per-frame deformation data
├── frame_0000/
│   ├── smplx_pose (J × 3)   # Joint angles
│   ├── smplx_shape (10)     # Shape coefficients
│   ├── smplx_expr (10)      # Expression coefficients
│   └── residual_offsets (N × 3)  # Per-kernel residual (if non-zero)
├── frame_0001/
│   └── ...
└── frame_NNNN/
```

### 5.2 Compositing in OctaneRender 2026

Octane 2026.2 natively renders Gaussian splats with full path tracing (confirmed). Beta kernels with b=0 are Gaussian-equivalent and importable. For full Beta kernel support, a custom Octane plugin or converter may be needed.

**Workflow:**
1. Import performer splats as path-traced primitives (PLY or SPZ format)
2. Assign PBR material attributes from the export:
   - Map albedo → Octane diffuse
   - Map roughness → Octane specular roughness
   - Map metallic → Octane metallic/conductor blend
3. Place performer in CGI environment (Octane scene)
4. Octane computes: GI, contact shadows, reflections, SSS (for skin-labeled kernels)
5. Render per-frame or export as pre-lit splat sequence

**Octane features confirmed available:**
- Native Gaussian splat path tracing
- SPZ format import
- Neural Radiance Cache (NRC) for noise reduction
- Splats visible in reflections and refractions
- Splats cast and receive shadows

**Not confirmed / may require custom work:**
- Beta kernel (b ≠ 0) native support
- Direct PBR attribute import per-splat
- Animated splat sequences

### 5.3 VR Delivery Pipeline

**Architecture:** Server-side compositing + primitive streaming to headset

1. **Server (DGX B300):**
   - Runs Octane (or custom renderer) to compute GI, shadows, and diffuse lighting per frame
   - Pre-lights the performer splats based on the virtual environment
   - Compresses deformation deltas using StreamSTGS-style codec (traditional video codecs for temporal coherence)
   - Sends pre-lit splat attributes via Wi-Fi 7 to headset

2. **Headset (Valve Steam Frame):**
   - Receives canonical model at session start (~20-50 MB compressed)
   - Receives per-frame deformation deltas + pre-computed lighting (~1-5 MB/frame)
   - Local GPU applies deformations and rasterizes splats at headset refresh rate
   - Adds local specular glints (simple Spherical Gaussian shader) for immediate head-tracking response
   - ASW / reprojection fills gaps to hit 120-144Hz

**Bandwidth requirement:** At 60fps with ~2 MB/frame average: ~120 MB/s = ~960 Mbps. Within Wi-Fi 7's practical throughput on the 6GHz band.

---

## Phase 6: Quality Validation

### 6.1 Quantitative Metrics

| Metric | Target | Measured On |
|--------|--------|-------------|
| PSNR | > 30 dB | Held-out camera views (leave 10-15 cameras out of training) |
| SSIM | > 0.95 | Same held-out views |
| LPIPS | < 0.05 | Same held-out views |
| Temporal consistency | < 0.5 px average flow error | Consecutive frames |
| Geometry accuracy | < 5mm Chamfer distance | Compare to multi-view stereo reconstruction |

### 6.2 Qualitative Checks

- [ ] No floating / detached kernels visible in VR
- [ ] Hands and fingers maintain integrity during fast movement
- [ ] Hair volume is plausible (not collapsed to skull)
- [ ] Clothing moves independently of body (visible secondary motion)
- [ ] Face/lip-sync detail visible at close range in VR
- [ ] No temporal flickering or popping when viewed from held-out angles
- [ ] Relighting produces plausible results in at least 3 different CGI environments
- [ ] A-pose OLAT BRDF reconstruction matches observed specular highlights within 5% RMSE
- [ ] Cross-polarized albedo matches ColorChecker reference within ΔE2000 < 3.0

### 6.3 VR-Specific Checks

- [ ] No visible artifacts when leaning in to < 1m distance
- [ ] Parallax is correct when moving head laterally
- [ ] No "cardboard cutout" effect from any viewing angle
- [ ] Specular glints track head movement in real-time (< 20ms latency)
- [ ] No motion sickness during 5-minute viewing session

---

## Appendix A: Software Stack

| Component | Tool | Repository / Source | License |
|-----------|------|-------------------|---------|
| Rendering primitive | Universal Beta Splatting | github.com/RongLiu-Leo/universal-beta-splatting | Apache 2.0 |
| Static Beta Splatting | Deformable Beta Splatting | github.com/RongLiu-Leo/beta-splatting | Apache 2.0 |
| Body model | SMPL-X | smpl-x.is.tue.mpg.de | Custom (academic) |
| Body estimation | PyMAF-X | github.com/HongwenZhang/PyMAF-X | Apache 2.0 |
| Segmentation | Grounded SAM 2 | github.com/IDEA-Research/Grounded-SAM-2 | Apache 2.0 |
| Hair orientation | CT2Hair / NeuralHaircut | Multi-view hair strand estimation | Various |
| Inverse rendering ref | GS-IR | github.com/lzhnb/GS-IR | MIT |
| Inverse rendering ref | Relightable 3DGS | github.com/NJU-3DV/Relightable3DGaussian | Custom |
| Human GS reference | GauHuman | github.com/skhu101/GauHuman | Custom |
| Human GS reference | ExAvatar | github.com/mks0601/ExAvatar_RELEASE | MIT |
| Camera calibration | COLMAP | colmap.github.io | BSD |
| Fast SfM (if needed) | GLOMAP | github.com/colmap/glomap | BSD |
| Compositing | OctaneRender 2026.2 | home.otoy.com/octane2026 | Commercial |
| Streaming reference | StreamSTGS | arXiv:2511.06046 | -- |
| Training framework | PyTorch 2.x + CUDA 12.x | pytorch.org | BSD |

## Appendix B: Estimated Processing Times

For a **5-minute performance** (18,000 frames) captured at 60fps from 120 cameras:

| Phase | Step | Est. Time | GPUs Used |
|-------|------|-----------|-----------|
| Pre-processing | Segmentation (SAM 2) | 2-4 hours | 2 |
| Pre-processing | Body estimation (PyMAF-X) | 4-8 hours | 4 |
| Pre-processing | Color correction | 30 min | 1 |
| Pre-processing | Point triangulation | 1-2 hours | 1 (CPU-heavy) |
| Optimization | Stage 1: Geometry (2K) | 2-4 hours | 8 |
| Optimization | Stage 2: Shape (4K) | 4-8 hours | 8 |
| Optimization | Stage 3: Appearance (full-res) | 3-6 hours | 8 |
| Optimization | Stage 4: Materials | 4-8 hours | 8 |
| Export | Format conversion + compression | 1-2 hours | 2 |
| **Total** | | **~20-45 hours** | |

**Note:** These are rough estimates. Actual times depend on kernel count, convergence behavior, and whether per-frame or keyframe-interpolated optimization is used. A DGX B300 with 2.1 TB VRAM and 144 PFLOPS FP4 should significantly outperform these estimates, which are based on extrapolation from RTX 3090/4090 benchmarks.

## Appendix C: Key Research Papers

All papers verified to exist (see `gemini-4d-splatting-research.md` for full verification):

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Deformable Beta Splatting (arXiv:2501.18630) | SIGGRAPH 2025 | Core rendering primitive |
| Universal Beta Splatting (arXiv:2510.03312) | ICLR 2026 | N-D Beta kernels for dynamic scenes |
| GauHuman (CVPR 2024) | CVPR 2024 | LBS-anchored Gaussians on SMPL |
| ExAvatar (ECCV 2024) | ECCV 2024 | SMPL-X with Gaussians, hands+face |
| GS-IR (arXiv:2311.16473) | CVPR 2024 | Inverse rendering for Gaussians |
| Relightable 3DGS (arXiv:2311.16043) | ECCV 2024 | BRDF decomposition + ray tracing |
| Gaussian Grouping (arXiv:2312.00732) | ECCV 2024 | Semantic ID per Gaussian |
| DashGaussian (arXiv:2503.18402) | CVPR 2025 | Resolution scheduling for fast training |
| SplatSuRe (arXiv:2512.02172) | 2025 | Selective super-resolution for GS |
| MeshSplatting (arXiv:2512.06818) | 2025 | Triangle-based alternative |
| Luminance-GS (arXiv:2504.01503) | CVPR 2025 | Exposure normalization |
| SKEL (SIGGRAPH Asia 2023) | TOG 2023 | Biomechanical body model |
| SAMa (arXiv:2411.19322) | 2024 | Material-aware 3D segmentation |
