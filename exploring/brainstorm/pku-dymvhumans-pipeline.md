# PKU-DyMVHumans Development Pipeline
## Dynamic Beta Splatting from Multi-View Studio Capture

**Version:** 1.0
**Date:** March 2026
**Input data:** PKU-DyMVHumans (Peking University, CVPR 2024) — 56-60 cameras, 1080p/4K, 25fps, PNG
**Prerequisite:** `optimization-protocol.md` (full production protocol — this document adapts it for available data)

---

## Overview

This is a development pipeline for proving out the Beta Splatting optimization stages using PKU-DyMVHumans data. This dataset offers immediate access (direct download, no approval forms) and extreme clothing deformation (flowing robes, ribbons, martial arts costumes) that stress-tests the LBS-anchored pipeline harder than everyday clothing would. It skips material decomposition (no OLAT/cross-pol data) and targets interactive visualization via viser + pixel streaming.

### Pipeline Summary

```
PKU-DYMVHUMANS DATA          PRE-PROCESSING              OPTIMIZATION                 DELIVERY
────────────────             ──────────────              ────────────                 ────────
56-60 cam @ 25fps        ──► SMPL-X fitting           ──► Stage 1: Geometry (1080p) ──► Viser viewer
1080p / 4K PNG (8-bit)       Semantic masking (SAM 2)     Stage 2: Shape (4K)          Pixel streaming
COLMAP calibration (ready)   Color normalization          Stage 3: Appearance (4K)
Foreground masks (coarse)    Point cloud (provided)
```

### What We Skip (vs. Production Protocol)

| Production Protocol | This Pipeline | Why |
|---|---|---|
| Phase 1: Calibration | Skip — COLMAP format provided | `cameras.bin`, `images.bin`, `points3D.bin` ready |
| Phase 2: Capture | Skip — data already captured | Using existing dataset |
| Phase B: A-pose OLAT | Skip — not captured | No cross-pol/OLAT in PKU-DyMVHumans |
| Stage 4: Material decomposition | Skip | No OLAT data; no relighting target |
| Phase 5: Octane compositing | Replace with viser | Development visualization |
| Phase 5: VR streaming | Replace with pixel streaming | Remote access to viser |

### What We Adapt

| Production Protocol | This Pipeline | Reason |
|---|---|---|
| 120 cameras, 60fps | 56-60 cameras, 25fps | Fewer views, lower temporal resolution |
| 12-bit raw | 8-bit PNG | Dataset limitation — but no JPEG artifacts |
| 26MP full-res | 4K (3840x2160, ~8MP) | Stage 3 trains at this ceiling |
| Per-camera color correction (CCM) | Histogram-based normalization | No color calibration data provided |
| SMPL-X from PyMAF-X | PyMAF-X multi-view fitting from scratch | No SMPL-X provided |
| Moderate clothing (production target) | Extreme clothing deformation | Robes, ribbons, headwear — harder problem |

### Key Dataset Characteristics

**What makes PKU-DyMVHumans different from our production target:**

| Factor | PKU-DyMVHumans | Our Production Rig |
|---|---|---|
| Angular spacing | ~6° (60 cameras in a circle) | ~3° (120 cameras in a hemisphere) |
| Camera arrangement | **Circle** (single ring, ~6m radius) | Hemisphere (multiple rings) |
| Vertical coverage | Limited — mostly at performer height | Full — floor to overhead |
| Calibration method | COLMAP SfM (feature-based) | Hardware calibration (ChArUco) |
| Clothing complexity | Extreme (far beyond production target) | Moderate (performance costumes) |

The circular camera arrangement means **limited vertical parallax** — the top of the head, shoulders from above, and feet from below are poorly constrained. This is a known limitation for this dataset and won't be an issue with our hemispherical rig.

---

## Phase 1: Data Preparation

### 1.1 Download

Direct download from HuggingFace — no approval process:

```bash
# Clone a specific scenario from Part 1 (pre-formatted)
# Part 1 has 8 scenarios with COLMAP format ready
huggingface-cli download zxyun/PKU-DyMVHumans --include "Part1/scenario_name/*" --local-dir ./data/
```

**Part 1 scenarios** come pre-formatted for multiple frameworks. Use the `data_COLMAP/` subdirectory:

```
scenario_name/
├── data_COLMAP/
│   ├── sparse/
│   │   ├── cameras.bin          # Per-camera intrinsics (PINHOLE or OPENCV model)
│   │   ├── images.bin           # Per-camera extrinsics (quaternion + translation)
│   │   └── points3D.bin         # Sparse SfM point cloud
│   └── images/
│       ├── cam_00/              # Per-camera image sequences (PNG)
│       │   ├── 000000.png
│       │   ├── 000001.png
│       │   └── ...
│       └── cam_59/
├── per_view/
│   └── cam_00..cam_59/
│       └── pha/                 # Foreground masks (BackgroundMattingV2)
└── data_NeuS/                   # Alternative format (not used)
```

**Part 2 scenarios** (37 additional) come as raw multi-view video — require running COLMAP yourself. Start with Part 1.

### 1.2 Resolution Inventory

Unlike ActorsHQ, PKU-DyMVHumans does not provide pre-computed multi-scale versions. We generate our own downscales:

```bash
# Generate 2x and 4x downscales for the staged optimization
python scripts/downscale_images.py --input data_COLMAP/images/ --scales 2 4
```

**Resolution ladder:**

| Scale | 1080p source | 4K source |
|---|---|---|
| 1x (full) | 1920x1080 | 3840x2160 |
| 2x | 960x540 | 1920x1080 |
| 4x | 480x270 | 960x540 |

**Note on 1080p vs 4K:** Some scenarios are captured at 1080p, others at 4K (56 cameras). For 1080p scenarios, the resolution ladder is shallower — Stage 1 trains at 480x270 (very low) and Stage 3 caps at 1920x1080. Prefer 4K scenarios when available.

### 1.3 Calibration Validation

The COLMAP calibration comes from SfM (feature matching), not hardware calibration. This can be less precise than a calibrated rig, so validate before committing to training:

1. Load `cameras.bin` and `images.bin` using COLMAP's Python API or `read_model()`
2. Re-project `points3D.bin` sparse points into 5-10 camera views
3. Compute mean reprojection error — target: < 1.0 px (COLMAP typically achieves this)
4. If error is high, run COLMAP's bundle adjuster with tighter convergence settings
5. Visually inspect: overlay projected sparse points on images, check for systematic drift

---

## Phase 2: Pre-Processing

### 2.1 SMPL-X Fitting (First)

SMPL-X fitting runs first because it provides the body prior that guides everything downstream — kernel initialization, skinning weights, and canonical-space anchoring.

**PKU-DyMVHumans provides no SMPL-X fits** — we fit from scratch using multi-view 2D evidence, identical to the approach we'll use with our own capture rig.

**Tool:** PyMAF-X (per-view initialization) + multi-view SMPLify-X (joint optimization)
**Input:** 4x downscaled footage from 12-16 selected cameras + foreground masks
**GPU allocation:** 2-4 GPUs

**Procedure:**

1. **Select 12-16 cameras** with good angular coverage around the circle. With 60 cameras in a ring, every 4th-5th camera gives ~24-30° spacing between selected views.
2. **Detect 2D keypoints** on all selected views using a whole-body pose estimator (e.g., ViTPose or HRNet-w48 with COCO-WholeBody format: 133 keypoints covering body, hands, face)
3. **Per-view initialization with PyMAF-X:**
   - Run PyMAF-X on each of the 12-16 selected camera views independently
   - Produces initial per-view SMPL-X estimates (pose, shape, expression, hands)
4. **Multi-view fusion (SMPLify-X style):**
   - Initialize from the median of per-view PyMAF-X estimates
   - Optimize a single set of SMPL-X parameters per frame that minimizes:
     ```
     L = λ_reproj · Σ_views Reproj_error(SMPLX_joints, detected_2D_keypoints)
         + λ_mask · Σ_views Silhouette_IoU(rendered_SMPLX, fg_mask)
         + λ_prior · GMM_pose_prior(θ)
         + λ_hand · Hand_prior(θ_hands)
         + λ_shape · ‖β‖²
     ```
   - 2D joint reprojection across 12-16 views is the primary signal
   - Silhouette overlap with foreground masks provides shape constraint
   - GMM pose prior prevents implausible poses
5. **Temporal optimization:**
   - Hold shape β fixed after convergence on the first ~10 frames (body shape doesn't change)
   - Optimize pose θ, expression ψ, hand pose per frame
   - Temporal smoothing: penalize large joint velocity/acceleration between consecutive frames
6. **Validate:**
   - Overlay SMPL-X mesh wireframe on RGB images from held-out cameras (not in the 12-16 used for fitting)
   - Per-joint MPJPE target: < 20mm
   - Silhouette IoU target: > 0.85 on held-out views (relaxed vs. ActorsHQ's 0.90 — loose clothing occludes joints, making 2D keypoint detection harder)

**Challenge — loose clothing:**
PKU-DyMVHumans performers wear flowing robes, wide sleeves, and layered costumes that heavily occlude the body. SMPL-X fits the *body under the clothing*, not the clothing surface. Expect:
- Lower silhouette IoU (the clothing silhouette is much larger than the SMPL-X body mesh)
- More 2D keypoint detection failures (wrists/ankles hidden by sleeves/robes)
- The fitted SMPL-X will look "too thin" relative to the clothed performer — this is correct and expected. The clothing volume is handled by the Beta kernels with relaxed surface constraints in Stage 2.

**Output per frame:**
- SMPL-X parameters: body pose θ (55×3 axis-angle), shape β (10 dims), expression ψ (10 dims), jaw/hand poses
- Fitted mesh vertices (10,475 × 3)
- Per-vertex skinning weights (from SMPL-X model, fixed)
- Global translation and orientation

### 2.2 Semantic Segmentation

**Tool:** Grounded SAM 2 (Grounding DINO-X + SAM 2)
**Input:** 4x downscaled footage
**GPU allocation:** 1-2 GPUs

The provided BackgroundMattingV2 masks are coarse — expect artifacts at hair boundaries, translucent fabrics, and flowing ribbons. We regenerate foreground masks with SAM 2 and add semantic labels.

**Procedure:**

1. Select ~8 anchor cameras distributed evenly around the ring
2. On frame 0 of each anchor camera, run Grounding DINO with text prompts:
   - `"person"` → full performer mask (replaces the coarse provided masks)
   - `"face"`, `"skin"`, `"hair"`, `"clothing"`, `"shoes"` → semantic part labels
   - For PKU-DyMVHumans specifically, consider additional prompts for complex costumes:
     - `"robe"`, `"ribbon"`, `"headdress"`, `"sash"` → helps SAM 2 track loose garment parts
3. SAM 2 propagates masks temporally through the full video sequence
4. Propagate masks spatially from anchor cameras to neighboring cameras using epipolar consistency
5. **SMPL-X cross-check:** Project SMPL-X body part labels into each camera view as a consistency check:
   - Regions where SAM 2 says "clothing" but SMPL-X projects nothing → loose garment extending beyond body (expected and valid)
   - Regions where SAM 2 says "skin" but SMPL-X projects "clothing" → flag for review
6. **Merge with provided masks:** Use the union of SAM 2 and provided BackgroundMattingV2 masks for the foreground binary mask — this catches cases where either method misses part of the performer
7. Quality check: manually verify masks on 5-10 random frames, paying special attention to:
   - Flowing fabric edges (ribbons, wide sleeves, robe hems)
   - Semi-transparent fabrics
   - Fast-moving extremities (spinning, jumping)

**Output per frame per camera:**
- `mask_fg.png` — binary foreground mask (union of SAM 2 + provided masks)
- `mask_semantic.png` — multi-class semantic labels (skin=1, hair=2, clothing=3, shoes=4, loose_garment=5)

**Estimated time:** ~1-2 hours for a 15-second sequence across 60 cameras

### 2.3 Color Normalization

PKU-DyMVHumans provides no per-camera color calibration data. The Z CAM E2 cameras are cinema-grade with reasonable color consistency, but normalization is still needed.

**Procedure:**

1. Select a reference camera (e.g., front-facing, well-exposed)
2. For each other camera, compute a per-channel affine color transform that matches its histogram to the reference camera's histogram (using only foreground-masked pixels)
3. Apply the per-camera affine transform to all frames
4. Validate: compute mean color difference across overlapping foreground regions in adjacent camera views — target ΔE < 5.0

**Output:** `color_normalization.json` — per-camera per-channel scale and offset

### 2.4 Sparse Point Cloud

**Status:** Already provided in `data_COLMAP/sparse/points3D.bin` from COLMAP SfM.

Validate the provided point cloud:
1. Load and visualize in 3D — points should form a rough human shape
2. Filter: remove any points outside the capture volume bounding box (~6m diameter circle)
3. Check density: typically 50K-200K points. If sparse, can supplement with additional SIFT feature extraction within foreground masks

The sparse point cloud supplements kernel initialization in areas where SMPL-X has gaps — critically important for this dataset given the extreme clothing volume (flowing robes, wide sleeves).

---

## Phase 3: Optimization

This is the core pipeline. We build on:
- **Universal Beta Splatting** (github.com/RongLiu-Leo/universal-beta-splatting) — rendering primitive
- **GauHuman-style** (github.com/skhu101/GauHuman) LBS anchoring — motion prior

No inverse rendering / material decomposition (Stage 4 from the production protocol is skipped entirely).

### Stage 1: Geometry Foundation

**Input:** 4x footage (all 56-60 views) + SMPL-X parameters + foreground masks
**Goal:** Establish temporally stable body geometry anchored to SMPL-X
**Duration:** ~10K-30K iterations, estimated 2-4 hours

#### Initialization

1. Sample 200K-500K seed points on the SMPL-X mesh surface:
   - Every mesh vertex (10,475 points)
   - Additional points at triangle face centers and edge midpoints
   - Denser sampling on face, hands, and clothing regions (guided by semantic masks from Phase 2.2)
2. **Supplement with sparse point cloud:** Add points from `points3D.bin` that fall outside the SMPL-X mesh bounding box but inside the foreground mask — these capture loose clothing geometry that SMPL-X cannot represent
3. For each seed point, create a Beta kernel with:
   - **Position:** On the SMPL-X mesh surface (or at sparse point location for supplemental points)
   - **Orientation:** Aligned to mesh face normal (or estimated from local point cloud normals)
   - **Scale:** Initial isotropic scale based on local point density
   - **Opacity:** 1.0
   - **Beta shape (b):** 0.0 (Gaussian-equivalent, frozen in this stage)
   - **Color:** Initialized from nearest pixel in 4x footage (using known camera projection)
   - **LBS weights:** Copied from nearest SMPL-X vertex (supplemental points inherit weights from the closest body part — e.g., a robe point near the arm inherits arm weights)

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
| Mask | BCE(rendered_alpha, fg_mask) | Prevent kernels outside performer silhouette | 0.1 |

#### Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 4x downscale | 480x270 (1080p source) or 960x540 (4K source) |
| Views per iteration | 4-6 (randomly sampled from 56-60) | Fewer views available than ActorsHQ |
| Learning rate (position Δ) | 1.6e-4, exponential decay | |
| Learning rate (color SH) | 2.5e-3 | |
| Learning rate (opacity) | 5e-2 | |
| Densification interval | Every 100 iterations, from iter 500 to 15000 | |
| Densification gradient threshold | 2e-4 | |
| Opacity reset | Every 3000 iterations | |
| Constraint strength | HIGH (λ_surf = 0.1) | |
| Beta shape (b) | Frozen at 0.0 | |

#### Densification

Standard adaptive density control from 3DGS:
- **Split:** Kernels with high positional gradient and large scale → split into 2 smaller kernels
- **Clone:** Kernels with high positional gradient and small scale → clone with small offset
- **Prune:** Kernels with opacity < 0.005 → remove
- **Additional prune:** Kernels with ‖Δ_i‖ > 0.15m (15cm from mesh) → remove (prevents floaters)

### Stage 2: Shape Refinement

**Input:** 2x footage + Stage 1 trained model
**Goal:** Refine geometry, activate Beta kernel shapes, handle clothing/hair
**Duration:** ~20K-50K iterations, estimated 4-8 hours

This is the stage where PKU-DyMVHumans data provides the hardest test. Flowing robes and ribbons can extend 30-50cm from the body surface — far beyond what SMPL-X represents. The semantic-aware constraint relaxation must give clothing kernels enough freedom to reach these volumes.

#### Key Changes from Stage 1

1. **Unfreeze Beta shape parameter (b):**
   - b < 0: flat/box-like kernels for skin, flat surfaces
   - b > 0: peaked kernels for fine details, hair tips, fabric edges
   - Learning rate for b: 1e-3

2. **Semantic-aware constraint relaxation (tuned for extreme clothing):**

   | Semantic label | Surface loss (λ_surf) | Constraint type | Notes |
   |---------------|----------------------|----------------|-------|
   | Skin | 0.05 (moderate) | L2 distance to mesh | Standard |
   | Clothing (tight) | 0.01 (relaxed) | Laplacian smoothness | Standard |
   | Clothing / loose_garment | **0.002 (very relaxed)** | Laplacian smoothness | **Lower than ActorsHQ** — robes need 30-50cm freedom |
   | Hair | 0.005 (minimal) | Laplacian smoothness | Standard |
   | Shoes | 0.05 (moderate) | L2 distance to mesh | Standard |

3. **Semantic sorting loss:**
   - If a kernel labeled "skin" renders onto a pixel masked as "clothing" → penalty
   - Prevents label confusion at boundaries
   - Weight: λ_sem = 0.05

4. **Temporal splitting (from UBS paper):**
   - If a kernel's motion residual exceeds a threshold across frames → split temporally
   - Creates frame-range-specific kernels for complex motion (e.g., a ribbon that extends during a spin and retracts)
   - **Especially important for PKU-DyMVHumans:** flowing garments have complex secondary motion that a single canonical kernel cannot represent

5. **Expanded pruning radius:** Increase Δ_i limit from 15cm to **50cm** to accommodate flowing robes and wide sleeves (vs. 30cm for ActorsHQ)

6. **LBS weight refinement:**
   - Loose garment kernels initially inherit LBS weights from the nearest body part, but this is only approximate
   - Allow small per-kernel adjustments to skinning weights during optimization (learning rate 1e-4, with a regularizer to keep weights sparse and positive)
   - This lets a robe kernel that was initialized near the torso partially inherit leg motion if the robe drapes over the legs

#### Resolution Schedule Within Stage 2

For 4K source (preferred):
- Iterations 0-10K: 2x (1920x1080) — establish medium-frequency detail
- Iterations 10K-30K: 2x with random crop augmentation (1024x1024 patches)
- Iterations 30K-50K: Full 2x frames, tighten normal alignment

For 1080p source:
- Iterations 0-10K: 2x (960x540)
- Iterations 10K-30K: 2x with random crop augmentation (512x512 patches)
- Iterations 30K-50K: Full 2x frames

### Stage 3: Appearance Mastering

**Input:** 1x footage (3840x2160 for 4K, or 1920x1080 for 1080p) + Stage 2 trained model
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
   - For 4K: randomly sample 512×512 pixel patches from the full-res ground truth
   - For 1080p: can render full frames (1920x1080 is manageable in VRAM) — no patch sampling needed
   - Each iteration uses patches/frames from 2-4 cameras

4. **Normal alignment tightening:**
   - Increase λ_norm to 0.1 (even without relighting, better normals improve view-dependent appearance)

5. **No densification:** Kernel count is fixed from Stage 2

#### 8-bit PNG Considerations

PKU-DyMVHumans provides 8-bit PNG (vs. production protocol's 12-bit raw). Unlike ActorsHQ's JPEG, PNG is lossless — no block artifacts or compression noise. The only limitation is reduced dynamic range from 8-bit quantization:
- Highlights and shadows may be clipped
- Smooth gradients in skin and fabric lose subtle tonal variation
- For development/proving the pipeline, 8-bit is sufficient — the quality ceiling is higher than JPEG

---

## Phase 4: Interactive Visualization

### 4.1 Export Format

**Per-sequence output:**

```
canonical_model.ply              # Canonical Beta kernels (T-pose)
├── positions (N × 3)            # Canonical space positions
├── scales (N × 3)               # Per-axis scale
├── rotations (N × 4)            # Quaternion orientation
├── beta_shape (N × 1)           # Beta kernel shape parameter (b)
├── opacity (N × 1)              # Alpha
├── sh_coefficients (N × K)      # Spherical Beta color (K coefficients)
├── skinning_weights (N × J)     # LBS weights for J joints
└── semantic_label (N × 1)       # Skin/hair/clothing/shoes/loose_garment

deformation_sequence.npz         # Per-frame deformation data
├── smplx_poses (T × J × 3)     # Joint angles per frame
├── smplx_shape (10,)            # Shape coefficients (shared)
├── smplx_expr (T × 10)          # Expression coefficients per frame
├── smplx_transl (T × 3)         # Global translation per frame
└── residual_offsets (T × N × 3) # Per-kernel residuals (sparse, if needed)
```

**No material attributes** (albedo, roughness, metallic) — those require OLAT data from the production pipeline.

### 4.2 Viser Interactive Viewer

**Tool:** viser (github.com/nerfstudio-project/viser) — Python-based 3D visualization server with a web UI

The viewer deforms the canonical Beta splats in real-time using SMPL-X LBS on the GPU and rasterizes them via the Beta Splatting CUDA rasterizer.

**Core architecture:**

```python
# Pseudocode — viser app structure

import viser
import torch

server = viser.ViserServer(host="0.0.0.0", port=8080)

# Load canonical model + deformation sequence
canonical = load_canonical_model("canonical_model.ply")
deformations = load_deformations("deformation_sequence.npz")

# --- UI Controls ---

# Playback
frame_slider = server.gui.add_slider("Frame", min=0, max=T-1, step=1, initial_value=0)
play_button = server.gui.add_button("Play/Pause")
fps_slider = server.gui.add_slider("FPS", min=1, max=25, step=1, initial_value=25)

# Visualization toggles
show_mesh = server.gui.add_checkbox("Show SMPL-X mesh", initial_value=False)
show_kernels = server.gui.add_checkbox("Show kernel centers", initial_value=False)
color_by_semantic = server.gui.add_checkbox("Color by semantic label", initial_value=False)
kernel_scale = server.gui.add_slider("Kernel scale", min=0.1, max=3.0, step=0.1, initial_value=1.0)

# Rendering
resolution = server.gui.add_dropdown("Resolution", options=["512", "1024", "2048"], initial_value="1024")


# --- Render Loop ---

@server.on_client_connect
def on_connect(client: viser.ClientHandle):
    # Per-client render loop:
    # 1. Get current frame from slider
    # 2. Apply LBS deformation to canonical kernels
    # 3. Rasterize Beta splats from client's camera viewpoint
    # 4. Send rendered image to client
    pass
```

**Key features to implement:**

| Feature | Description |
|---|---|
| Frame scrubbing | Slider to jump to any frame in the sequence |
| Playback | Play/pause at original 25fps or adjustable speed |
| Free camera | Orbit, pan, zoom around the performer (viser provides this by default) |
| SMPL-X overlay | Toggle wireframe overlay of the fitted body model |
| Semantic colorization | Color kernels by semantic label (skin=peach, hair=brown, clothing=blue, loose_garment=purple) |
| Kernel visualization | Show kernel centers as points, optionally with orientation/scale gizmos |
| Clothing offset heatmap | Color kernels by ‖Δ_i‖ distance from SMPL-X surface — highlights where clothing volume is largest |
| Novel pose | Load arbitrary SMPL-X pose parameters and deform the canonical model (stretch goal) |

### 4.3 Pixel Streaming

For remote access (e.g., from a laptop viewing a model rendered on the DGX), wrap the viser server behind a pixel streaming setup:

**Option A: Native viser (simplest)**

Viser already serves a web UI over HTTP/WebSocket. For LAN or tunnel access:
```bash
# On the DGX
python viewer.py --host 0.0.0.0 --port 8080

# Remote access via SSH tunnel
ssh -L 8080:localhost:8080 user@dgx
# Then open http://localhost:8080 in browser
```

The viser client runs WebGL in the browser. For Beta Splatting rasterization, we need server-side rendering (the CUDA rasterizer runs on the DGX) with the rendered frames streamed to the browser — viser supports this via its scene API and camera callback system.

**Option B: viser + WebRTC (lower latency)**

For interactive frame rates over WAN:
1. Render frames server-side using the Beta Splatting CUDA rasterizer
2. Encode as H.264 via NVENC on the DGX GPU
3. Stream via WebRTC to the browser client
4. Client sends camera pose + UI state back over the WebRTC data channel
5. Target: < 100ms motion-to-photon latency on a good connection

**Bandwidth:** At 1080p H.264, ~5-15 Mbps. Comfortable over Wi-Fi or a decent WAN link.

---

## Phase 5: Quality Validation

### 5.1 Quantitative Metrics

| Metric | Target | Measured On |
|--------|--------|-------------|
| PSNR | > 27 dB | Held-out camera views (leave 8-10 cameras out of training) |
| SSIM | > 0.91 | Same held-out views |
| LPIPS | < 0.08 | Same held-out views |
| Temporal consistency | < 0.7 px average flow error | Consecutive frames |
| Geometry accuracy | < 5mm Chamfer distance | Compare to multi-view stereo reconstruction |

**Note:** Targets are more relaxed than ActorsHQ pipeline (PSNR 27 vs 28, SSIM 0.91 vs 0.93, LPIPS 0.08 vs 0.07) for two reasons:
1. Fewer cameras (56-60 vs 160) means sparser training views and harder novel-view synthesis
2. Extreme clothing deformation is inherently harder to reconstruct — flowing robes viewed from a novel angle are less constrained than tight clothing

Temporal consistency target is relaxed to 0.7 px (vs 0.5 px) because loose garments have genuine high-frequency temporal motion that is hard to interpolate.

### 5.2 Qualitative Checks

- [ ] No floating / detached kernels visible when orbiting
- [ ] Hands and fingers maintain integrity during motion
- [ ] Hair volume is plausible (not collapsed to skull)
- [ ] **Loose clothing volume is preserved** — robes are not shrink-wrapped to body
- [ ] **Flowing fabric has secondary motion** — sleeves and robes continue moving after the body stops
- [ ] **Ribbons and sashes maintain continuity** — no fragmentation into disconnected splat clusters
- [ ] Face detail visible at close range
- [ ] No temporal flickering or popping from held-out angles
- [ ] Smooth playback at 25fps in the viser viewer
- [ ] Novel viewpoints degrade gracefully (noting limited vertical coverage from the circular rig)

### 5.3 Comparison Baselines

To validate that Beta kernels outperform standard Gaussians, train a parallel 3DGS model (same data, same SMPL-X anchoring, same stages) and compare:

| Model | Expected Advantage |
|---|---|
| Standard 3DGS + LBS | Baseline — known to work from GauHuman |
| Beta Splatting + LBS (ours) | Better edge quality (fabric edges, ribbon tips), fewer kernels for equivalent quality |

**PKU-DyMVHumans-specific comparison:** The extreme clothing should amplify the Beta kernel advantage — peaked kernels (b > 0) should better represent thin fabric edges and ribbon tips than Gaussians, which tend to produce blobby edges on thin structures.

---

## Appendix A: Resolution Mapping (PKU-DyMVHumans → Production)

For **4K source** (preferred):

| Production Protocol | Resolution | PKU-DyMVHumans Scale | PKU-DyMVHumans Resolution |
|---|---|---|---|
| Stage 1: Geometry | 2K (1284x1284) | 4x downscale | 960x540 |
| Stage 2: Shape | 4K (2568x2568) | 2x downscale | 1920x1080 |
| Stage 3: Appearance | Full-res (5136x5136) | 1x (native) | 3840x2160 |

For **1080p source:**

| Production Protocol | Resolution | PKU-DyMVHumans Scale | PKU-DyMVHumans Resolution |
|---|---|---|---|
| Stage 1: Geometry | 2K (1284x1284) | 4x downscale | 480x270 |
| Stage 2: Shape | 4K (2568x2568) | 2x downscale | 960x540 |
| Stage 3: Appearance | Full-res (5136x5136) | 1x (native) | 1920x1080 |

1080p source means substantially lower resolution at every stage. The pipeline still works — it just produces a lower quality ceiling. Strongly prefer 4K scenarios.

## Appendix B: Estimated Processing Times

For a **15-second sequence** (~375 frames) at 25fps from 60 cameras:

| Phase | Step | Est. Time | GPUs Used |
|-------|------|-----------|-----------|
| Pre-processing | SMPL-X fitting | 2-4 hours | 2-4 |
| Pre-processing | Semantic segmentation (SAM 2) | 30 min-1 hour | 1-2 |
| Pre-processing | Color normalization | 10 min | 1 |
| Pre-processing | Point cloud validation | 10 min | 1 (CPU) |
| Optimization | Stage 1: Geometry (4x) | 1-2 hours | 8 |
| Optimization | Stage 2: Shape (2x) | 3-6 hours | 8 |
| Optimization | Stage 3: Appearance (1x) | 2-4 hours | 8 |
| Export | Format conversion | 15 min | 1 |
| **Total** | | **~9-18 hours** | |

Shorter than ActorsHQ pipeline (~15-30 hours) because sequences are shorter (~15s vs ~100s) and there are fewer cameras (60 vs 160).

## Appendix C: Software Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Rendering primitive | Universal Beta Splatting | github.com/RongLiu-Leo/universal-beta-splatting |
| Body model | SMPL-X | smpl-x.is.tue.mpg.de |
| Body estimation | PyMAF-X + SMPLify-X (multi-view) | Multi-view 2D keypoint + silhouette fitting |
| Segmentation | Grounded SAM 2 | github.com/IDEA-Research/Grounded-SAM-2 |
| Human GS reference | GauHuman | github.com/skhu101/GauHuman |
| Camera calibration | COLMAP (provided) | colmap.github.io |
| Interactive viewer | viser | github.com/nerfstudio-project/viser |
| Pixel streaming | viser WebSocket / WebRTC | Built-in or custom H.264 via NVENC |
| Training framework | PyTorch 2.x + CUDA 12.x | pytorch.org |

## Appendix D: Recommended Scenarios for Initial Testing

Start with Part 1 scenarios (pre-formatted COLMAP). Prioritize based on pipeline development goals:

| Priority | Scenario Type | Why |
|----------|--------------|-----|
| 1st | Simple motion, tight clothing | Validate basic pipeline end-to-end before tackling hard cases |
| 2nd | Simple motion, loose clothing | Test Stage 2 constraint relaxation and expanded pruning radius |
| 3rd | Fast motion, tight clothing | Test temporal stability and densification under motion blur |
| 4th | Fast motion, loose clothing (kung fu robes, dance ribbons) | Full stress test — the hardest case |

This progression isolates failure modes: if Stage 2 fails on loose clothing, you know it's the constraint tuning, not a temporal issue.
