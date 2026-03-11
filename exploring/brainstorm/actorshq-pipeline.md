# ActorsHQ Development Pipeline
## Dynamic Beta Splatting from Multi-View Studio Capture

**Version:** 1.0
**Date:** March 2026
**Input data:** ActorsHQ dataset (Synthesia) — 160 cameras, 12MP, 25fps, 8-bit JPEG
**Prerequisite:** `optimization-protocol.md` (full production protocol — this document adapts it for available data)

---

## Overview

This is a development pipeline for proving out the Beta Splatting optimization stages using ActorsHQ data before our own capture stage is built. It skips material decomposition (no OLAT/cross-pol data available) and targets interactive visualization via viser + pixel streaming rather than Octane compositing or VR delivery.

### Pipeline Summary

```
ACTORSHQ DATA                PRE-PROCESSING              OPTIMIZATION                 DELIVERY
────────────                 ──────────────              ────────────                 ────────
160 cam @ 25fps          ──► SMPL-X fitting           ──► Stage 1: Geometry (2K)  ──► Viser viewer
12MP JPEG (8-bit)            Semantic masking (SAM 2)     Stage 2: Shape (4K)         Pixel streaming
COLMAP calibration           Color normalization          Stage 3: Appearance (8K)
Foreground masks (provided)  Point triangulation
```

### What We Skip (vs. Production Protocol)

| Production Protocol | This Pipeline | Why |
|---|---|---|
| Phase 1: Calibration | Skip — provided by ActorsHQ | COLMAP export script included |
| Phase 2: Capture | Skip — data already captured | Using existing dataset |
| Phase B: A-pose OLAT | Skip — not captured | No cross-pol/OLAT in ActorsHQ |
| Stage 4: Material decomposition | Skip | No OLAT data; no relighting target |
| Phase 5: Octane compositing | Replace with viser | Development visualization |
| Phase 5: VR streaming | Replace with pixel streaming | Remote access to viser |

### What We Adapt

| Production Protocol | This Pipeline | Reason |
|---|---|---|
| 120 cameras, 60fps | 160 cameras, 25fps | More views, lower temporal resolution |
| 12-bit raw | 8-bit JPEG | ActorsHQ limitation — less dynamic range |
| 26MP full-res | 12MP full-res (4112x3008) | Stage 3 trains at this ceiling |
| Per-camera color correction (CCM) | Histogram-based normalization | No color calibration data provided |
| SMPL-X from PyMAF-X | PyMAF-X multi-view fitting | Provided EasyMocap fits are poor quality |

---

## Phase 1: Data Preparation

### 1.1 Download and Organize

Use the ActorsHQ `download_manager.py` to pull a single actor/sequence at multiple scales:

```bash
# Download one sequence at all three scales
python download_manager.py --sequence Actor01/Sequence1 --scales 1x 2x 4x \
    --types rgbs masks calibration
```

**Expected directory structure:**
```
Actor01/Sequence1/
├── calibration.csv                    # 160 cameras: axis-angle rotation, translation, focal length
├── 1x/rgbs/Cam001..Cam160/            # 4112x3008 JPEG per frame
├── 2x/rgbs/Cam001..Cam160/            # 2056x1504
├── 4x/rgbs/Cam001..Cam160/            # 1028x752
└── 1x/masks/Cam001..Cam160/           # foreground masks (PNG)
```

### 1.2 COLMAP Calibration Export

Use the provided `export_colmap.py` from the ActorsHQ toolbox:

```bash
python export_colmap.py --input calibration.csv --output colmap/
```

**Output:**
- `cameras.txt` — PINHOLE model with pixel-space fx, fy, cx, cy per camera
- `images.txt` — quaternion (w,x,y,z) + translation per camera (world-to-camera)
- `points3D.txt` — empty (populated later in Phase 2.4)

**Validation:** Re-project a few COLMAP sparse points (from Phase 2.4) into 5-10 camera views and overlay on the RGB images. Reprojection error should be < 2 px.

---

## Phase 2: Pre-Processing

### 2.1 SMPL-X Fitting (First)

SMPL-X fitting runs first because it provides the body prior that guides everything downstream — kernel initialization, skinning weights, and canonical-space anchoring.

**Why refit:** The provided EasyMocap fits have flat fingers, ~20kg body shape error, and non-standard global rotation convention. We refit from scratch using multi-view 2D evidence — the same approach we'll use with our own capture rig.

**Tool:** PyMAF-X (per-view initialization) + multi-view SMPLify-X (joint optimization)
**Input:** 4x downscaled footage (1028x752) from 16 selected cameras + foreground masks
**GPU allocation:** 2-4 GPUs

**Procedure:**

1. **Select 16 cameras** with good coverage (front, back, sides, 45° angles, top-down)
2. **Detect 2D keypoints** on all selected views using a whole-body pose estimator (e.g., ViTPose or HRNet-w48 with COCO-WholeBody format: 133 keypoints covering body, hands, face)
3. **Per-view initialization with PyMAF-X:**
   - Run PyMAF-X on each of the 16 selected camera views independently
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
   - 2D joint reprojection across 16 views is the primary signal
   - Silhouette overlap with foreground masks provides shape constraint
   - GMM pose prior prevents implausible poses
5. **Temporal optimization:**
   - Hold shape β fixed after convergence on the first ~10 frames (body shape doesn't change)
   - Optimize pose θ, expression ψ, hand pose per frame
   - Temporal smoothing: penalize large joint velocity/acceleration between consecutive frames
6. **Validate:**
   - Overlay SMPL-X mesh wireframe on RGB images from held-out cameras (not in the 16 used for fitting)
   - Per-joint MPJPE target: < 20mm
   - Silhouette IoU target: > 0.90 on held-out views

**Output per frame:**
- SMPL-X parameters: body pose θ (55×3 axis-angle), shape β (10 dims), expression ψ (10 dims), jaw/hand poses
- Fitted mesh vertices (10,475 × 3)
- Per-vertex skinning weights (from SMPL-X model, fixed)
- Global translation and orientation

### 2.2 Semantic Segmentation

**Tool:** Grounded SAM 2 (Grounding DINO-X + SAM 2)
**Input:** 4x downscaled footage (1028x752)
**GPU allocation:** 1-2 GPUs

Semantic labels are used in Stage 2 for constraint relaxation (hair gets more freedom than skin) and in Stage 3 for appearance priors. Running this after SMPL-X means we can use the fitted body model to validate and refine masks.

**Procedure:**

1. Select ~10 anchor cameras distributed evenly around the hemisphere
2. On frame 0 of each anchor camera, run Grounding DINO with text prompts:
   - `"person"` → full performer mask (should closely match provided ActorsHQ masks)
   - `"face"`, `"skin"`, `"hair"`, `"clothing"`, `"shoes"` → semantic part labels
3. SAM 2 propagates masks temporally through the full video sequence
4. Propagate masks spatially from anchor cameras to neighboring cameras using epipolar consistency
5. **SMPL-X cross-check:** Project SMPL-X body part labels (from the fitted mesh) into each camera view. Use as a consistency check against SAM 2 semantic labels:
   - If SAM 2 says "skin" but SMPL-X projects "clothing" → flag for manual review
   - SMPL-X body part projection provides coarse labels for torso/limbs; SAM 2 provides fine-grained material boundaries
6. Quality check: manually verify masks on 5-10 random frames across different cameras

**Output per frame per camera:**
- `mask_fg.png` — binary foreground mask (use ActorsHQ provided masks as primary; SAM 2 as fallback/refinement)
- `mask_semantic.png` — multi-class semantic labels (skin=1, hair=2, clothing=3, shoes=4)

**Estimated time:** ~1-2 hours for a 100-second sequence across 160 cameras

### 2.3 Color Normalization

ActorsHQ provides no per-camera color calibration data (no ColorChecker, no CCMs). We use histogram-based normalization as a best-effort substitute.

**Procedure:**

1. Select a reference camera (e.g., front-facing, well-exposed)
2. For each other camera, compute a per-channel affine color transform that matches its histogram to the reference camera's histogram (using only foreground-masked pixels to avoid background bias)
3. Apply the per-camera affine transform to all frames
4. Validate: compute mean color difference across overlapping foreground regions in adjacent camera views — target ΔE < 5.0 (relaxed vs. production's ΔE < 2.0 since we have no ground truth)

**Output:** `color_normalization.json` — per-camera per-channel scale and offset

### 2.4 Sparse Point Cloud

**Tool:** COLMAP `point_triangulator`
**Input:** 4x masked footage + known camera poses from calibration

**Procedure:**
1. Extract SIFT features from masked 4x frames (only within foreground mask)
2. Match features across views using spatial matching (exploit known camera positions for efficiency)
3. Triangulate points using known camera extrinsics
4. Filter: remove points outside the capture volume bounding box (1.6m diameter × 2.2m height)

**Output:** Sparse 3D point cloud (typically 50K-200K points) — used as sanity check and to supplement kernel initialization in areas where SMPL-X has gaps (hair, loose clothing)

---

## Phase 3: Optimization

This is the core pipeline. We build on:
- **Universal Beta Splatting** (github.com/RongLiu-Leo/universal-beta-splatting) — rendering primitive
- **GauHuman-style** (github.com/skhu101/GauHuman) LBS anchoring — motion prior

No inverse rendering / material decomposition (Stage 4 from the production protocol is skipped entirely).

### Stage 1: Geometry Foundation

**Input:** 4x footage (1028x752, all 160 views) + SMPL-X parameters + foreground masks
**Goal:** Establish temporally stable body geometry anchored to SMPL-X
**Duration:** ~10K-30K iterations, estimated 2-4 hours

#### Initialization

1. Sample 200K-500K seed points on the SMPL-X mesh surface:
   - Every mesh vertex (10,475 points)
   - Additional points at triangle face centers and edge midpoints
   - Denser sampling on face, hands, and clothing regions (guided by semantic masks from Phase 2.2)
2. For each seed point, create a Beta kernel with:
   - **Position:** On the SMPL-X mesh surface
   - **Orientation:** Aligned to mesh face normal
   - **Scale:** Initial isotropic scale based on local vertex density
   - **Opacity:** 1.0
   - **Beta shape (b):** 0.0 (Gaussian-equivalent, frozen in this stage)
   - **Color:** Initialized from nearest pixel in 4x footage (using known camera projection)
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
| Mask | BCE(rendered_alpha, fg_mask) | Prevent kernels outside performer silhouette | 0.1 |

#### Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 4x (1028x752) | Lowest ActorsHQ scale |
| Views per iteration | 4-8 (randomly sampled from 160) | More views available than production |
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

**Input:** 2x footage (2056x1504) + Stage 1 trained model
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
   - If a kernel labeled "skin" renders onto a pixel masked as "hair" → penalty
   - Prevents label confusion at boundaries
   - Weight: λ_sem = 0.05

4. **Temporal splitting (from UBS paper):**
   - If a kernel's motion residual exceeds a threshold across frames → split temporally
   - Creates frame-range-specific kernels for complex motion

5. **Expanded pruning radius:** Increase Δ_i limit from 15cm to 30cm to allow clothing volume

#### Resolution Schedule Within Stage 2

- Iterations 0-10K: 2x (2056x1504) — establish medium-frequency detail
- Iterations 10K-30K: 2x with random crop augmentation (1024x1024 patches)
- Iterations 30K-50K: Full 2x frames, tighten normal alignment

### Stage 3: Appearance Mastering

**Input:** 1x footage (4112x3008, 12MP) + Stage 2 trained model
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
   - Rendering full 4112x3008 images would exhaust VRAM
   - Randomly sample 512×512 pixel patches from the full-res ground truth
   - Render only the corresponding patch region
   - Each iteration uses patches from 2-4 cameras

4. **Normal alignment tightening:**
   - Increase λ_norm to 0.1 (even without relighting, better normals improve view-dependent appearance)

5. **No densification:** Kernel count is fixed from Stage 2

#### 8-bit JPEG Considerations

ActorsHQ provides 8-bit JPEG only (vs. production protocol's 12-bit raw). This affects Stage 3 specifically:
- **Reduced dynamic range:** Highlights and shadows may be clipped — the photometric loss cannot recover detail that was lost to compression
- **JPEG artifacts:** At 12MP the compression ratio is reasonable, but block artifacts may appear in smooth gradients (skin, fabric). Consider adding a small perceptual loss (LPIPS) alongside L1+SSIM to be less sensitive to compression noise
- **Practical impact:** For development/proving the pipeline, 8-bit is sufficient. The architecture and training procedure are identical — only the quality ceiling changes

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
└── semantic_label (N × 1)       # Skin/hair/clothing/shoes

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
| Semantic colorization | Color kernels by semantic label (skin=peach, hair=brown, clothing=blue, shoes=gray) |
| Kernel visualization | Show kernel centers as points, optionally with orientation/scale gizmos |
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
| PSNR | > 28 dB | Held-out camera views (leave 15-20 cameras out of training) |
| SSIM | > 0.93 | Same held-out views |
| LPIPS | < 0.07 | Same held-out views |
| Temporal consistency | < 0.5 px average flow error | Consecutive frames |
| Geometry accuracy | < 5mm Chamfer distance | Compare to multi-view stereo reconstruction |

**Note:** Targets are relaxed vs. production protocol (PSNR 28 vs 30, SSIM 0.93 vs 0.95, LPIPS 0.07 vs 0.05) because 8-bit JPEG input limits the achievable quality ceiling.

### 5.2 Qualitative Checks

- [ ] No floating / detached kernels visible when orbiting
- [ ] Hands and fingers maintain integrity during motion
- [ ] Hair volume is plausible (not collapsed to skull)
- [ ] Clothing moves independently of body (visible secondary motion)
- [ ] Face detail visible at close range
- [ ] No temporal flickering or popping from held-out angles
- [ ] Smooth playback at 25fps in the viser viewer
- [ ] Novel viewpoints (outside the 160-camera hemisphere) degrade gracefully

### 5.3 Comparison Baselines

To validate that Beta kernels outperform standard Gaussians, train a parallel 3DGS model (same data, same SMPL-X anchoring, same stages) and compare:

| Model | Expected Advantage |
|---|---|
| Standard 3DGS + LBS | Baseline — known to work from GauHuman |
| Beta Splatting + LBS (ours) | Better edge quality (hair, clothing boundaries), fewer kernels for equivalent quality |

---

## Appendix A: Resolution Mapping (ActorsHQ → Production)

The three ActorsHQ scales map to the production protocol's resolution stages:

| Production Protocol | Resolution | ActorsHQ Equivalent | ActorsHQ Resolution |
|---|---|---|---|
| Stage 1: Geometry | 2K (1284x1284) | 4x scale | 1028x752 |
| Stage 2: Shape | 4K (2568x2568) | 2x scale | 2056x1504 |
| Stage 3: Appearance | Full-res (5136x5136) | 1x scale | 4112x3008 |

The aspect ratios differ (production is square, ActorsHQ is 4:3) but this only affects the rasterizer's output dimensions — the optimization procedure is identical.

## Appendix B: Estimated Processing Times

For a **100-second sequence** (2,500 frames) at 25fps from 160 cameras:

| Phase | Step | Est. Time | GPUs Used |
|-------|------|-----------|-----------|
| Pre-processing | SMPL-X refitting | 4-8 hours | 2-4 |
| Pre-processing | Semantic segmentation (SAM 2) | 1-2 hours | 1-2 |
| Pre-processing | Color normalization | 15 min | 1 |
| Pre-processing | Point triangulation | 30 min-1 hour | 1 (CPU-heavy) |
| Optimization | Stage 1: Geometry (4x) | 2-4 hours | 8 |
| Optimization | Stage 2: Shape (2x) | 4-8 hours | 8 |
| Optimization | Stage 3: Appearance (1x) | 3-6 hours | 8 |
| Export | Format conversion | 30 min | 1 |
| **Total** | | **~15-30 hours** | |

## Appendix C: Software Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Rendering primitive | Universal Beta Splatting | github.com/RongLiu-Leo/universal-beta-splatting |
| Body model | SMPL-X | smpl-x.is.tue.mpg.de |
| Body estimation | PyMAF-X + SMPLify-X (multi-view) | Multi-view 2D keypoint + silhouette fitting |
| Segmentation | Grounded SAM 2 | github.com/IDEA-Research/Grounded-SAM-2 |
| Human GS reference | GauHuman | github.com/skhu101/GauHuman |
| Camera calibration | COLMAP (via ActorsHQ export) | colmap.github.io |
| Interactive viewer | viser | github.com/nerfstudio-project/viser |
| Pixel streaming | viser WebSocket / WebRTC | Built-in or custom H.264 via NVENC |
| Training framework | PyTorch 2.x + CUDA 12.x | pytorch.org |
