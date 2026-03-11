# D4DGS Pipeline for PKU-DyMVHumans

Dynamic 4D Gaussian Splatting pipeline for monocular-to-multiview human performance capture.

## Overview

```
PKU-DyMVHumans dataset (55 cameras, 250 frames, 4K)
    |
    v
[1] prep_pku.py           -- Extract frames, convert cameras to LLFF format
    |
    +---> frames_4d/train/frame_XXXX/cam*.jpg
    +---> poses_bounds.npy
    |
    v
[2] fit_smplx.py           -- Per-frame SMPL-X body estimation (SMPLest-X)
    |   Uses --fixed_cam cam01 for consistent coordinate frame
    |   All cameras contribute to shape (betas), one camera defines pose
    |
    +---> smplx_params/frame_XXXX.npz
    |
    v
[3] generate_semantic_masks.py  -- Per-pixel semantic labels (SCHP model)
    |   5 classes: person, skin/face, hair, clothing, shoes
    |
    +---> frames_4d/train/frame_XXXX/cam*_semantic.png
    |
    v
[4] prepare_smplx_for_training.py  -- Camera-space mesh -> world-space init
    |   Depth-scale optimization via multi-camera mask overlap
    |   Vertex color sampling + semantic label assignment
    |
    +---> smplx_init.npz  (10,475 verts, 20,908 faces, 55 joints)
    |
    v
[5] verify_smplx_init.py  -- Project vertices onto cameras, check alignment
    |
    +---> smplx_verify/cam*_overlay.jpg
    |
    v
[6] train_4d_3dgs.py       -- 4D Gaussian Splatting training
    |   Temporal model: position = xyz + velocity * (mu_t - t)
    |   Opacity modulated by temporal Gaussian: exp(-0.5*((t-mu_t)/s_t)^2)
    |   Body-aware losses: surface proximity + LBS velocity regularization
    |
    +---> point_cloud.ply  (4D Gaussians with mu_t, s_t, velocity)
    |
    v
[7] viewer_4d.py           -- Interactive viser viewer with time scrubbing
```

## Scripts

### Stage 1: Data Preparation

**`prep_pku.py`**
Converts PKU-DyMVHumans dataset into the N3V-style format our pipeline expects.
- Parses PKU camera files (w2c extrinsics, K intrinsics, depth bounds)
- Extracts frames from video files via ffmpeg, downsamples 2x (4K -> 1080p)
- Converts cameras to LLFF `poses_bounds.npy` format (OpenCV RDF -> LLFF DRB)
- Holds out camera 0 for test split, rest for train

```bash
python prep_pku.py
```

**`downsample_frames.py`**
Bulk downsamples extracted frames (e.g. fullres -> 2x). Hardcoded paths, edit before use.

### Stage 2: SMPL-X Body Estimation

**`fit_smplx.py`**
Runs SMPLest-X (pre-trained SMPL-X regressor) on per-camera images for each frame.

- Detects person via YOLOv8, runs SMPLest-X on the crop
- Uses `--fixed_cam cam01` to lock the reference camera for pose (root_pose, body_pose, mesh_cam, cam_trans) — prevents coordinate frame jumps when person rotates
- All cameras still contribute to shape (betas) via cross-view median
- Outputs per-frame `.npz` with SMPL-X parameters + camera-space mesh (10,475 vertices)

```bash
cd /workspace/SMPLest-X
python fit_smplx.py \
  -s /workspace/Data/PKU-DyMVHumans/4K_Studios_Show_Single_f16 \
  --fixed_cam cam01 --n_cams 12
```

| Output field | Shape | Description |
|---|---|---|
| root_pose | (3,) | Global orientation (axis-angle) |
| body_pose | (63,) | 21 body joints x 3 |
| lhand_pose / rhand_pose | (45,) | 15 hand joints x 3 |
| jaw_pose | (3,) | Jaw rotation |
| betas | (10,) | Shape coefficients (median across views) |
| expression | (10,) | Face expression coefficients |
| cam_trans | (3,) | Camera-space translation |
| mesh_cam | (10475, 3) | Camera-space vertex positions |

### Stage 3: Semantic Masks

**`generate_semantic_masks.py`**
Generates per-pixel semantic segmentation using human parsing models.

- **SCHP** (recommended): 20-class model, best for clothing segmentation (88.5% accuracy)
- **FASHN**: 18-class model, 10x faster, good for quick iteration (60.3%)
- **Sapiens**: Body-part model, unsuitable for clothing (0.5%)

Output: 5-class label images (0=person/generic, 1=skin/face, 2=hair, 3=clothing, 4=shoes)

```bash
python generate_semantic_masks.py \
  -s $SCENE --model schp --split train
```

### Stage 4: World-Space Initialization

**`prepare_smplx_for_training.py`**
Transforms SMPL-X camera-space mesh into world-space Gaussian initialization.

1. Finds best-view camera by mask overlap
2. Computes crop camera intrinsics from detection bbox
3. Optimizes depth scale (1.0-2.5) via coarse+fine sweep maximizing multi-camera mask overlap
4. Transforms: crop-camera -> physical-camera -> world-space (via LLFF c2w)
5. Samples per-vertex colors from most front-facing cameras
6. Assigns semantic labels via majority vote across camera semantic masks
7. Saves vertices + face centers as initial Gaussian positions (10,475 + 20,908 = 31,383 points)

```bash
python prepare_smplx_for_training.py \
  -s $SCENE --frame_idx 0 --downsample 2
```

Output `smplx_init.npz`:
| Field | Shape | Description |
|---|---|---|
| vertices | (10475, 3) | World-space positions |
| faces | (20908, 3) | Triangle indices |
| skinning_weights | (10475, 55) | LBS weights per joint |
| colors | (10475, 3) | RGB from camera sampling |
| semantic_labels | (10475,) | 0-4 class labels |
| depth_scale | scalar | Optimized depth multiplier |

**`verify_smplx_init.py`**
Projects world-space vertices onto multiple cameras, saves overlay images. Check that mask overlap score >= 0.70.

```bash
python verify_smplx_init.py -s $SCENE
```

### Stage 5: 4D Gaussian Splatting Training

**`train_4d_3dgs.py`**
Main training script. Based on D4DGS (Disentangled 4D Gaussian Splatting) with vanilla 3DGS rasterizer.

**Temporal model:**
- Each Gaussian has: position `xyz`, `velocity` (3D), `mu_t` (temporal center), `s_t` (temporal width)
- At time t: `position(t) = xyz + clamp(velocity) * (mu_t - t)`
- Opacity modulated: `opacity(t) = sigmoid(opacity) * exp(-0.5 * ((t - mu_t) / exp(s_t))^2)`
- Velocity clamped to `max_velocity=0.05` to prevent explosion

**Body-aware losses (when `--use_body_init`):**
- `lambda_surface` (0.01): Pulls body-region Gaussians toward initial mesh surface
- `lambda_lbs` (0.05): Regularizes velocity using LBS skinning weights

**Key training settings:**
- Batch size 12: random (camera, timestamp) pairs per optimizer step
- 20K iterations, SH degree 3
- Gradient clipping (max_norm=1.0)
- Split/clone resets velocity to zero (prevents explosion inheritance)
- Densification: gradient threshold 5e-5, cap 400K Gaussians

```bash
python train_4d_3dgs.py \
  -s $SCENE \
  -m /workspace/output/pku_single_v5 \
  --use_body_init --lambda_surface 0.01 --lambda_lbs 0.05 \
  --max_velocity 0.05 --iterations 20000 --batch_size 12
```

### Stage 6: Viewing

**`viewer_4d.py`**
Interactive viewer using viser + diff-gaussian-rasterization. Renders 4D Gaussians in real-time.

Controls: time slider, play/pause, SH degree override, velocity clamping, temporal threshold.

```bash
python viewer_4d.py /workspace/output/pku_single_v5/point_cloud/iteration_20000/point_cloud.ply \
  --port 8080
```

**`view_smplx_sequence.py`**
Debug viewer for SMPL-X fit sequence. Pure numpy forward kinematics (no SMPLest-X dependency). Uses Procrustes alignment to world space.

```bash
python view_smplx_sequence.py -s $SCENE --port 7090
```

**`view_gaussians.py`**
Static point cloud viewer. Loads PLY, decodes SH to RGB, optional camera frustums and SMPL-X mesh overlay.

### Experimental: MHR Body Model

**`prepare_mhr_for_training.py`** / **`verify_mhr_init.py`**
Alternative to SMPL-X using Meta's MHR (18,439 vertices, 127 joints). Converts SMPL-X mesh to MHR topology via barycentric mapping. Phase 1 only (no direct MHR estimation yet).

## File Layout (Scene Directory)

```
scene_dir/
  videos/                          # Raw camera videos
  cams/                            # PKU camera calibration files
  poses_bounds.npy                 # LLFF cameras (N, 17)
  colmap/sparse_triangulated/      # COLMAP sparse points
  frames_4d/
    train/
      frame_0000/
        cam01.jpg                  # RGB frame
        cam01_mask.png             # Foreground mask
        cam01_semantic.png         # 5-class semantic labels
      frame_0001/
      ...
    test/
      ...
  smplx_params/
    frame_0000.npz                 # Per-frame SMPL-X fit
    frame_0001.npz
    ...
  smplx_debug/                     # Wireframe overlays from fitting
  smplx_init.npz                   # World-space init for training
  smplx_verify/                    # Verification overlays
```

## Training History

| Version | Init | Notes | PSNR |
|---------|------|-------|------|
| v1 | Random (SfM) | Baseline, no body model | 20.54 dB |
| v2 | SMPL-X | Broken semantics, uniform weights | 20.42 dB |
| v3 | SMPL-X | SCHP semantic masks | - |
| v4 | SMPL-X | Velocity clamping, gradient clipping, zero-vel split | 20.44 dB |
| v5 | SMPL-X (fixed_cam) | Locked reference camera for stable coordinate frame | pending |

## Key Lessons

1. **Lock the reference camera** (`--fixed_cam cam01`): Without this, `best_view_idx` changes per frame when the person rotates, causing root_pose coordinate frame jumps that corrupt the entire downstream pipeline.

2. **Velocity must be tightly controlled**: Unclamped velocity + inheritance on split/clone causes Gaussian explosion (splats fly off the body into vertical clouds).

3. **SCHP > FASHN > Sapiens** for clothing segmentation: Sapiens is a body-part model and cannot distinguish clothing from skin.

4. **Depth scale optimization is critical**: The SMPLest-X camera-space mesh has arbitrary depth. Multi-camera mask overlap sweep finds the correct scale to place the mesh in world space.

## Dependencies

- **Training**: diff-gaussian-rasterization, PyTorch, numpy
- **SMPL-X fitting**: SMPLest-X, ultralytics (YOLOv8), torchvision
- **Semantic masks**: SCHP (or FASHN/Sapiens)
- **Viewers**: viser, plyfile
- **Data prep**: ffmpeg, OpenCV, PIL

## Infrastructure

- **RunPod**: A100 80GB, SSH alias `ssh runpod`
- **Local scripts**: `/tmp/dbs4d/` (synced to RunPod via `scp`)
- **RunPod scripts**: `/workspace/gaussian-splatting/`
- **Scene data**: `/workspace/Data/PKU-DyMVHumans/` (ephemeral — back up to network volume!)
