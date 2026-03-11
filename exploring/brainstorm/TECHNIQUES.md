# 3DGS Quality Techniques Tracker

Status key: **ACTIVE** = implemented and enabled | **IMPLEMENTED** = in code but disabled | **TODO** = not yet coded | **SKIPPED** = evaluated and rejected

## Initialization

| Technique | Status | Notes |
|-----------|--------|-------|
| COLMAP sparse triangulation (3.4K pts) | ACTIVE | `colmap_triangulate.py` — pycolmap SIFT + known poses |
| COLMAP dense stereo (287K pts) | ACTIVE | `patch_match_stereo` + `stereo_fusion`, filtered to 287K |
| Depth reinitialization (Mini-Splatting) | TODO | Render depth mid-training, sample ~3.5M points, reinitialize |

## Loss Functions

| Technique | Status | Notes |
|-----------|--------|-------|
| L1 photometric loss | ACTIVE | Weight: `1 - ssim_weight` |
| D-SSIM loss | ACTIVE | Weight: `ssim_weight` (0.2 default) |
| Monocular depth (Pearson correlation) | FAILED (v7) | Depth Anything V2 — part of v7 compound failure (see POSTMORTEM.md). Needs retest in isolation at lower weight |
| COLMAP stereo depth supervision | SKIPPED | Tested in v6 — noisy depth hurt test PSNR (24.43 vs 25.78 dB) |
| Depth distortion loss (2DGS-style) | TODO | `sum w_i * w_j * |z_i - z_j|` — forces surface-like behavior |
| Normal consistency loss (2DGS/GOF) | TODO | Align Gaussian normals with depth-gradient normals |
| LPIPS / perceptual loss | TODO | Small weight (0.01-0.1), add late in training |
| Frequency regularization (FreGS) | TODO | Progressive Fourier-space regularization |
| Opacity regularization | FAILED (v7) | 0.001 weight was too weak to substitute for opacity reset. Needs retest at higher weight or alongside reset |
| Scale regularization | TODO | Penalize very large Gaussians |
| Smoothness loss (DepthRegGS) | TODO | Edge-masked depth smoothness |

## Rendering

| Technique | Status | Notes |
|-----------|--------|-------|
| Anti-aliased rasterization (Mip-Splatting) | IMPLEMENTED | `rasterize_mode="antialiased"` in gsplat. Used in v6/v7, untested in isolation |
| AbsGrad for densification | IMPLEMENTED | `absgrad=True`. Used in v6/v7, untested in isolation |
| Depth rendering (expected depth) | IMPLEMENTED | `render_mode="RGB+ED"`. Infrastructure for depth supervision |

## Densification & Pruning

| Technique | Status | Notes |
|-----------|--------|-------|
| Gradient-based clone/split | ACTIVE | Threshold 0.00008 (aggressive) |
| Opacity pruning | ACTIVE | Threshold 0.002 |
| Large Gaussian pruning | ACTIVE | Prune if scale > 5x scene_extent |
| Opacity reset | ACTIVE | Every 5K steps. Removing it in v7 was catastrophic — non-negotiable for now |
| Blur-split (Mini-Splatting) | TODO | Split oversized Gaussians causing blur |
| GaussianPro propagation | TODO | Patch-matching inspired densification |

## Training Schedule

| Technique | Status | Notes |
|-----------|--------|-------|
| Exponential LR decay (means) | ACTIVE | 0.00016*extent → 1% over training |
| SH degree ramp 0→max | ACTIVE | Ramp over first 1K steps per degree |
| SH degree cap at 2 | UNTESTED | Part of v7 compound failure — needs isolated test |
| Progressive depth weight decay | FAILED (v7) | Exponential decay 0.5→0.01 too aggressive. Needs retest at much lower weight |

## Architecture

| Technique | Status | Notes |
|-----------|--------|-------|
| Standard 3D Gaussians | ACTIVE | Means, scales, quats, opacity, SH coeffs |
| Scaffold-GS (anchor + neural) | TODO | View-adaptive MLP prediction, much better for sparse views |
| 2D Gaussian Splatting | TODO | Planar disks instead of ellipsoids |

---

## Version History

| Version | Test PSNR | Gaussians | Key Changes |
|---------|-----------|-----------|-------------|
| v3 | 13.38 dB | 564K | Random init, fixed LLFF coords |
| v4 | 21.73 dB | 157K | Sparse SfM init (3.4K pts) |
| v5 | 25.78 dB | 379K | Dense SfM init (287K pts), full res |
| v6 | 24.43 dB | 374K | + COLMAP depth supervision (HURT quality) |
| v7 | 15.35 dB | 1,185K | + Mono depth, SH=2, no opacity reset, antialiased — REGRESSION, floaters |
