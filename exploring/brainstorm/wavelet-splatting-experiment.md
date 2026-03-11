# DoG Wavelet Splatting — Experiment Plan

**Date:** March 2026
**Goal:** Test whether DoG detail splats can push PSNR beyond the ceiling that a converged standard 3DGS reaches, using the existing gsplat rasterizer with zero CUDA modifications.

**Hypothesis:** A converged 3DGS model leaves residual error concentrated at sharp edges and thin structures because smooth Gaussians are fundamentally the wrong basis for these features. A DoG detail layer — band-pass primitives with positive center and negative surround — can capture this residual signal that more Gaussians cannot.

---

## 1. Scene Selection

**Dataset:** NeRF Synthetic (already have download script at `src-wavelets/download_data.sh`)

**Scene: `drums`**

Why drums:
- **Baseline PSNR ~26 dB** — plenty of room for improvement. Not so good that there's nothing left to fix, not so bad that the base model is fundamentally broken.
- **Thin structures:** Cymbal edges, drumstick tips, wire stands. Standard Gaussians struggle with these — they're smooth primitives trying to represent sharp, thin geometry. This is exactly where a band-pass DoG primitive should outperform.
- **Clear failure mode to measure:** The residual `I_gt - I_base` will show high error concentrated at cymbal edges and stand wires. We can visualize whether DoG splats fix these specific regions.
- **View-independent appearance:** Drums are mostly diffuse (metal cymbals have some specular, but it's mild). This keeps the experiment clean — we're testing spatial efficiency, not view-dependent effects.

**Control scene: `lego`**

Run the same experiment on lego (~35 dB baseline) as a control. Lego is "easier" — the high baseline means less room for improvement. If DoG splats still show gains on lego, the result is stronger. If they only help on drums, that tells us DoG wavelets specifically help with thin structures (still useful, but narrower claim).

---

## 2. Experiment Design

### 2.1 The Core Question

A converged 3DGS model hits a quality ceiling. More training doesn't help. More splats don't help (v5→v7 history: adding parameters degraded quality). The residual error is structural — smooth Gaussians cannot represent certain features no matter how many you add.

**The test:**
1. Train a base 3DGS to convergence → record PSNR (the ceiling)
2. Freeze the base, add DoG detail splats, train only the detail layer → record new PSNR
3. If PSNR goes up, the DoG primitives captured signal that Gaussians fundamentally could not

No comparison against "more Gaussians" needed — we already know that doesn't work past convergence.

**What to measure:**
- PSNR improvement (any improvement is meaningful — we're past the Gaussian ceiling)
- Where the improvement happens (per-pixel PSNR map — should concentrate at edges and thin structures)
- How many DoG splats it takes to saturate the improvement (diminishing returns curve)

### 2.2 Rendering Without a Custom Rasterizer

**Two-pass rendering, combined in PyTorch:**

```
Pass 1: Render base model normally            → C_base  (H×W×3)
Pass 2: Render detail splats separately       → C_detail (H×W×3, signed)
Combine: C_final = clamp(C_base + C_detail, 0, 1)
```

Each pass uses the standard gsplat rasterizer. No CUDA modifications.

**Making the detail pass additive:**

The gsplat rasterizer does front-to-back alpha compositing. To approximate additive blending:
- Set all detail splat opacities to a small fixed value (logit-space: ~-4.0 → sigmoid ≈ 0.018)
- At this opacity, transmittance stays near 1.0 through the entire splat stack
- Each splat's contribution ≈ α · c ≈ 0.018 · c, nearly independent of depth order
- The accumulated result ≈ Σ (0.018 · cᵢ · ψᵢ(pixel)) — additive

Compensate for the low opacity by scaling up the color values: if the true desired contribution is `d`, set the splat color to `d / 0.018`. The gsplat SH evaluation doesn't clamp internally, so large SH coefficients are fine.

**Even simpler alternative — single-pass rendering:**

Render base + detail splats together in one gsplat call. The detail splats have low opacity (0.018) so they add tiny corrections on top of whatever the base splats render. No two-pass needed.

The advantage of two-pass: cleaner gradient separation (base frozen, detail trained). The advantage of single-pass: simpler code, and the detail splats automatically respect occlusion from base splats.

**Recommendation: Start with two-pass** for the initial experiment (cleaner measurement). Switch to single-pass if results are promising and we want to test joint training.

### 2.3 DoG Splat Representation

Each DoG splat is **two standard Gaussians** stored in a separate `GaussianModel` (the detail model):

```python
class DoGDetailModel:
    """Detail layer: pairs of Gaussians forming DoG wavelets."""

    def __init__(self, device="cuda"):
        self.device = device

    def init_from_residuals(self, positions, normals, detail_colors, scale_narrow):
        """
        Each DoG = narrow Gaussian (positive) + broad Gaussian (negative).
        Both share the same position, orientation, and color magnitude.
        """
        n = positions.shape[0]
        k = 2.0  # scale ratio (one octave)

        # Narrow Gaussian (positive lobe)
        self.means_narrow = positions.clone().detach().requires_grad_(True)
        self.scales_narrow = scale_narrow.clone().detach().requires_grad_(True)
        self.quats_narrow = normals_to_quats(normals).detach().requires_grad_(True)

        # Broad Gaussian (negative lobe) — same position, scaled up
        self.means_broad = positions.clone().detach().requires_grad_(True)
        self.scales_broad = (scale_narrow + math.log(k)).detach().requires_grad_(True)
        self.quats_broad = self.quats_narrow.clone().detach().requires_grad_(True)

        # Fixed low opacity for additive approximation
        self.opacity_value = -4.0  # sigmoid(-4) ≈ 0.018

        # Signed color (the "wavelet coefficient")
        # Scaled up by 1/sigmoid(-4) ≈ 55 to compensate for low opacity
        self.detail_sh0 = detail_colors.clone().detach().requires_grad_(True)

    def get_all_gaussians(self):
        """Return concatenated narrow + broad Gaussians for rasterization."""
        n = self.means_narrow.shape[0]
        means = torch.cat([self.means_narrow, self.means_broad])
        scales = torch.cat([self.scales_narrow, self.scales_broad])
        quats = torch.cat([self.quats_narrow, self.quats_broad])
        opacities = torch.full((2 * n,), self.opacity_value, device=self.device)

        # Narrow gets +color, broad gets -color
        sh0_narrow = self.detail_sh0
        sh0_broad = -self.detail_sh0
        sh0 = torch.cat([sh0_narrow, sh0_broad]).unsqueeze(1)  # (2N, 1, 3)
        shN = torch.zeros(2 * n, 15, 3, device=self.device)

        return means, quats, scales, opacities, sh0, shN
```

**Parameter count per DoG splat:**
- Position: 3 floats (shared center — constrain both to move together)
- Scale narrow: 3 floats (broad scale = narrow + log(k), not independent)
- Quaternion: 4 floats (shared orientation)
- Color (signed): 3 floats
- **Total: 13 floats = 52 bytes per DoG**

A standard Gaussian has: 3 (pos) + 3 (scale) + 4 (quat) + 1 (opacity) + 48 (SH) = 59 floats = 236 bytes.

So a DoG splat is **~4.5× cheaper** than a standard Gaussian in storage. But it renders as 2 Gaussians, so render cost per DoG is ~2× a standard Gaussian.

---

## 3. Step-by-Step Protocol

### Step 0: Download and Prepare Data

```bash
cd /Users/jonathanbelolo/dev/claude/code/4D/src-wavelets
bash download_data.sh  # gets NeRF Synthetic from HuggingFace
```

### Step 1: Train Base 3DGS Model

Train a standard 3DGS on `drums` using the existing `train.py`. Adapt for NeRF Synthetic format (transforms.json instead of LLFF — may need a small data loader addition).

```bash
python train.py --scene_dir ../data/nerf_synthetic/drums \
    --output_dir outputs/drums_base \
    --num_steps 30000 \
    --densify_until 15000 \
    --sh_degree_max 3
```

Train for 30K steps (standard for NeRF Synthetic). Save the converged model. Record baseline test PSNR.

**Also save a snapshot at 15K steps** (before densification ends) — this gives us a "half-trained" base for comparison.

### Step 2: Compute Residuals

Render the base model on all training views. Compute per-pixel residuals:

```python
residuals = []  # list of (H, W, 3) tensors
for i in range(N_train):
    rendered = render(base_model, camtoworlds[i], K[i], W, H)
    residual = gt_images[i] - rendered.clamp(0, 1)
    residuals.append(residual)
```

Visualize the residuals (save as images). They should show:
- High error at cymbal edges
- High error at thin wire stands
- Low error on drum bodies (already well-represented)

### Step 3: Spawn DoG Detail Splats

Place DoG splats where residuals are high. Two spawning strategies:

**Strategy A — Surface-based spawning (preferred):**
1. For each base Gaussian, compute its average residual contribution across training views
2. Rank by residual magnitude
3. Spawn DoG splats at the positions of the top-K base Gaussians (the ones with highest remaining error)
4. Initialize DoG scale to 0.5× the parent base Gaussian's scale (capture finer detail)
5. Initialize DoG color from the average residual at that location

**Strategy B — Image-space spawning (simpler):**
1. For each training view, find pixels with |residual| > threshold
2. Backproject these pixels to 3D using rendered depth from the base model
3. Cluster the 3D points (to avoid redundant DoG splats at the same location from different views)
4. Spawn DoG splats at cluster centers

Start with Strategy A (simpler, no backprojection needed).

**Number of DoG splats to spawn:**
Start with N_DoG = 0.5 × N_base. If the base has 200K Gaussians, spawn 100K DoG splats.
At 52 bytes per DoG vs 236 bytes per Gaussian, 100K DoGs add ~5 MB vs the base model's ~47 MB — a ~10% size increase. This gives a fair comparison against adding ~22K more standard Gaussians (same MB budget).

### Step 4: Train DoG Detail Layer

Freeze the base model entirely. Train only the DoG detail splats:

```python
# Training loop for detail layer
for step in range(10000):
    idx = random_camera()
    gt = gt_images[idx]

    # Pass 1: render frozen base
    with torch.no_grad():
        C_base = render(base_model, ...)

    # Pass 2: render detail DoGs
    means, quats, scales, opacities, sh0, shN = detail_model.get_all_gaussians()
    C_detail, _, _ = gsplat.rasterization(
        means, quats, torch.exp(scales),
        torch.sigmoid(opacities), torch.cat([sh0, shN], dim=1),
        viewmats, Ks, W, H, near, far,
        sh_degree=0,  # DC only for detail splats
    )

    # Combine
    C_final = (C_base + C_detail[0]).clamp(0, 1)

    # Loss on combined result
    loss = F.l1_loss(C_final, gt) + 0.2 * ssim_loss(C_final, gt)
    loss.backward()
    detail_optimizer.step()
```

Record test PSNR at 1K, 2K, ..., 10K steps.

### Step 5: Measure

**Primary metric — did we break through the ceiling?**

| Model | Test PSNR | Notes |
|---|---|---|
| Base 3DGS (converged, 30K steps) | ? dB | The ceiling |
| Base + DoG detail (10K DoG training steps) | ? dB | Any improvement = hypothesis confirmed |

**Any PSNR improvement is meaningful.** The base model is converged — it has hit the limit of what smooth Gaussians can represent. If DoG splats push past that ceiling, they're capturing signal that the Gaussian basis fundamentally cannot.

**Secondary metrics:**

| Metric | What It Shows |
|---|---|
| Per-pixel PSNR improvement map | Where do DoGs help? (should be edges/thin structures) |
| Residual before vs. after detail layer | How much error did the DoGs remove? |
| DoG coefficient magnitude distribution | Are most coefficients small? (wavelet sparsity → compressibility) |

### Step 6: Ablations (if primary result is positive)

1. **Scale ratio k:** Test k = √2, 2, 4. Which octave spacing works best?
2. **Number of DoG splats:** Sweep from 10K to 500K. Where does improvement saturate?
3. **Multi-level DoG:** Spawn DoGs at 2 scales (fine + medium). Does the second level help?
4. **Joint fine-tuning:** Unfreeze the base and train everything together for 5K steps. Does joint optimization find a better solution than the frozen-base setup?
5. **Lego scene:** Repeat on lego to test generality.

---

## 4. What Would Confirm the Hypothesis

**Confirmed:** PSNR improves beyond the converged base model. The per-pixel improvement map shows gains concentrated at edges and thin structures. The DoG detail layer is capturing signal that the Gaussian basis cannot.

**Null result:** PSNR stays flat. The converged base model's residual error is not edge-shaped — it's noise, view-dependent effects, or other signal that DoG wavelets don't help with either. The Gaussian ceiling is a general representation limit, not a basis-shape problem.

**Negative result:** PSNR drops. The DoG structure over-constrains the detail representation — the forced negative surround introduces artifacts. Sometimes you just need to add brightness at a point without subtracting from the surround.

---

## 5. Data Loader Note

The existing `data.py` loads LLFF/Neu3D format. NeRF Synthetic uses `transforms.json` (Blender convention). Need to either:

1. Add a NeRF Synthetic loader to `data.py` (standard format, many reference implementations available)
2. Convert NeRF Synthetic to LLFF format offline

Option 1 is cleaner. The format is simple:
```json
{
  "camera_angle_x": 0.6911,
  "frames": [
    {"file_path": "./train/r_0", "transform_matrix": [[4x4]]}
  ]
}
```

Camera convention: Blender (right-handed, Y-up, -Z forward). Convert to OpenCV (right-handed, Y-down, Z-forward) by flipping Y and Z rows of the transform matrix.

---

## 6. Implementation Priority

| Step | Effort | Blocks | Priority |
|---|---|---|---|
| NeRF Synthetic data loader | ~1 hour | Everything else | Do first |
| Train base model on drums | ~1 hour (GPU) | Steps 2-5 | Do second |
| Compute + visualize residuals | ~30 min | Step 3 | Quick sanity check — if residuals aren't edge-shaped, stop here |
| DoGDetailModel class | ~2 hours | Step 4 | Core implementation |
| Detail layer training loop | ~2 hours | Step 5 | Core implementation |
| Measurement + visualization | ~1 hour | — | Do last |
| **Total engineering time** | **~7 hours** | | |
| **Total GPU time** | **~2-3 hours** | | |
