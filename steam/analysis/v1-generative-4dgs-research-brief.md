# vZero Research Brief: Generative 4DGS from Video Diffusion Models

**Classification:** Research Proposal — Confidential
**Date:** February 21, 2026

---

## Motivation

One of vZero's highest-value content categories is volumetric performances of artists who cannot be physically captured — deceased performers, aged artists recreated in their prime, or historical performances that never had volumetric capture. This content category has no competition: no one else can produce 6DOF walkable performances of these artists at any quality level.

The 2D version of this problem is essentially solved. Current video generation models (Sora 2, Veo 3, Kling 3.0, HunyuanVideo) can generate photorealistic video of a performer who never stood in front of a camera — correct likeness, natural motion, temporally consistent, emotionally convincing. But the output is flat: a single-viewpoint video, not a volumetric 3D representation.

The question: **can we extend this capability to 4D Gaussian Splatting?**

---

## Proposed Approach

### Core Insight

Video generation models already contain strong implicit 3D knowledge — they understand human anatomy, how light falls on faces, how clothing drapes, how bodies move through space. This knowledge is encoded in the model's weights but expressed only as 2D pixel output. The proposal is to **extract this implicit 3D knowledge into an explicit 4DGS representation** by using the model's own multi-view outputs as a supervision signal.

### Production Workflow

The pipeline is driven by a **directed source performance**: a live stand-in performer, working under a director in a studio, acts out the deceased artist's performance — blocking, choreography, singing, facial expressions, emotional timing. This source video captures every creative decision. The AI pipeline then transfers the deceased performer's identity onto this directed motion, and lifts the result into 4DGS.

```
┌──────────────────────────────────────────────────────────────────┐
│                     PRODUCTION STAGE                              │
│                                                                  │
│  Director + Stand-in Performer                                   │
│  ↓                                                               │
│  Source video: the directed performance                           │
│  (standard video capture — single or multi-camera)               │
│  ↓                                                               │
│  Motion extraction:                                              │
│  ├── SMPL-X body pose sequence (per-frame 3D skeleton)           │
│  ├── Facial action units / expression parameters (per-frame)     │
│  ├── 3D spatial trajectory (where the performer moves in space)  │
│  ├── Hand articulation (per-frame finger poses)                  │
│  └── Timing / sync points (audio alignment)                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Video Generation Model (fine-tuned)                             │
│  (HunyuanVideo / Wan / CosXL — open-source, strong human prior) │
│                                                                  │
│  Conditioning inputs:                                            │
│  ├── Deceased performer identity embedding (face + body)         │
│  ├── Motion signal from source video (SMPL-X + expressions)      │
│  ├── 64 virtual camera positions (SynCamMaster multi-view)       │
│  └── Optional: wardrobe/style/lighting prompts                   │
│                                                                  │
│  The model generates the deceased performer executing the        │
│  stand-in's exact motion, from 64 simultaneous camera angles.    │
│  ↓                                                               │
│  64 video streams (the "synthetic multi-camera rig")             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌──────────────────┐          ┌──────────────────────────┐
│ 64 generated     │          │ 4DGS Decoder Head        │
│ video frames     │          │ (new trainable layers)   │
│ (2D supervision  │          │                          │
│  signal)         │          │ Outputs: Gaussian        │
│                  │          │ positions, covariances,  │
│                  │          │ opacities, SH coeffs     │
│                  │          │ per timestep             │
│                  │          │                          │
│                  │          │ Initialized on SMPL-X    │
│                  │          │ body scaffold            │
└────────┬─────────┘          └────────────┬─────────────┘
         │                                 │
         │                                 ▼
         │                    ┌──────────────────────────┐
         │                    │ Differentiable 4DGS      │
         │                    │ Renderer                 │
         │                    │                          │
         │                    │ Renders 4DGS from same   │
         │                    │ 64 camera positions      │
         │                    └────────────┬─────────────┘
         │                                 │
         ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│                  Photometric Loss                       │
│                                                        │
│  L = Σ(i=1..64) || VideoFrame_i - 4DGSRender_i ||    │
│                                                        │
│  + perceptual loss (LPIPS)                             │
│  + geometric regularization (SMPL-X prior)             │
│  + temporal coherence loss                             │
│  + opacity / size regularization                       │
│  + identity preservation loss (ArcFace embedding)      │
│  + motion fidelity loss (source pose vs. output pose)  │
│                                                        │
│  Gradients flow back through differentiable splatting  │
│  into the 4DGS decoder layers                          │
└────────────────────────────────────────────────────────┘
```

### Why Source Video Conditioning Makes This Much Easier

Unconditional generation (text prompt → performance) requires the AI to solve everything simultaneously: what the performer looks like, how they move, where they go, what expression they make, the timing of every gesture. This is the hardest possible formulation.

Source video conditioning **factorizes the problem**:

| What the AI must solve | What the director/stand-in provides |
|----------------------|-------------------------------------|
| Deceased performer's identity/appearance | Body motion and blocking |
| 3D multi-view consistency | Facial expressions and emotion |
| Clothing/hair physics on the target body | Choreography and spatial trajectory |
| | Singing timing and lip sync |
| | Creative direction and performance arc |

The AI's job reduces to **identity transfer + 3D lifting** — both well-studied problems. The creative and performative intelligence comes entirely from the human director and stand-in. This is analogous to how de-aging works in film (capture a real performance, transfer the younger face), but extended to volumetric 3D.

### Why This Works (The Geometric Bottleneck Argument)

A 4DGS representation is a single set of 3D Gaussians. It physically cannot represent view-inconsistent content. If view 12 says the nose points left and view 37 says it points right, the Gaussians converge to the best 3D consensus that minimizes total photometric error across all 64 views simultaneously.

This is identical in principle to standard 3DGS reconstruction from real multi-camera footage. The optimization doesn't care whether the supervision images come from physical cameras or from a generative model. The 4DGS acts as a **consensus mechanism** that distills the video model's approximate 3D knowledge into a geometrically precise 3D structure.

Even if the video model's individual frames are slightly inconsistent view-to-view, the 4DGS can't cheat — it must find a single 3D configuration that best explains all 64 views. Inconsistencies are resolved by the optimization, not propagated.

---

## Relationship to Existing Work

The proposal is novel in its end-to-end formulation but closely related to several published methods:

| Paper | Relationship | Difference |
|-------|-------------|------------|
| **NVIDIA Lyra** (ICLR 2026) | Self-distills 3D from video diffusion into 3DGS | Lyra uses latent-space distillation; this proposal uses photometric supervision from rendered views |
| **CAT4D** (CVPR 2025) | Multi-view video diffusion → 4DGS optimization | CAT4D is two-stage (generate then reconstruct); this proposal is end-to-end trainable |
| **Splat4D** (SIGGRAPH 2025) | Video diffusion → 4DGS with text-guided editing | Demonstrates the video-to-4DGS lift works; does not train the decoder end-to-end |
| **Virtually Being** (SIGGRAPH Asia 2025) | Trains video diffusion on volumetric captures | Inverse direction: capture → video model. This proposal: video model → 4DGS |
| **GEN3C** (CVPR 2025 Highlight) | 3D cache-conditioned video generation | Provides the multi-view conditioning architecture; could serve as the base model |
| **DEGS Fixer** (ACM TOG 2025) | Diffusion-based 4DGS quality enhancement | Complementary: could enhance the output of this pipeline as a post-process |
| **Disco4D** (CVPR 2025) | Disentangled 4D human generation from single image; separates clothing from body | Demonstrates motion-conditioned 4DGS generation with appearance control; validates that pose-driven identity transfer works in GS space |
| **Animate Anyone / MagicAnimate** | Pose-conditioned video generation of specific identities | Solves the 2D version of our core problem (source motion → target identity video); we extend to multi-view + 4DGS |

**The novel contribution** is the combination of: (a) using a directed source performance to drive the motion of an AI-generated performer (factorizing the problem into motion capture + identity transfer + 3D lifting), (b) using a video generation model as a synthetic multi-camera rig for subjects that can't be physically captured, (c) training a 4DGS decoder end-to-end against the model's own multi-view outputs, and (d) the geometric bottleneck forcing 3D consistency from potentially inconsistent 2D supervision.

---

## Technical Design

### Base Model Selection

Requirements for the video generation backbone:

| Requirement | Rationale | Candidates |
|------------|-----------|------------|
| Open-source / open-weights | Must be fine-tunable | HunyuanVideo 1.5, Wan, CogVideoX |
| Strong human prior | Anatomically correct generation | All major models |
| **Motion/pose conditioning** | Accept SMPL-X pose sequence as driving signal | ControlNet-Pose, Animate Anyone, MagicAnimate, Disco4D |
| Multi-view capability | Generate 64 consistent views | SynCamMaster plugin, or fine-tune with epipolar attention (CVD) |
| Camera control | Specify exact virtual camera positions | GEN3C-style 3D cache conditioning |
| Identity conditioning | Preserve specific performer likeness | IP-Adapter, face embedding injection |
| Expression transfer | Map facial action units from source to target | FLAME/EMOCA expression parameters, Audio2Face |

**Recommended starting point:** HunyuanVideo 1.5 (Tencent, open-source) with SynCamMaster multi-view plugin, ControlNet-Pose for motion conditioning, and strong identity conditioning via reference images. The motion conditioning stack (SMPL-X + facial expression + hand articulation) is the primary input; the multi-view and identity conditioning layers wrap around it.

### 4DGS Decoder Architecture

The decoder head takes features from the video model's intermediate representations and outputs 4DGS primitives:

**Per-Gaussian outputs:**
- Position (x, y, z) — 3 params
- Covariance (rotation quaternion + scale) — 7 params
- Opacity — 1 param
- Color (spherical harmonics, degree 2) — 27 params
- **Total: 38 params per Gaussian**

**Temporal model:** Hybrid 3D-4D architecture (Oh et al., 2025). Static Gaussians (background, body core) are 3D. Dynamic Gaussians (clothing, hair, expressions) use a deformation field conditioned on time. This reduces parameter count and improves temporal stability.

**Geometric scaffold:** Initialize Gaussians on an SMPL-X body model mesh surface. The decoder learns offsets from this prior, not absolute positions from scratch. This constrains the search space and prevents degenerate solutions (flat billboards, single-color blobs, geometric collapse).

### Training Strategy

**Progressive view count:**

| Phase | Views | Purpose |
|-------|-------|---------|
| Phase 1 | 4–8 views | Learn coarse geometry. Consistency is easier to maintain with fewer views. |
| Phase 2 | 16 views | Refine body proportions, face structure. |
| Phase 3 | 32 views | Add fine detail — clothing folds, hair, hand articulation. |
| Phase 4 | 64 views | Full coverage. Polish. |

Starting with 64 views risks overwhelming the decoder with inconsistent supervision. Progressive training lets the model build a stable geometric foundation before adding difficult viewpoints.

**Loss function:**

```
L_total = λ_photo · L_photometric       (L1 + SSIM across all views)
        + λ_perc  · L_perceptual         (LPIPS for high-level features)
        + λ_geom  · L_geometric          (SMPL-X mesh surface proximity)
        + λ_temp  · L_temporal            (Gaussian trajectory smoothness)
        + λ_reg   · L_regularization      (opacity sparsity, scale bounds)
```

**Identity anchoring:**

For a specific deceased performer, fine-tune the video model on all available reference material — photographs, video footage, film — from every available angle and time period. Use face embedding loss (ArcFace/AdaFace) and body proportion constraints. The stronger the identity prior in the video model, the more consistent the multi-view output, the sharper the resulting 4DGS.

**Source video integration:**

The stand-in's directed performance is processed into structured conditioning signals before feeding the model:

1. **SMPL-X body pose** — extracted per-frame using 4DHumans, SMPLer-X, or similar. Provides full skeletal pose including spine, limbs, hands.
2. **Facial expression parameters** — extracted via EMOCA or FLAME fitting. Captures every micro-expression, lip shape, eyebrow raise.
3. **3D spatial trajectory** — the stand-in's path through the performance space, mapped to the virtual stage coordinates.
4. **Audio alignment** — the stand-in sings or lip-syncs the original audio. Timing markers are extracted for lip sync conditioning (Audio2Face / wav2lip).
5. **Hand articulation** — dedicated hand pose extraction (HaMeR, FrankMocap) for expressive finger motion.

These signals condition the video model frame-by-frame. The model generates the deceased performer's appearance following these exact motion cues. Because the motion is physically performed by a real human, it is inherently natural — no uncanny motion artifacts from AI generation.

### Two-Head Design (Recommended Enhancement)

Rather than replacing the video output with 4DGS, maintain both heads:

- **Video head:** Generates multi-view frames (supervision signal)
- **4DGS head:** Generates Gaussians (target output)

Both are rendered and compared. Additionally, add a **cross-head consistency loss** where the 4DGS renders are compared against the video frames using a discriminator or contrastive loss. This provides richer gradients and allows the 4DGS head to learn from the video head's perceptual quality while the geometric bottleneck enforces 3D precision.

---

## Expected Quality

### Quality Ceiling Analysis

The 4DGS output quality is bounded by the multi-view consistency of the base video model. **Source video conditioning significantly improves this ceiling** because the model doesn't need to invent motion — it only needs to transfer identity onto a given pose sequence, which is a much more constrained generation task with less room for view-to-view inconsistency.

| Multi-view consistency level | Estimated 4DGS quality | Based on |
|-----------------------------|----------------------|----------|
| Unconditional generation (text prompt only) | ~25–30 dB PSNR | Published multi-view NVS benchmarks |
| Current SOTA multi-view (SynCamMaster, GEN3C) | ~28–33 dB PSNR | Published benchmarks |
| + Source video motion conditioning | +2–4 dB | Motion is given, not generated — fewer degrees of freedom for inconsistency |
| + Identity-anchored fine-tuning | +2–3 dB | Stronger identity prior = more consistent generation |
| + Progressive training + geometric scaffold | +2–3 dB | Consensus mechanism resolves remaining inconsistencies |
| **Estimated ceiling (2026)** | **~32–37 dB PSNR** | |
| **Estimated ceiling (2028, projected)** | **~37–42 dB PSNR** | Multi-view consistency improving ~3–5 dB/year |

The source video conditioning is a meaningful quality boost because it eliminates an entire category of inconsistency: **temporal motion disagreement**. When the model generates motion from a text prompt, different views can disagree about the exact arm position at frame N. When motion is provided as a structured SMPL-X pose, all 64 views are conditioned on the same skeleton — the only remaining inconsistency is in appearance/texture, not geometry.

### Comparison to Alternatives

| Method | Quality | 6DOF | Deceased performers |
|--------|---------|------|-------------------|
| Hologram concerts (Pepper's ghost) | Low | No | Yes |
| ABBA Voyage (motion capture + CG) | Medium | No | Yes (with living reference) |
| De-aging VFX (film) | High | No (2D film only) | Yes |
| This proposal (source-video-driven) | Medium-high (32–37 dB) | **Yes** | **Yes** |
| Real multi-camera capture | Highest (40+ dB) | Yes | No (requires physical presence) |

**For deceased performers, this is the only viable path to 6DOF volumetric content.** Even at 30 dB, a volumetric performance that 12 people can walk around in a 200 m² room is unprecedented and unchallenged.

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Multi-view inconsistency caps quality at <30 dB** | Medium | High | Progressive view count training; geometric scaffold; fine-tune base model with epipolar losses |
| **Janus problem (different identity per view)** | Medium | Critical | Strong identity anchoring via face embeddings and reference fine-tuning; SMPL-X body prior |
| **Training instability / mode collapse** | Medium | High | Two-head design; regularization; progressive training; curriculum learning |
| **Temporal incoherence (flickering 4DGS)** | Medium | Medium | Temporal smoothness loss; Hybrid 3D-4D architecture; deformation field |
| **Compute requirements exceed budget** | Low | Medium | Start with shorter sequences (5–10 seconds); reduce view count if needed; leverage existing fine-tuned checkpoints |
| **Uncanny valley at 28–33 dB** | Low | High | Art direction (lighting, environment) to set emotional context; avoid extreme closeups initially |

### IP & Legal Risks

| Risk | Mitigation |
|------|------------|
| Performer likeness rights | Secure rights from estate/rights holders before production, same as any posthumous use |
| Training data provenance | Use only licensed/public-domain reference material for identity fine-tuning |
| Video model license | Use permissively licensed open-source models (Apache 2.0: HunyuanVideo, Wan) |
| Patentability of the method | The end-to-end architecture (video diffusion → geometric bottleneck → 4DGS) is novel; pursue patent filing |

---

## Resource Requirements

### Phase 1: Proof of Concept (3–4 months)

| Resource | Spec | Purpose |
|----------|------|---------|
| **Researchers** | 2–3 (ML + 3D vision) | Architecture design, training, evaluation |
| **Compute** | 8× H100 (or equivalent) | Fine-tuning video model + 4DGS decoder training |
| **Reference data** | Licensed performer imagery | Identity anchoring |
| **Evaluation** | Perceptual study + PSNR/LPIPS | Quality assessment |

**Deliverable:** 10-second 4DGS clip of a specific performer, viewable from multiple angles, at estimated 28–33 dB quality.

### Phase 2: Quality Push (3–4 months)

| Resource | Spec | Purpose |
|----------|------|---------|
| **Researchers** | 3–4 | Progressive training, identity conditioning, temporal coherence |
| **Compute** | 16× H100 | Larger model, more views, longer sequences |

**Deliverable:** 60-second 4DGS performance at 30–35 dB quality, suitable for venue viewing distance (>1 meter).

### Phase 3: Production Integration (3–4 months)

| Resource | Spec | Purpose |
|----------|------|---------|
| **Engineers** | 2–3 | Pipeline integration, quality tooling |
| **Artists** | 1–2 | Art direction, lighting, environment composition |

**Deliverable:** Full 20-minute show-quality 4DGS performance integrated into the vZero playback pipeline, with AI-enhanced detail (DEGS Fixer post-process) and authored environment.

---

## Strategic Value for vZero

### Content Unlock

If this research succeeds at 32+ dB quality, vZero gains access to a content catalog that no competitor can match:

- **Legacy artists:** Performances by deceased legends recreated volumetrically — viewed from any angle, walked around, experienced as spatial presence. A director works with a stand-in to craft every moment of the performance; the AI delivers the legend's likeness.
- **Prime recreations:** Living artists recreated at a younger age — a 2026 artist performing as they were in 1985. The artist themselves could serve as the motion source (their body, their voice, their emotion), with only the visual identity transferred to their younger self.
- **Historical performances:** Iconic concert moments recreated as volumetric experiences, with motion reference from the original footage driving new multi-view generation.
- **Cross-era duets:** A young artist performing alongside their older self, or alongside another artist from a different era. Each driven by a separate stand-in, composed into the same 4DGS scene.
- **Unlimited repertoire from a single performer:** One stand-in session generates the motion; the AI can re-render that same motion with any licensed performer identity. The same directed choreography could produce versions with different artists.

This is exclusive content by definition. The AI pipeline is the moat — no one can capture these performances because the subjects don't exist. The directed production workflow ensures every performance meets artistic standards — a director controls every creative choice through the stand-in, just as they would on a film set.

### Competitive Position

| Competitor | Can they do this? |
|-----------|------------------|
| The Sphere | No (LED screens, not volumetric) |
| ABBA Voyage | Partially (motion capture + CG avatars, not volumetric) |
| Hologram concert companies | No (2D projection, not 6DOF) |
| Other VR venues | No (no generative 4DGS pipeline) |

### Content Economics

A single deceased artist's estate licenses their likeness once. The AI pipeline generates unlimited performances — different songs, different eras, different visual treatments — from that single license. Content production cost per show drops dramatically after the initial research investment and identity fine-tuning.

---

## Recommended Next Steps

1. **Select base video model** — evaluate HunyuanVideo 1.5, Wan, and CogVideoX for multi-view consistency and human generation quality. Benchmark each on a standardized multi-view reconstruction task.

2. **Build identity conditioning pipeline** — assemble reference material for one target performer. Fine-tune the selected model with identity preservation loss. Evaluate likeness fidelity.

3. **Implement 4DGS decoder head** — design the decoder architecture, differentiable renderer integration, and loss function. Start with static 3DGS (single frame, 4–8 views) before adding temporal dimension.

4. **Proof of concept** — produce a 10-second clip. Evaluate quality (PSNR, LPIPS), geometric consistency (multi-view depth coherence), and perceptual realism (human evaluation).

5. **File provisional patent** — the end-to-end architecture (video diffusion model self-supervising a 4DGS decoder through a geometric bottleneck) is novel. File before publication.

---

## Relationship to vZero Roadmap

This research project runs **parallel to the V1 critical path**, not on it. V1 launches with captured performer content. This pipeline targets V2/V3 content expansion or a dedicated "Legacy" content vertical.

| vZero Phase | This Research | Integration |
|-------------|--------------|-------------|
| V1 (M0–M30) | Proof of concept (Phase 1–2) | None — V1 uses captured content only |
| V2 (post-launch) | Production pipeline (Phase 3) | Legacy artist shows as premium content |
| V3 (future) | Refined pipeline + AI interaction | AI-generated performers could respond to users if quality permits |

The research requires 2–4 dedicated researchers and ~$500K–$1M in compute over 9–12 months. This is a small investment relative to the Series A ($40M) with potentially transformative content implications.

---

## Supporting Research

- **State-of-the-art survey:** exploring/md/_state-of-the-art-february-2026.md
- **V1 delivery architecture:** steam/analysis/v1-delivery-architecture.md
- **Generative AI for 4DGS landscape:** (findings from research session, February 21, 2026)

### Key References

- NVIDIA Lyra — Self-distillation from video diffusion to 3DGS (ICLR 2026)
- CAT4D — Multi-view video diffusion → 4DGS (CVPR 2025)
- Splat4D — Video diffusion → 4DGS with text-guided editing (SIGGRAPH 2025)
- Virtually Being — Volumetric-capture-trained video diffusion (SIGGRAPH Asia 2025)
- GEN3C — 3D cache-conditioned video generation (CVPR 2025 Highlight)
- SynCamMaster — Multi-camera video generation (Dec 2024)
- DEGS — Diffusion-based 4DGS quality enhancement (ACM TOG 2025)
- Disco4D — Disentangled 4D human generation (CVPR 2025)
- Hybrid 3D-4D GS — Static/dynamic Gaussian decomposition (2025)
- AGORA — Adversarial generation of animatable 3DGS head avatars (Dec 2025)
- GSFix3D — Diffusion-guided repair of GS novel views (2025)
- Animate Anyone / MagicAnimate — Pose-conditioned identity-preserving video generation
- SMPL-X / 4DHumans / SMPLer-X — Body pose extraction for motion conditioning
- EMOCA / FLAME — Facial expression parameter extraction

---

*This document is part of the vZero Strategic Document Suite. It describes a research proposal, not a committed production plan. All quality estimates are projections based on February 2026 state-of-the-art benchmarks.*
