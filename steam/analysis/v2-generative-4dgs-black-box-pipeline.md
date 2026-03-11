# vZero Production Pipeline: Generative 4DGS via Black Box Video Models

**Classification:** Production Architecture — Confidential
**Date:** February 21, 2026
**Related:** v1-generative-4dgs-research-brief.md (end-to-end research approach)

---

## Overview

This document describes the **production-ready** pipeline for generating 4DGS content of performers who cannot be physically captured — deceased artists, aged performers recreated in their prime, or historical performances. It uses state-of-the-art closed-source video generation models as a black box, decoupled from a standard 4DGS reconstruction stage.

This is the simpler, higher-quality, faster-to-deploy variant of the approach described in v1-generative-4dgs-research-brief.md. That document proposes an end-to-end trainable architecture requiring open-weights models. This document treats the video model as an opaque API and achieves higher output quality by accessing the best available closed-source models.

**The two approaches are complementary, not competing.** This pipeline is the production path for V2 content. The end-to-end research track builds proprietary capability for V3 (interactive AI performers) where API dependency is unacceptable.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 0: DIRECTED PERFORMANCE                │
│                                                                  │
│  Director + Stand-in Performer (studio or capture stage)         │
│  ↓                                                               │
│  The stand-in performs under direction:                           │
│  blocking, choreography, singing, facial expression, emotion     │
│  ↓                                                               │
│  Captured as standard video (single or multi-camera)             │
│  ↓                                                               │
│  Motion extraction:                                              │
│  ├── SMPL-X body pose sequence        (4DHumans / SMPLer-X)     │
│  ├── Facial expression parameters     (EMOCA / FLAME)           │
│  ├── Hand articulation                (HaMeR / FrankMocap)      │
│  ├── 3D spatial trajectory            (stage coordinates)        │
│  └── Audio/lip sync alignment         (wav2lip / Audio2Face)    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 1: MULTI-VIEW GENERATION               │
│                     (Black Box — Best Available API)              │
│                                                                  │
│  API Input:                                                      │
│  ├── Deceased performer identity (reference images/video)        │
│  ├── Motion conditioning (SMPL-X + expression + hands)           │
│  ├── 64 virtual camera positions (known geometry)                │
│  ├── Audio track (for lip sync)                                  │
│  └── Style/wardrobe/lighting prompts (text or reference)         │
│                                                                  │
│  API Output:                                                     │
│  └── 64 synchronized video streams                               │
│      (the deceased performer executing the directed motion,      │
│       viewed from 64 angles)                                     │
│                                                                  │
│  Model: Sora 2 / Veo 3 / Kling 3.0 / whatever is SOTA          │
│  Switchable — no lock-in to any single provider                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 2: 4DGS RECONSTRUCTION                 │
│                     (Standard pipeline — fully owned)             │
│                                                                  │
│  Input: 64 video streams + known camera positions                │
│                                                                  │
│  2a. Camera calibration                                          │
│      Camera positions are known (we specified them).             │
│      Optional: refine with COLMAP if generated views have        │
│      slight position drift.                                      │
│                                                                  │
│  2b. 4DGS optimization                                           │
│      Standard multi-view 4DGS reconstruction:                    │
│      ├── Hybrid 3D-4D GS (static body core + dynamic details)   │
│      ├── SMPL-X geometric scaffold (same poses used in Stage 1)  │
│      ├── Progressive optimization (coarse → fine)                │
│      └── Temporal regularization (smooth Gaussian trajectories)  │
│                                                                  │
│  2c. Quality assessment                                          │
│      ├── Multi-view consistency check (render from held-out      │
│      │   viewpoints not in the 64, compare to additional         │
│      │   generations from those viewpoints)                      │
│      ├── Identity verification (face embedding distance)         │
│      ├── Motion fidelity (pose comparison vs. source)            │
│      └── Temporal coherence (flicker detection, swimming)        │
│                                                                  │
│  Output: Raw 4DGS asset                                          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 3: AI ENHANCEMENT                      │
│                     (Quality push — fully owned)                  │
│                                                                  │
│  3a. DEGS Fixer-style enhancement                                │
│      Diffusion-based detail recovery: skin texture, fabric       │
│      weave, hair strands, fine facial features.                  │
│      Trained on paired data from real volumetric captures.        │
│                                                                  │
│  3b. GSFix3D / 3DGS-Enhancer                                    │
│      Diffusion-guided repair of novel-view artifacts.            │
│      Hole filling, artifact removal, consistency enforcement.    │
│                                                                  │
│  3c. Relighting (GS³ / SVG-IR)                                  │
│      Material decomposition → re-render under target lighting    │
│      to match the authored environment.                          │
│                                                                  │
│  Output: Enhanced 4DGS asset                                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 4: COMPOSITION & BAKE                  │
│                                                                  │
│  4a. Composite performer into authored 4DGS environment          │
│      (meadow, concert stage, abstract space — from the           │
│       CG environment pipeline)                                   │
│                                                                  │
│  4b. Lighting integration                                        │
│      Match performer lighting to environment lighting.           │
│      Relightable GS makes this a parameter adjustment,           │
│      not a re-render.                                            │
│                                                                  │
│  4c. Final quality pass                                          │
│      Anti-aliasing, floater removal, temporal polish.            │
│                                                                  │
│  4d. Compression                                                 │
│      PCGS progressive compression + RTGS hierarchical subsets    │
│      for foveated streaming.                                     │
│                                                                  │
│  Output: Production 4DGS asset ready for venue playback          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why Black Box Is the Right Production Choice

### 1. Quality ceiling is higher

Closed-source models lead open-source by an estimated 2–5 dB in human generation quality and multi-view consistency. For production content, this gap directly translates to output quality.

| Model tier | Estimated multi-view consistency | Notes |
|-----------|--------------------------------|-------|
| Open-source SOTA (HunyuanVideo 1.5, Wan) | ~28–33 dB | Good, rapidly improving |
| Closed-source SOTA (Sora 2, Veo 3, Kling 3.0) | ~30–37 dB | Best available; proprietary training data and compute |

With source video conditioning adding +2–4 dB and the 4DGS geometric bottleneck adding +2–3 dB on top, the production ceiling is:
- **Open-source path:** ~32–37 dB
- **Black box path:** ~34–40 dB

The black box path approaches the 40 dB production target.

### 2. No engineering debt

The video generation model is someone else's problem. No fine-tuning infrastructure, no GPU cluster for training, no model maintenance. The vZero team focuses on what they own: the directed performance workflow, the 4DGS reconstruction pipeline, the AI enhancement stack, and the venue playback system.

### 3. Always on the quality frontier

When Sora 3 or Veo 4 ships with better multi-view consistency, you switch. No retraining. No architecture changes. The 4DGS reconstruction pipeline doesn't care where the frames came from — it just needs 64 views.

### 4. Faster time to production

No research phase needed for the generation side. The pipeline can be built and tested as soon as a suitable API supports the required conditioning inputs (identity + pose + multi-view cameras).

---

## API Requirements

For this pipeline to work, the video generation API must support:

| Capability | Required | Status (Feb 2026) |
|-----------|----------|-------------------|
| **Identity conditioning** — generate a specific person from reference images | Yes | Widely available (Sora 2, Veo 3, Kling 3.0, Runway Gen-4 all support reference-based identity) |
| **Pose/motion conditioning** — drive generation with SMPL-X or skeleton input | Yes | Available via ControlNet-style adapters in most platforms; native in Kling 3.0 (physics-based motion) |
| **Multi-view generation** — generate the same moment from specified camera angles | Yes | **Emerging.** Sora 2 supports camera trajectory control. Veo 3 has limited camera specification. No API natively generates 64 simultaneous synchronized views. |
| **Facial expression conditioning** — drive expressions from FLAME/EMOCA params | Preferred | Limited. Lip sync from audio is common; fine-grained expression transfer is not yet standard in APIs. |
| **Deterministic seeding** — same prompt + seed = same output across views | Preferred | Most APIs support seeding for reproducibility. |

### The Multi-View Gap

The biggest limitation today: **no commercial API natively generates 64 synchronized camera views from a single prompt.** Current workarounds:

**Option A — Sequential generation with strong conditioning:**
Generate each of the 64 views separately, conditioned on the same identity + pose + a specific camera angle. Rely on the conditioning to maintain consistency. Risk: view-to-view drift in appearance details.

**Option B — Few-view generation + novel view synthesis:**
Generate 4–8 key views from the API. Use Stable Virtual Camera or GEN3C to interpolate the remaining 56–60 views. The key views anchor the identity and geometry; the NVS model fills in between.

**Option C — Iterative refinement:**
Generate 64 views (Option A). Reconstruct initial 4DGS. Identify inconsistent regions. Re-generate those specific views with additional conditioning (render the current 4DGS from nearby views as reference). Iterate until convergence.

**Option D — Wait for native multi-view APIs:**
SynCamMaster (research, Dec 2024) and GEN3C (NVIDIA, open-source) demonstrate that multi-view generation is technically feasible. Commercial APIs will likely support it within 12–18 months. Develop the rest of the pipeline now; plug in native multi-view generation when available.

**Recommended approach:** Start with Option B (few key views + NVS interpolation). This is production-viable today. Upgrade to native multi-view API when available.

---

## Quality Control Workflow

The black box model is not directly controllable, so quality must be enforced through **iterative generation and selection**:

### Per-Sequence QC

```
Generate 64 views
       ↓
Reconstruct 4DGS
       ↓
Automated QC checks:
├── Multi-view photometric consistency (PSNR across held-out views)
├── Identity verification (ArcFace distance to reference < threshold)
├── Pose fidelity (SMPL-X joint angle error vs. source < threshold)
├── Temporal coherence (optical flow smoothness, flicker score)
└── Geometric quality (depth map consistency, floater count)
       ↓
Pass? → Proceed to enhancement
Fail? → Re-generate problematic frames/views with adjusted conditioning
```

### Iterative Refinement Loop

When specific views fail QC:

1. Render the current 4DGS from the failing viewpoint
2. Use this render as additional conditioning for re-generation ("make it look like this, but from this angle")
3. Replace the failing views with re-generated ones
4. Re-optimize the 4DGS with the updated views
5. Re-check QC

This loop typically converges in 2–3 iterations. The 4DGS consensus progressively improves as the worst views are replaced.

### Human Review Gates

| Gate | Who | What they check |
|------|-----|----------------|
| Performance direction | Director | Does the stand-in performance capture the intended emotion, timing, choreography? |
| Identity approval | Art director + estate representative | Does the generated performer look correct? (Critical for deceased artist estates.) |
| 4DGS spatial review | Technical artist | Walk around the 4DGS in VR. Check for geometric artifacts, uncanny valley, clothing physics. |
| Final composition | Director | Performer + environment + lighting + audio together. Full show context. |

---

## Cost Model

### Per-Performance Generation Cost

| Stage | Cost driver | Estimated cost |
|-------|-----------|---------------|
| **Directed performance** | Studio time + stand-in fee + director | $5K–$20K per session (standard production cost) |
| **Motion extraction** | GPU compute for pose/expression fitting | ~$100 (automated pipeline) |
| **Multi-view generation** | API calls: 64 views × show duration | $2K–$10K per 20-minute show (depends on API pricing, resolution, iteration count) |
| **4DGS reconstruction** | GPU compute: ~8 H100-hours per minute of content | $500–$2K per 20-minute show |
| **AI enhancement** | GPU compute: DEGS Fixer + GSFix3D + relighting | $500–$1K per show |
| **QC + iteration** | 2–3 refinement rounds × partial re-generation | ~2× the generation cost |
| **Composition + bake** | Artist time + GPU compute | $2K–$5K per show |
| **Total per 20-minute show** | | **~$15K–$50K** |

Compare to a real volumetric capture show: $500K–$2M (capture studio rental, artist fees, 4DGS processing, VFX integration). The generative pipeline is **10–100× cheaper per show** once the directed performance is recorded.

### Marginal Cost of Variants

Because the motion is captured once and identity is a parameter:

| Variant | Additional cost |
|---------|----------------|
| Same motion, different performer identity | ~$5K–$15K (re-generate + reconstruct, no new stand-in session) |
| Same performer, different wardrobe | ~$3K–$10K (re-generate with wardrobe prompt change) |
| Same performer, different environment | ~$2K–$5K (re-composite into new environment) |
| Same performer, different lighting | ~$500–$1K (relight only, no re-generation) |

This is the **content economics multiplier**: one directed performance session → many shows.

---

## Risk Assessment

### Risks Specific to the Black Box Approach

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **API discontinuation or terms change** | Medium | Critical | Maintain compatibility with 3+ providers. Pipeline is provider-agnostic by design. In parallel, develop the open-weights end-to-end approach (v1-generative-4dgs-research-brief.md) as a fallback. |
| **API doesn't support required conditioning** | Medium | High | Use Option B (few views + NVS interpolation) as a bridge. Lobby API providers for multi-view support. |
| **API rate limits / cost spikes** | Medium | Medium | Budget for worst-case iteration counts. Negotiate enterprise pricing. Pre-generate during off-peak. |
| **API content policy blocks deceased performer generation** | Medium | High | Pre-negotiate with API provider. Use estate authorization documentation. If blocked, fall back to open-weights pipeline. |
| **Generated content quality varies between API versions** | Low | Medium | Pin to specific API versions. Maintain QC pipeline to catch regressions. |
| **Competitor accesses the same API** | High | Low | The moat is not the video generation — it's the full pipeline (directed performance + 4DGS reconstruction + enhancement + venue playback). Anyone can generate video; no one else has the 4DGS-to-venue delivery chain. |

### The Content Policy Risk

This deserves special attention. Major AI providers (OpenAI, Google, Meta) have content policies around generating likenesses of real people, especially deceased individuals. Mitigations:

1. **Estate authorization:** Obtain explicit written permission from the performer's estate/rights holders before any generation. This is required regardless for commercial use of the likeness.
2. **Enterprise agreements:** Negotiate custom content policies with the API provider under an enterprise contract. Enterprise tiers typically have more permissive policies for authorized commercial use.
3. **Provider diversification:** If one provider blocks it, another may allow it. Maintain compatibility with multiple APIs.
4. **Fallback to open weights:** The end-to-end research pipeline (v1-generative-4dgs-research-brief.md) has no content policy restrictions — you control the model.

---

## Comparison: Black Box vs. End-to-End

| Dimension | Black Box (this document) | End-to-End (research brief) |
|-----------|--------------------------|----------------------------|
| **Time to production** | 6–9 months | 12–18 months |
| **Quality ceiling (2026)** | ~34–40 dB | ~32–37 dB |
| **Quality ceiling (2028)** | ~38–44 dB | ~37–42 dB |
| **Engineering complexity** | Low (pipeline integration) | High (ML research + custom training) |
| **Capital requirement** | Low (~$200K pipeline dev + API costs) | High (~$1M+ compute + researchers) |
| **IP ownership** | Pipeline only; model is third-party | Full stack including generation model |
| **API dependency** | Yes (critical) | No |
| **Content policy risk** | Yes | No |
| **Upgradability** | Swap to better API any time | Requires retraining |
| **Path to V3 (interactive AI)** | No (can't run black box in real-time venue loop) | Yes (own model, can optimize for latency) |

**Recommendation:** Pursue both tracks.

- **Black box pipeline** for V2 production content (legacy artist shows, prime recreations). Ship quality content faster, at lower cost, using the best available models.
- **End-to-end research** for V3 capability (interactive AI performers). Build proprietary technology that doesn't depend on third-party APIs and can eventually run in the real-time venue loop.

---

## Implementation Timeline

### Phase 1: Pipeline Development (3–4 months)

| Deliverable | Detail |
|------------|--------|
| Motion extraction pipeline | Automated SMPL-X + expression + hand extraction from source video |
| API integration layer | Abstract interface supporting Sora 2, Veo 3, Kling 3.0 APIs with conditioning inputs |
| 4DGS reconstruction pipeline | Optimized for generated (not captured) multi-view input, with SMPL-X scaffold |
| QC automation | Automated consistency, identity, pose, and temporal checks |
| Few-view + NVS bridge | Option B pipeline (4–8 API views → Stable Virtual Camera interpolation → 64 views) |

### Phase 2: First Production (2–3 months)

| Deliverable | Detail |
|------------|--------|
| Target performer selection | Select deceased artist with estate authorization and abundant reference material |
| Stand-in casting + direction | Cast stand-in with similar build. Record directed performance (1 song, ~4 minutes). |
| Generation + reconstruction | Produce 4DGS via the pipeline. Iterate through QC until quality targets met. |
| Enhancement + composition | DEGS enhancement, relighting, environment composition. |
| Review | Director + estate review of final 4DGS in VR headset. |

### Phase 3: Scale (ongoing)

| Deliverable | Detail |
|------------|--------|
| Multi-performer library | Expand to additional deceased/legacy artists |
| API upgrade integration | Switch to native multi-view generation when available |
| Pipeline optimization | Reduce iteration count, automate more QC, decrease per-show cost |

**Total timeline from start to first viewable 4DGS performance: ~6 months.**

---

## Relationship to vZero Roadmap

| vZero Phase | This Pipeline | Content Impact |
|-------------|--------------|----------------|
| V1 (M0–M30) | Pipeline development (Phase 1–2) | Not on V1 critical path. V1 launches with captured content. |
| V2 (post-launch) | Production deployment (Phase 3) | Legacy artist shows as premium exclusive content. Dramatically expands the content library without additional capture sessions. |
| V3 (future) | Continues as production tool | Black box pipeline produces content; end-to-end research (separate track) enables interactive AI performers. |

---

## Supporting Documents

- **End-to-end research approach:** steam/analysis/v1-generative-4dgs-research-brief.md
- **V1 delivery architecture:** steam/analysis/v1-delivery-architecture.md
- **State-of-the-art survey:** exploring/md/_state-of-the-art-february-2026.md

---

*This document is part of the vZero Strategic Document Suite. It describes a production pipeline, not a research proposal. Quality estimates are based on February 2026 state-of-the-art benchmarks and will improve as video generation models advance.*
