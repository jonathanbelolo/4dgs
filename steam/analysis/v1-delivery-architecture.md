# vZero V1: Delivery Architecture Analysis

**Classification:** Technical Analysis — Confidential
**Date:** February 21, 2026

---

## Overview

This document captures the refined V1 technical architecture for vZero, based on February 2026 state-of-the-art research and design decisions made during this analysis session. It supersedes the rendering and streaming assumptions in the original Technical Architecture document (steam/summaries/2-technical-architecture.md).

The key insight: **V1 is a playback engine, not a real-time rendering engine.** All content — performer, environment, lighting, VFX — is pre-baked into a single 4DGS asset during offline production. The venue server plays it back and renders foveated viewpoints. This dramatically simplifies the real-time pipeline and reduces GPU requirements.

---

## Design Decisions

### What V1 Is

- **Pre-baked 4DGS playback** of a complete show (performer + environment + all VFX)
- **12 simultaneous foveated viewpoints** rendered server-side from the same pre-computed scene
- **Wireless delivery** via Wi-Fi 7 to Steam Frame headsets
- **On-device AI upscaling and temporal reprojection** on the Steam Frame's Snapdragon 8 to achieve high frame rates and mask latency
- **Semi-transparent bounding volumes** for co-present users (spatial awareness only, not volumetric reconstruction)
- **WFS spatial audio + haptic floor** synchronized to the visual playback

### What V1 Is Not

- No real-time ray tracing or VFX computation
- No real-time environment generation
- No interactive elements (no user-triggered effects)
- No AI-driven performer behavior
- No volumetric reconstruction of users — just tracked positions rendered as simple ghost-like indicators

---

## The Two Hard Problems

### 1. Offline Content Production

**Goal:** Noise-free, 40+ dB PSNR 4DGS with fully integrated VFX at room scale.

The entire show — performer captured volumetrically, environments designed and authored, lighting baked, VFX (steam, particles, volumetric light) composited — must be processed into a single flawless 4DGS asset. There is no time constraint on this process; compute and iteration time are unlimited.

**Pipeline:**
1. **Capture** — 64-96 synchronized cameras, 4K+, global shutter, 30 FPS
2. **Reconstruction** — 4DGS processing (evaluate multiple methods: DBS/UBS-7D, Anchored 4DGS, Hybrid 3D-4DGS, etc.)
3. **VFX integration** — Lighting, environmental effects, particles baked into the Gaussian representation. Tools: OctaneRender 2026 (path-traced relightable GS), GSOPs (Houdini), Nuke 17.0
4. **Quality assurance** — Anti-aliasing (AA-2DGS, AAA-Gaussians), floater elimination (StopThePop), noise reduction. Target: zero visible artifacts from any viewpoint within the 200 m² usable area
5. **Compression** — PCGS progressive compression for adaptive streaming. Output in glTF KHR_gaussian_splatting format for standardization

**Quality targets:**
- PSNR: 40+ dB (cinema-grade, indistinguishable from source)
- Zero floaters, zero popping, zero aliasing artifacts
- Temporal consistency across all frames (no flickering or swimming)
- View consistency across the full 200 m² viewing volume (no view-dependent artifacts)

### 2. Real-Time Delivery Chain

**Goal:** <20ms motion-to-photon latency for 12 simultaneous viewers at 144 Hz.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VENUE SERVER ROOM                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────┐                             │
│  │ GPU Cluster (2-4 Rubin GPUs for V1)         │                             │
│  │                                              │                             │
│  │  4DGS Scene Memory (full show in HBM4)      │                             │
│  │  ↓                                           │                             │
│  │  12x Foveated Viewpoint Renderer             │                             │
│  │  (eye tracking data → gaze-contingent splat) │                             │
│  │  ↓                                           │                             │
│  │  12x Foveated Frame Encoder                  │                             │
│  │  + Gaussian metadata (depth, motion vectors) │                             │
│  └──────────────┬──────────────────────────────┘                             │
│                 │ ~150 Mbps × 12 = 1.8 Gbps total                            │
│                 │ (~22% of Wi-Fi 7 practical capacity)                        │
└─────────────────┼────────────────────────────────────────────────────────────┘
                  │ Wi-Fi 7 (~3ms)
                  │
┌─────────────────┼────────────────────────────────────────────────────────────┐
│                 ▼        PERFORMANCE ROOM (300 m², 200 m² usable)           │
│                                                                              │
│  ┌─────────────────────────────────────────────┐                             │
│  │ Steam Frame Headset (×12)                    │                             │
│  │ Snapdragon 8 | 2160×2160/eye | 144Hz        │                             │
│  │                                              │                             │
│  │  Decode MV-HEVC frame          (~1ms)        │                             │
│  │  ↓                                           │                             │
│  │  SGSR 2 temporal upscale 2×    (~1ms)        │                             │
│  │  (Qualcomm native Adreno, reduced ratio)     │                             │
│  │  ↓                                           │                             │
│  │  Mob-FGSR frame generation     (~2ms)        │                             │
│  │  (non-neural, uses GS-LK motion vectors)     │                             │
│  │  ↓                                           │                             │
│  │  GS-aware reprojection         (~1.5ms)      │                             │
│  │  (depth warp + partial re-splat of ~20K GS   │                             │
│  │   for disocclusion fill. Runs at 144Hz,      │                             │
│  │   decoupled from server render rate.)        │                             │
│  │  ↓                                           │                             │
│  │  Composite user bounding volumes (~0.5ms)    │                             │
│  │  ↓                                           │                             │
│  │  Display @ 144Hz                             │                             │
│  └─────────────────────────────────────────────┘                             │
│                                                                              │
│  ┌─────────────────────────────────────────────┐                             │
│  │ WFS Speaker Array (1,200+ drivers)           │                             │
│  │ Haptic Floor (96-128 transducers)            │                             │
│  │ Synchronized to 4DGS playback timeline       │                             │
│  └─────────────────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Latency budget (research-validated):**

| Stage | Target | Validated By | Notes |
|-------|--------|-------------|-------|
| Server-side 4DGS render (foveated) | ~1.5ms | RTGS (7.9x speedup), Fov-GS (11.3x) | Hierarchical GS subsets with gaze-contingent LOD. 1000+ FPS baseline × foveation savings. |
| Server-side encode | ~1ms | MV-HEVC via NVENC | At 150 Mbps/headset, foveal encoding is perceptually lossless (~45–47 dB PSNR). 27% stereo bitrate savings. Split-frame encoding for sub-frame latency. |
| Wi-Fi 7 transmission | ~3ms | Wi-Fi 7 MLO benchmarks | 1.8 Gbps total across 12 headsets (~22% of practical Wi-Fi 7 capacity). Sub-5ms confirmed; 93% reduction in 90th-percentile latency. Dedicated 6GHz link per Steam Frame. |
| On-device decode | ~1ms | Snapdragon 8 Gen 3 HW decoder | Hardware MV-HEVC decoding on Adreno 750. |
| AI upscale | ~1ms | SGSR 2 (Qualcomm) | With 150 Mbps bitrate, server renders at higher resolution → reduced upscaling ratio (2× instead of 4×). Less reconstruction burden, higher output fidelity. |
| Frame generation | ~2ms | Mob-FGSR | Non-neural frame generation: 1.9–2.2ms on Snapdragon 8 Gen 3. Higher-bitrate source frames give the generator better input, reducing interpolation artifacts. |
| GS-aware reprojection | ~1.5ms | GS-LK motion vectors + depth warp | Analytical warp fields from Gaussian primitives (no neural network). Depth-aware warping + partial re-splatting for disocclusion fill. |
| User bounding volumes | ~0.5ms | — | Simple semi-transparent shapes at tracked positions. |
| Display scanout | ~3ms | — | 144 Hz panel refresh. |
| **Total** | **~14.5ms** | | **Under 20ms target with ~5ms margin** |

**Effective perceived latency:** With GS-aware reprojection running at display refresh rate (144 Hz), head motion is compensated locally every 6.9ms. The user perceives reprojection latency (~1.5ms), not the full server round-trip. **Effective motion-to-photon: ~3–5ms** — comparable to native local rendering.

**Key innovation — GS-aware temporal reprojection:**

Standard VR reprojection (ATW/ASW) warps the image based on the new head pose, but creates disocclusion holes where previously hidden content is revealed. With standard video streaming, you can only fill these with crude inpainting.

**No published implementation of GS-aware ATW/ASW exists.** This is a novel IP opportunity for vZero. The closest published work:

- **GS-LK (ICLR 2025):** Derives analytical warp fields and motion vectors directly from Gaussian primitive parameters — no neural network required. This provides the mathematical foundation: each Gaussian's position, covariance, and opacity yield exact per-pixel depth and motion vectors that can be transmitted as lightweight metadata alongside the rasterized frame.

- **VR-Splatting (Bernhard et al.):** Demonstrates peripheral reprojection using GS depth data in a VR context. Renders full quality in the foveal region and reprojects from a previous frame in the periphery using depth-aware warping. Achieves 10.9ms/frame at VR resolution — but does not implement full ATW/ASW-style head motion compensation.

vZero's approach combines and extends both. Because the server streams Gaussian metadata (depth map, analytical motion vectors from GS-LK) alongside the rasterized frame, the Snapdragon 8 can:

1. **Depth-aware warping** — Use the per-pixel depth map from the Gaussian render for 2.5D reprojection (not just 2D image warp). Depth discontinuities identify exact disocclusion boundaries.
2. **Analytical motion prediction** — GS-LK motion vectors predict where each pixel will be at the next head pose, enabling forward warping without neural inference.
3. **Partial re-splatting for disocclusion fill** — The server sends a sparse buffer of 10–30K key Gaussians (depth-edge and boundary primitives). The Snapdragon 8's Adreno 750 re-splats these in <3ms to fill holes revealed by head motion — validated by on-device GS rendering benchmarks showing 50–100K Gaussians feasible at 144 Hz.
4. **IMU-driven pre-compensation** — Predict future head pose from IMU data (available 2–3ms ahead of display) and pre-warp the reprojected frame.

This transforms reprojection from a 2D image operation into a 2.5D/3D operation with principled disocclusion handling. The result: **the user perceives ~3–5ms effective motion-to-photon latency**, because reprojection runs at the full 144 Hz display rate, decoupled from the server render rate.

**Perceptual safety margins:** Research confirms 50–70ms eye-to-photon latency is tolerable for foveated rendering before users perceive the foveal shift. Saccadic suppression provides additional ~100–200ms windows during rapid eye movements when visual perception is naturally reduced. vZero's ~3–5ms effective latency operates well within these thresholds.

---

## Bandwidth Allocation & Image Quality

### Wi-Fi 7 Capacity in the Venue

4 enterprise APs, dedicated 6 GHz band, 320 MHz channels. Practical throughput per AP after overhead: ~2–3 Gbps. With 3 headsets per AP: ~700 Mbps–1 Gbps available per headset. Conservative floor accounting for RF interference from 12 moving bodies and the WFS array: **~500 Mbps per headset.**

### Allocation: 150 Mbps per headset

Total: 1.8 Gbps across 12 headsets — **~22% of practical Wi-Fi 7 capacity.** This leaves ~78% headroom for retransmissions, burst traffic, GS metadata sideband, and future V2/V3 bandwidth needs.

| Bitrate | Foveal PSNR (HEVC) | Quality Level |
|---------|-------------------|---------------|
| 30 Mbps | ~38–40 dB | Very good |
| 80 Mbps | ~42–44 dB | Excellent — approaching transparent |
| **150 Mbps** | **~45–47 dB** | **Perceptually lossless at VR pixel density** |
| 250 Mbps | ~48+ dB | Exceeds display panel capability |

At 150 Mbps with foveated bitrate allocation, the foveal region (~10% of frame area) receives the majority of bits — an effective foveal bitrate equivalent to 500+ Mbps for a full frame. At this level, **HEVC encoding becomes perceptually transparent**: a trained observer in a controlled A/B test cannot distinguish the encoded stream from the uncompressed source.

### Impact on the On-Device Pipeline

The higher bitrate changes the quality equation at every downstream stage:

1. **Encoding ceases to be the quality bottleneck.** At 150 Mbps, the foveal PSNR (~45–47 dB) exceeds the 40+ dB target of the pre-baked source. The codec preserves the full quality of the offline production pipeline.

2. **Server renders at higher resolution.** With more bandwidth to transmit, the server renders the foveal region at higher resolution before encoding. The on-device upscaling ratio drops from ~4× to ~2×, meaning SGSR 2 reconstructs less and preserves more. Upscaling artifacts during fast motion are substantially reduced.

3. **Frame generation gets better input.** Mob-FGSR generates intermediate frames from server-rendered keyframes. Higher-quality keyframes (less compression artifact, higher resolution) produce cleaner interpolated frames with fewer boundary artifacts.

4. **Reprojection artifacts are reduced.** The depth map and motion vectors transmitted alongside the frame are more precise at higher bitrate (less quantization noise), improving the accuracy of the 2.5D warp and disocclusion detection.

### Delivered Image Quality

| Region | Delivered PSNR | Perceptual Quality |
|--------|---------------|-------------------|
| **Foveal center** (~2°) | ~45–47 dB | Perceptually lossless — indistinguishable from uncompressed source |
| **Parafoveal** (~5°) | ~42–44 dB | Excellent — no visible artifacts under normal viewing |
| **Near periphery** (~15°) | ~36–40 dB | Good — intentionally reduced, below visual acuity threshold |
| **Far periphery** (>15°) | ~30–34 dB | Adequate — heavily reduced, well below peripheral acuity |

**The 40+ dB quality achieved in offline production arrives intact in the foveal region.** The delivery chain is no longer the quality bottleneck. The remaining visible compromises are limited to:
- Slight softening during very fast head rotation (reprojection/frame generation transients, ~100–200ms duration, masked by saccadic suppression)
- Disocclusion edges during lateral movement (filled by sparse Gaussian re-splatting — good but not identical to the full render)

These are sub-perceptual under normal viewing conditions.

---

## Server Hardware (V1)

| Component | Spec | Rationale |
|-----------|------|-----------|
| **GPUs** | 2-4 NVIDIA Rubin | Pre-baked playback + 12 foveated renders. At 1000+ FPS/GPU with TC-GS, 2 GPUs may suffice. 4 for headroom. |
| **GPU Memory** | 288 GB HBM4 per GPU | Entire show fits in memory — no streaming from storage during playback |
| **Networking** | 100 GbE internal | GPU-to-encoder fabric |
| **Wi-Fi 7 APs** | 4× enterprise APs | 3 headsets per AP. Redundancy for RF interference. |
| **Encoder** | MV-HEVC via NVENC | 12 simultaneous foveated stereo streams. MV-HEVC provides 27% bitrate savings for stereo pairs. Split-frame encoding for sub-frame latency. |
| **Storage** | NVMe array | Show loading (not real-time streaming — entire show loads to GPU memory at startup) |
| **CPU** | Standard server | Orchestration, tracking aggregation, audio sync |
| **Cooling** | Liquid cooling loop | 2-4 Rubin GPUs generate significant heat |

**Built for V3:** The server room is provisioned for 8-12 GPU slots, full power and cooling capacity, even though V1 only populates 2-4 slots. Upgrading to V2/V3 is a GPU insertion, not a room rebuild.

---

## User Co-Presence

Each Steam Frame reports its 6DOF position via inside-out tracking. The server aggregates all 12 positions and includes them in each foveated frame as simple geometry:

- **Semi-transparent bounding volume** (capsule or soft glow) at each tracked user position
- **Purpose:** Prevent collisions, maintain spatial awareness, contribute to the shared experience feeling
- **Rendering cost:** Negligible — 11 simple transparent shapes per viewpoint
- **No volumetric capture of users required**
- **No room-mounted cameras required for user reconstruction**

Users know others are there. They can see ghostly silhouettes moving around. But the focus is the content, not the other viewers.

---

## Phased Upgrade Path

The architecture is designed to evolve without rebuilding:

### V1 — Playback (Launch)

| Aspect | Implementation |
|--------|---------------|
| Content | Pre-baked 4DGS + VFX |
| Server compute | 4DGS playback + 12 foveated renders |
| Interactivity | None (fixed timeline) |
| User representation | Tracked bounding volumes |
| GPU requirement | 2-4 Rubin |

### V2 — Interactive VFX (Post-Launch)

| Aspect | Implementation |
|--------|---------------|
| Content | Pre-baked 4DGS base + real-time VFX layer |
| Server compute | V1 + Unreal Engine overlay compositing |
| Interactivity | User-triggered effects (hearts, light beams, particle systems) |
| User representation | Tracked bounding volumes (possibly upgraded to stylized avatars) |
| GPU requirement | 4-6 Rubin |
| New capability | Depth-correct occlusion between 4DGS and real-time VFX via Interactive Overlay SDK |

### V3 — AI-Driven Performer (Future)

| Aspect | Implementation |
|--------|---------------|
| Content | AI-modified 4DGS stream responding to users |
| Server compute | V2 + real-time 4DGS generation/deformation via AI |
| Interactivity | Performer responds to speech, looks at users, improvises |
| User representation | Potentially upgraded to volumetric (research-dependent) |
| GPU requirement | 8-12 Rubin |
| New capability | NVIDIA Lyra-class models deforming/generating 4DGS in real-time conditioned on user input |

**Physical infrastructure (power, cooling, rack space, Wi-Fi APs) is built once for V3 capacity.** Only GPUs and software change between versions.

---

## Implications for Original Architecture Documents

### Changes from Technical Architecture (summaries/2-technical-architecture.md)

| Original | Updated (V1) |
|----------|-------------|
| DBS/UBS-7D as sole reconstruction method | Evaluate multiple methods during Phase 1 (DBS/UBS-7D, Anchored 4DGS, Hybrid 3D-4DGS) |
| ~4 MB/frame streaming target | Foveated streaming at ~150 Mbps per viewpoint (perceptually lossless in fovea). 1.8 Gbps total, ~22% of Wi-Fi 7 capacity. |
| 8-12 Rubin GPUs per venue | 2-4 for V1 (playback), scaling to 8-12 for V3 (AI-driven) |
| Real-time interactive overlay (Unreal Engine) | Deferred to V2. V1 is pure pre-baked playback. |
| Custom compression pipeline | Adopt PCGS progressive compression + glTF KHR_gaussian_splatting standard format |
| Home version: PC-rendered degraded | Home version: server-streamed (GIFStream/AirGS) or local rendering on capable hardware |

### Changes from Execution Roadmap (summaries/4-execution-roadmap.md)

| Original Deliverable | Updated |
|----------------------|---------|
| 2.2: Interactive Overlay SDK (M17) | Deferred to post-launch (V2). Not on V1 critical path. |
| 2.3: Content authoring toolchain v1 (M18) | Focus on 4DGS + VFX baking pipeline, not interactive scripting |
| Gate 1 (M11): Multi-viewer rendering proof | Now specifically: 12 foveated viewpoints from pre-baked 4DGS on 2-4 GPUs |
| Phase 1 tech prototype | Must also prove: on-device AI upscaling + GS-aware temporal reprojection on Steam Frame |

### New Critical Path Item

**On-device Snapdragon 8 pipeline** — The AI upscaling and temporal reprojection running on the Steam Frame's chipset is novel engineering. This should be prototyped in Phase 1 (M5-M11) alongside the server-side rendering proof. If the on-device pipeline cannot achieve <8ms total (decode + upscale + reproject), the latency budget breaks.

---

## Research Findings

### 1. GS-Aware Temporal Reprojection — NOVEL IP OPPORTUNITY

**Finding:** No published implementation of GS-aware ATW/ASW exists. This is genuinely novel.

The two closest works are GS-LK (ICLR 2025), which provides the analytical motion vector foundation, and VR-Splatting, which demonstrates GS-depth-aware peripheral reprojection. Neither implements full head-motion-compensating temporal reprojection using Gaussian metadata. vZero would be the first to combine GS-LK analytical warp fields with ATW/ASW-style reprojection and partial re-splatting for disocclusion fill.

**Recommendation:** Pursue patent filing on GS-aware temporal reprojection during Phase 1. The combination of (a) GS-derived analytical motion vectors, (b) depth-aware 2.5D reprojection, and (c) sparse Gaussian re-splatting for disocclusion fill represents patentable novel art.

### 2. On-Device Partial Re-Splatting — FEASIBLE

**Finding:** The Snapdragon 8 Gen 3's Adreno 750 can render 50–100K Gaussians at 144 Hz based on published mobile GS benchmarks (3DGS.zip, Mini-Splatting, MobileR2L). For disocclusion fill, 10–30K Gaussians are sufficient, achievable in <3ms.

Key enablers:
- **Mob-FGSR:** Non-neural frame generation in 1.9–2.2ms on Snapdragon 8 Gen 3. Generates intermediate frames using motion vectors — directly applicable to GS-derived motion data.
- **SGSR 2 (Qualcomm):** Temporal upscaling in ~1ms on Adreno GPUs, natively optimized.
- **3DGS.zip:** 10.6× model compression reduces memory footprint for the sparse Gaussian buffer that rides alongside each frame.

**Practical budget:** The server sends a sparse set of ~20K depth-boundary Gaussians per frame (~200 KB compressed). These are the Gaussians most likely to be revealed during head motion. The Adreno 750 re-splats them in ~2ms to fill disocclusion holes.

### 3. Foveated GS Rendering — 2–11× SPEEDUP CONFIRMED

**Finding:** Multiple published methods demonstrate substantial foveated GS rendering gains:

| Method | Speedup | Approach |
|--------|---------|----------|
| **RTGS** (Hasselgren et al.) | 7.9× | Hierarchical GS subsets — pre-computes LOD tree, renders fewer Gaussians in periphery. Ideal for vZero because LOD hierarchy is computed offline (pre-baked). |
| **Fov-GS** | 11.33× | Gaussian forest structure with per-region LOD selection. Highest speedup but requires custom data structure. |
| **VRSplat** | 2.1× | Single-pass foveated rasterizer with variable Gaussian density. Simplest to implement. |
| **VR-Splatting** | ~2× | Peripheral reprojection from previous frame + full foveal render. |

**Recommendation for vZero V1:** Use **RTGS hierarchical subsets** as the server-side approach. The LOD hierarchy is computed once during offline production (fits the pre-baked pipeline). At render time, the server selects the appropriate subset of Gaussians per screen region based on each viewer's gaze position. Foveal region gets the full Gaussian set; periphery gets progressively sparser subsets. This is natively compatible with the pre-baked content model.

**Pixel savings:** 50–72% reduction in rendered pixels is standard across foveated approaches, with no perceptual degradation. Color channels can be degraded more aggressively than luminance in the periphery (human CSF properties).

### 4. Multi-User Foveated Batching — OPEN PROBLEM, TRACTABLE

**Finding:** No published work addresses multi-user foveated GS rendering from a shared scene. This is an open research problem. However, the architecture is tractable:

**What can be shared across 12 viewpoints:**
- Gaussian culling against the room's bounding volume (one pass, shared)
- Scene decompression / memory layout (one copy in HBM4)
- Temporal frame data (same time step for all viewers)

**What must be per-viewpoint:**
- Depth sorting (view-dependent — the most expensive operation)
- Foveal region selection (each viewer has different gaze)
- Final rasterization

**Key optimization — Neo (ASPLOS '26):** Achieves 10× sorting throughput via temporal reuse of Gaussian ordering between frames. For 12 viewers in the same scene at the same time step, sorting results from one viewpoint can seed the sort for nearby viewpoints, reducing the per-viewer sorting cost. This is especially effective when viewers are clustered in a 200 m² space (viewpoints are relatively similar compared to arbitrary camera positions).

**Estimated server load:** With RTGS foveated rendering (7.9× speedup) and Neo sorting optimization (10× throughput), 12 foveated viewpoints from a single 4DGS scene should require 2 Rubin GPUs, with 4 providing comfortable headroom.

### 5. Eye Tracking → Foveated Render Latency — GENEROUS BUDGET

**Finding:** The perceptual budget is much more forgiving than initially feared.

- **Steam Frame eye tracking latency:** 8–12ms (inside-out tracking with IR cameras)
- **Perceptual threshold for foveal shift detection:** 50–70ms eye-to-photon
- **Saccadic suppression window:** ~100–200ms during rapid eye movements where visual perception is naturally suppressed

The total budget from eye movement to updated foveal content:
- Eye tracking: 8–12ms
- Transmit gaze to server: ~3ms (Wi-Fi 7)
- Server re-renders foveal region: ~1.5ms (foveated GS render)
- Encode + transmit + decode: ~5ms
- **Total: ~18–22ms**

This is well within the 50–70ms perceptual threshold. Users will not perceive the foveal shift. Furthermore, **EyeNexus** (combined gaze + bandwidth adaptive foveation) demonstrates 70.9% latency reduction in foveated streaming, suggesting even these conservative numbers have room for optimization.

### 6. Progressive GS Compression for Bandwidth Adaptation — MATURE

**Finding:** Progressive GS compression is production-ready and directly supports the foveated streaming architecture:

- **PCGS:** Progressive coding with base layer + enhancement layers. Foveal region receives all layers; periphery receives base only. Bandwidth adapts per-region.
- **LapisGS:** Layer-wise progressive compression achieving high quality at aggressive compression ratios.
- **GoDe:** Gaussian decomposition for bandwidth-efficient streaming with quality-adaptive LOD.

These methods pair naturally with the RTGS hierarchical subsets: the server sends the full LOD for the foveal region and progressively lower LODs (fewer enhancement layers) toward the periphery, all from the same pre-computed compressed asset.

## Remaining Open Questions

1. **Optimal sparse Gaussian buffer format** — What is the most efficient encoding for the ~20K boundary Gaussians sent alongside each frame for on-device re-splatting? This needs profiling on actual Snapdragon 8 hardware.

2. **Neo sorting reuse across viewpoints** — The temporal sorting reuse in Neo is designed for frame-to-frame coherence within a single viewpoint. Adapting it for cross-viewpoint sorting reuse in a multi-user scenario needs engineering validation.

3. **Steam Frame foveated streaming protocol** — The Steam Frame's native 4-view foveated streaming (2 low-res full FOV + 2 high-res gaze center) needs to be characterized. Can the server inject GS metadata into this protocol, or does vZero need a custom streaming layer?

4. **Thermal envelope on Snapdragon 8** — Running decode + SGSR2 upscaling + Mob-FGSR frame generation + GS-aware reprojection continuously at 144 Hz will generate sustained heat. The thermal throttling behavior of the Steam Frame under this workload needs testing in Phase 1.

---

## Supporting Research

- **State-of-the-art survey:** exploring/md/_state-of-the-art-february-2026.md
- **Original technical architecture:** steam/summaries/2-technical-architecture.md
- **Original execution roadmap:** steam/summaries/4-execution-roadmap.md

---

*This document is part of the vZero Strategic Document Suite. It reflects design decisions made on February 21, 2026, informed by the state-of-the-art research survey of the same date.*
