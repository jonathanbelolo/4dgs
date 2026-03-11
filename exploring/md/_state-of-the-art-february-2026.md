# 4D Gaussian Splatting: State of the Art
## February 21, 2026

**Classification:** Research Survey — Confidential
**Purpose:** Inform vZero technical architecture decisions

---

## Executive Summary

The field of 4D Gaussian Splatting has undergone a transformation since our November 2025 snapshot. What was a promising research direction is now a **production-validated technology**: the Superman film (Framestore, February 2026) shipped ~40 shots using 4D Gaussian Splatting — the first major motion picture to do so. Simultaneously, rendering has broken 1,000 FPS for dynamic scenes, compression ratios routinely exceed 100x, the Khronos Group has released a glTF extension for Gaussian Splats (ratification expected Q2 2026), and MPEG/JVET has formally begun exploring Gaussian Splat Coding standardization.

### What Changed Since November 2025

| Area | November 2025 | February 2026 |
|------|--------------|---------------|
| **Production validation** | Research demos only | Superman (192-camera 4DGS), Perfume music video (4DViews HOLOSYS), A$AP Rocky "Helicopter" |
| **Rendering speed** | ~135 FPS (best 4D) | 1,000+ FPS (4DGS-1K), 343 FPS (Disentangled4DGS), 735 FPS (FlashGS + TC-GS) |
| **Compression** | Early results (10-50x) | OMG4 (1000x), MEGA (190x), CodecGS (146x), Light4GS (120x), P-4DGS (90x) |
| **Streaming** | Proof-of-concept | AirGS (Dec 2025): >30 dB PSNR, 50% size reduction; GIFStream (CVPR 2025): 30 Mbps real-time |
| **Standardization** | None | Khronos glTF extension (RC Feb 2026), MPEG Part 45 exploration (CfP May 2026) |
| **Hardware acceleration** | Software-only | TC-GS Tensor Core mapping (4.76x), GauRast hardware mod (23x), AMD Radiance Cores announced |
| **Mobile/VR** | Limited | Mobile-GS: 116 FPS on mobile; VRSplat: 72+ FPS VR; Meta Hyperscape: standalone Quest capture |
| **AI generation** | DreamGaussian4D | NVIDIA Lyra (text/image/video→3DGS/4DGS, Apache 2.0), NeoVerse (1M+ videos trained) |
| **Feed-forward 4D** | Per-scene optimization required | 4D-LRM: <1.5s on A100; Instant4D: 2 min from monocular video; AnySplat: seconds |
| **Capture systems** | Lab setups | Canon 20-camera portable prototype (CES 2026), Evercoast/Depthkit Mavericks 4 (<500ms live) |
| **On-device** | None | Apple SHARP: single image → 3DGS in <1 second on-device; visionOS 26 native support |

---

## 1. Rendering Performance

### Speed Records (Dynamic/4D Scenes)

| Method | Venue | FPS | Resolution | GPU | Key Innovation |
|--------|-------|-----|-----------|-----|---------------|
| **4DGS-1K** | NeurIPS 2025 | >1,000 (N3V) / >2,400 (D-NeRF) | Standard | RTX 3090 | Spatial-Temporal Variation Score pruning; per-frame active Gaussian mask |
| **Disentangled4DGS** | ArXiv Oct 2025 | 343 | 1352x1014 | RTX 3090 | Decoupled temporal/spatial components; avoids 4D matrix calculations |
| **SpeeDe3DGS** | ArXiv Jun 2025 | 13.71x speedup | Standard | — | Temporal Sensitivity Pruning + GroupFlow (shared SE(3) transforms) |
| **Faster-GS** | ArXiv Feb 2026 | 5x faster training | — | — | Optimization improvements extending to 4D |
| **FlashGS + TC-GS** | CVPR 2025 + SIGGRAPH Asia 2025 | 479–736 | Up to 4K | RTX GPUs | Tensor Core alpha-blending (2-4.76x speedup) |

### Hardware Acceleration

**TC-GS (SIGGRAPH Asia 2025):** Maps alpha-blending to matrix multiplication on Tensor Cores. Algorithm-independent, plug-and-play module. 2.03x–4.76x speedup on alpha blending alone.

**GauRast (NVIDIA Research 2025):** Hardware modification to existing GPU triangle rasterizers. 23x speed, 24x energy reduction, only 0.2% die area overhead. Reduces memory traffic by 90%.

**AMD Radiance Cores + Neural Arrays (RDNA 5, ~2027):** Dedicated hardware for ray traversal and ML rendering. Part of AMD/Sony "Project Amethyst" partnership. FSR Radiance Caching scheduled for 2026 production release.

### Current GPU Landscape

| GPU | VRAM | Bandwidth | GS Relevance |
|-----|------|-----------|--------------|
| **NVIDIA RTX 5090** (Jan 2025) | 32 GB GDDR7 | 1,792 GB/s | ~35% faster than 4090; CUDA 12.8+ required |
| **NVIDIA RTX PRO 6000 Blackwell** | 96 GB GDDR7 ECC | 1,792 GB/s | Explicitly cited for "rendering 3D Gaussian Splats"; 2.5x faster AI training vs Ada |
| **NVIDIA Rubin** (mid-2026) | 288 GB HBM4 | TBD | Target for venue servers; can hold entire shows in memory |
| **Apple M5** (Oct 2025) | Unified | — | SHARP model runs on-device; visionOS 26 native GS support |
| **Qualcomm Snapdragon 8 Elite Gen 5** | — | — | Research published on on-device 3DGS avatars |

### VR-Specific Rendering

**VRSplat (SIGGRAPH 2025):** First systematically evaluated 3DGS approach for VR. 72+ FPS on RTX 4090, displayed on Quest 3 at 2064×2272 per eye (stereoscopic). Eliminates popping artifacts and stereo-disrupting floaters.

**Standalone Quest 3 Limits:** ~400,000 Gaussians maximum for acceptable frame rate. Near objects render sharply; backgrounds lose detail. SqueezeMe: 3 animated Gaussian avatars at 72 FPS.

**Foveated Rendering:** VR-Splatting combines foveated radiance field rendering with 3DGS, reducing peripheral rendering cost while maintaining foveal quality.

### Anti-Aliasing

- **AA-2DGS** (NeurIPS 2025): World-space flat smoothing kernel + object-space Mip filter
- **AAA-Gaussians** (ICCV 2025): Correct aliasing filter with efficient 3D culling
- **Multi-Sample AA:** Quadruple subsampling at four offset subpixel locations

---

## 2. Compression & Storage

### Compression Ratios (Ranked)

| Method | Venue | Compression | FPS | Approach |
|--------|-------|------------|-----|----------|
| **OMG4** | ArXiv Oct 2025 | ~1,000x (60%+ model size reduction) | — | Three-stage progressive pruning + Sub-Vector Quantization for 4D |
| **MEGA** | ICCV 2025 | 190x (Technicolor) / 125x (N3V) | Comparable | Shared AC color predictor + entropy-constrained deformation + FP16 |
| **CodecGS** | ICCV 2025 | 146x | — | Grid-based feature planes compressed via HEVC (Fraunhofer HHI) |
| **Light4GS** | ArXiv Jan 2026 rev | 120x | +20% over baseline | Spatio-Temporal Significance Pruning (64%+ primitives removed) + deep context model |
| **P-4DGS** | ArXiv Oct 2025 | 90x (NeRF-DS) / 40x (D-NeRF) | >260 | Video-compression-inspired spatial-temporal prediction + entropy coding |
| **4DGS-1K** | NeurIPS 2025 | 41x storage reduction | >1,000 | Spatial-Temporal Variation Score pruning |
| **ExGS** | Under review ICLR 2026 | >100x | — | Universal Gaussian Compression + diffusion priors (GaussPainter) |
| **GSICO** | ArXiv Jan 2026 | 20.2x | — | Arranges GS params into structured images, encodes with standard codecs |
| **PCGS** | AAAI 2026 (Oral) | Matches SOTA | — | **First progressive compression** — adaptive bitstream, on-demand quality |

### Video Codec Integration

| Method | Venue | Codec | Speed | Key Innovation |
|--------|-------|-------|-------|---------------|
| **GSVC** | NOSSDAV 2025 | N/A (native 2DGS) | 1,500 FPS @ 1080p | 2D Gaussian splat video with inter-frame prediction |
| **CodecGS** | ICCV 2025 | HEVC | — | Feature planes → HEVC encoding |
| **GSCV** | Under review ICLR 2026 | Various standard | — | Inter-PLAS for enhanced inter-frame performance |
| **LGSCV** | ArXiv Dec 2025 | Standard | 50% faster encoding | Lightweight PLAS sorting replacement |

### File Formats

| Format | Origin | Compression vs PLY | Status |
|--------|--------|--------------------|--------|
| **glTF KHR_gaussian_splatting** | Khronos Group | Standard interchange | Release candidate Feb 2026; ratification Q2 2026 |
| **SPZ** | Niantic/Scaniverse | ~90% (~10x) | MIT license; companion glTF compression extension |
| **SOG/SOGS** | PlayCanvas | ~2-3x vs Compressed PLY | Open source |
| **.ply** | Original | Baseline | Universal but uncompressed |

### Standardization Timeline

- **Khronos glTF Extension (KHR_gaussian_splatting):** Release candidate February 2026. Defines Gaussian splats as glTF mesh primitives (position, orientation, scale, color, opacity). Companion compression extension uses SPZ. Contributors: Autodesk, Bentley, Huawei, Niantic, NVIDIA. Ratification expected Q2 2026.

- **MPEG Part 45 (Gaussian Splat Coding):** Officially in "Exploration" stage at JVET 41st meeting (January 2026). Call for Proposals expected May 15, 2026. Registration August 2026, proposals due January 2027. This is the path toward a formal video coding standard for Gaussian Splats.

---

## 3. Streaming & Delivery

### Streaming Systems

| Method | Venue | Bitrate | Quality | Key Innovation |
|--------|-------|---------|---------|---------------|
| **AirGS** | ArXiv Dec 2025 | 50% per-frame reduction | >30 dB frame-level PSNR | Multi-channel 2D conversion; ILP-based keyframe selection; 6x training acceleration |
| **GIFStream** | CVPR 2025 | 30 Mbps | High quality | Canonical space + deformation field + feature streams; end-to-end compression |
| **DASS** | ArXiv Mar 2025 rev | — | — | Dynamics-aware selective inheritance; distinguishes static/dynamic primitives |
| **Evercoast Cloudbreak** | Commercial | — | Production | <500ms global latency for live streaming volumetric video |
| **Infinite Realities** | Commercial | 100-200 Mbps | Cinema-grade | Long-duration 4DGS content delivery |

### Progressive & Adaptive Delivery

**PCGS (AAAI 2026 Oral):** First progressive compression for Gaussian Splats. Incrementally adds anchors while refining existing ones, enabling on-demand adaptive bitstream. This is critical for network delivery where bandwidth varies.

**Splatter.app:** LOD viewer with instant display via progressive loading. Edge server distribution worldwide.

---

## 4. Capture Systems & Methods

### High-End Production Rigs (50-200+ Cameras)

| System | Cameras | FPS | Output | Production Use |
|--------|---------|-----|--------|---------------|
| **Infinite Realities Deus** | 192 (global shutter, gen-locked, strobe-synced) | 24-48 | ~20M splats/frame (6 GB/frame raw), cropped to ~6M | **Superman** (Feb 2026) — first major motion picture with 4DGS |
| **4DViews HOLOSYS** | 32 or 48 FLIR cameras | 60 | Dual: mesh+texture OR Gaussian splatting | Perfume music video (Crescent Inc. Tokyo) |
| **XIMEA** | 100+ at 12 Mpix | 60 | Raw for processing | Research/production |
| **An Ark (The Shed, NYC)** | 52 cameras | — | Volumetric AR (4 actors) | Immersive theater, reviewers described "uncanny" lifelike presence |

**Superman Production Details (Framestore):**
- 192-camera Deus Capture Stage
- 2-minute takes trained into dynamic splat sequences
- ~20 million splats per frame (6 GB/frame), cropped to ~6 million
- 3 hours processing per close-up frame; 2 weeks total GPU cluster training
- Naturally captures hair, cloth, transparencies
- ~40 shots of holographic characters
- Complete freedom to reposition cameras, change focal lengths, re-stage in post

### Mid-Range Systems (3-20 Cameras)

| System | Cameras | Key Feature | Status |
|--------|---------|------------|--------|
| **Canon 20-Camera Prototype** | 20 synchronized | Dramatic reduction from 100+ camera systems; compact/portable | At ASU MIX Center; CES 2026 demo |
| **Canon Portable Volumetric** | Compact transportable | Simultaneous volumetric + motion capture | CES 2026 announcement |
| **Evercoast/Depthkit Mavericks 4** | 3-100+ | <500ms live streaming; cloud processing; 20x cost reduction vs traditional | Commercial (SaaS pricing) |
| **ScannedReality** | 10 Orbbec Femto Bolt | Affordable professional capture | Commercial |

### Depth Cameras (Azure Kinect Replacements)

| Camera | Resolution | FOV | Accuracy | Price |
|--------|-----------|-----|----------|-------|
| **Orbbec Femto Bolt** | 1MP ToF (1024x1024 @ 15fps) | 120° | <11mm systematic | ~$350-500 |
| **Orbbec Gemini 305** (CES 2026) | Megapixel depth/color | 88x65° | Sub-millimeter @ 15cm | TBD |
| **Orbbec Gemini 345Lg** (CES 2026) | Megapixel | 104x87° | IP67, -20 to 65°C | TBD |

### Sparse-View Methods (3-16 Cameras)

The research community has made dramatic progress in reducing camera requirements:

| Method | Views Needed | Quality | Speed | Venue |
|--------|-------------|---------|-------|-------|
| **InstantSplat** (NVIDIA) | As few as 3 (pose-free) | SSIM 0.3755→0.7624 vs COLMAP | 7.5 seconds (20x faster than COLMAP+3DGS) | NVIDIA Research |
| **DropGaussian** | As few as 3 | Competitive with prior-based methods | Standard training | CVPR 2025 |
| **DropoutGS** | Sparse (varies) | SOTA on Blender, LLFF, DTU | Standard training | CVPR 2025 |
| **SparseGS** | 3 (forward-facing) or 12 (360°) | Good | Fast training, real-time rendering | — |
| **SparseSurf** | Sparse | SOTA on DTU, BlendedMVS, Mip-NeRF360 | — | AAAI 2026 |
| **SV-GS** | Sparse async (e.g., security cameras) | Reconstructs dynamic targets | — | Jan 2026 |

### Monocular / Single-Camera Methods

| Method | Input | Speed | Quality | Venue |
|--------|-------|-------|---------|-------|
| **Apple SHARP** | Single photo | <1 second on-device | LPIPS -25-34%, DISTS -21-43% vs prior best | Open source (Dec 2025) |
| **Instant4D** | Casual monocular video | 2-7 minutes | 92% Gaussian reduction via grid pruning | NeurIPS 2025 |
| **MoDGS** | Slow/static monocular video | Standard | Superior PSNR, SSIM, LPIPS | ICLR 2025 |
| **GFlow** | Monocular, no camera params | 10-20 minutes | Accurate even on water ripples | AAAI 2025 |
| **U-4DGS** | Monocular with occlusions | — | SOTA on ZJU-MoCap, OcMotion | ArXiv Feb 2026 |
| **Deblur4DGS** | Blurry monocular video | — | Handles motion blur | — |
| **Mono4DGS-HDR** | Unposed monocular LDR | — | First 4D HDR from monocular | — |

### Feed-Forward Methods (No Per-Scene Optimization)

| Method | Input | Speed | Quality | Venue |
|--------|-------|-------|---------|-------|
| **AnySplat** | Uncalibrated images (sparse or dense) | Seconds (single forward pass) | Matches pose-aware baselines; 92%+ user preference | SIGGRAPH Asia 2025 |
| **4D-LRM** | Sparse posed views at arbitrary times | <1.5s on A100 | First large-scale 4D reconstruction model | — |
| **DepthSplat** | Multi-view (as few as 2) | Feed-forward | Good | CVPR 2025 |
| **IDESplat** | Multi-view | Feed-forward | +0.33 dB over DepthSplat, 10.7% parameters | Jan 2026 |
| **EcoSplat** | Multi-view | Feed-forward, efficiency-controllable | SOTA with up to 10x fewer primitives | Dec 2025 |

### Cost Comparison

| Approach | Cameras | Hardware Cost | Processing | Quality |
|----------|---------|--------------|------------|---------|
| Infinite Realities Deus | 192 | $500K+ (studio) | Days on GPU cluster | Cinema |
| 4DViews HOLOSYS | 32-48 | $100K+ | 10 hrs/min offline | Production |
| Canon 20-cam prototype | 20 | Research/prototype | Near real-time | High |
| Evercoast/Depthkit | 3-10 Femto Bolt | $3,500-$5,000 (cameras) | Cloud processing | Professional |
| Single iPhone + Polycam | 1 (LiDAR) | $0 (existing phone) | Minutes on device | Consumer |
| Monocular (Instant4D) | 1 video | $0 | 2-7 min (GPU) | Research |
| Feed-forward (AnySplat) | 2-16 photos | $0 | Seconds (GPU) | Research/good |

---

## 5. Tools, Frameworks & Production Pipelines

### Training Frameworks

| Tool | License | Platform | Key Strength |
|------|---------|----------|-------------|
| **gsplat** (Nerfstudio) | Apache 2.0 | CUDA/PyTorch | 4x less memory, published in JMLR Vol. 26 |
| **Nerfstudio** (Splatfacto) | Open source | CUDA | Full pipeline with Splatfacto and Splatfacto-W methods |
| **OpenSplat** | AGPLv3 | NVIDIA/AMD/Metal/CPU | Multi-GPU vendor support; production-grade |
| **LichtFeld Studio** | GPLv3 | CUDA | 2.4x faster rasterization; C++23 + CUDA 12.8+ |
| **GSCodec Studio** | Open source | CUDA | Unified compression research platform for static + dynamic GS |

### Game Engine Integration

| Engine | Plugin | Key Feature | License |
|--------|--------|-------------|---------|
| **Unreal Engine 5** | XScene-UEPlugin (XVERSE) | Full editing, hybrid rendering | Free, open source |
| **Unreal Engine 5** | Volinga Plugin Pro | Relighting, ACES, cast shadows | Commercial |
| **Unreal Engine 5** | Luma AI Plugin | Import captures from iOS | Free |
| **Unity 6+** | UnityGaussianSplatting | 147 FPS / 6.1M splats | Open source |

### VFX / DCC Tools

| Tool | Status | Key Feature |
|------|--------|-------------|
| **OTOY OctaneRender 2026** | Production | First production path-traced relightable Gaussian splats + NRC |
| **Nuke 17.0** (Foundry) | Open beta | Native 3DGS import/rendering; Fields node isolation |
| **GSOPs 2.6** (Houdini) | Free/AGPL | Full edit/relight/mesh/animate in Houdini |
| **Irrealix** | Commercial | After Effects, Nuke, DaVinci Resolve plugins; GPU real-time |
| **Postshot** (Jawset) | Commercial | Desktop GUI; used for Superman VFX; After Effects integration |

### Web Viewers

| Viewer | Platform | Key Feature |
|--------|----------|-------------|
| **SuperSplat** (PlayCanvas) | Browser, AR, VR | Editor + viewer; Quest 2/3, Vision Pro support |
| **Spark** (Three.js) | WebGL2/WebXR | Programmable "Dynos" engine; skeletal animation; SPZ/SOG support |
| **Gauzilla Pro** | Rust/WASM/WebGPU | 4D digital twin capabilities; no CUDA dependency |
| **Visionary** | WebGPU + ONNX | 2-16ms per frame; supports 3DGS, MLP-based, 4DGS, neural avatars |

### Volumetric Video Production Pipelines (Commercial)

| Company | Pipeline | Status | Key Feature |
|---------|----------|--------|-------------|
| **Gracia AI** | Capture → edit → deploy | Commercial ($1.7M funded Dec 2025) | First full-stack 4DGS infrastructure; Unity/Unreal/WebGPU/Quest/Pico |
| **Infinite Realities InfiniteStudio** | 192-cam → 4DGS | Commercial | Cinema-grade; captures skin pores, stray hairs, cloth threads |
| **4DViews HOLOSYS Splatting** | Multi-cam → dual export | Commercial | Simultaneous mesh+texture AND Gaussian splatting output |
| **Evercoast/Depthkit** | 3-100+ cameras → cloud | Commercial (SaaS) | <500ms live streaming; 20x cost reduction |

### NVIDIA Ecosystem

| Tool | Type | Key Feature |
|------|------|-------------|
| **3DGUT/3DGRUT** | Research (open source) | Unscented Transform; fisheye, rolling shutter, reflections, refractions; Vulkan hybrid rendering |
| **vk_gaussian_splatting** | Vulkan sample (open source) | Rasterization + ray tracing + hybrid comparison; memory/VRAM profiling |
| **gsplat** (co-maintained) | Library (Apache 2.0) | 3DGUT integrated; PyTorch bindings |
| **Omniverse NuRec** | Libraries | Ray-traced 3DGS for Physical AI simulation; integrated into CARLA |
| **AlpaSim** | Simulator (open source) | Autonomous driving; powered by 3DGUT |

---

## 6. AI-Powered 4D Generation

### Text/Image/Video → 4D Gaussian Splatting

| System | Input | Speed | Quality | Open Source | Venue |
|--------|-------|-------|---------|-------------|-------|
| **NVIDIA Lyra** | Text, image, or monocular video | Seconds (feed-forward) | SOTA for both 3D and 4D | Apache 2.0 | ICLR 2026 |
| **NeoVerse** | Monocular videos (trained on 1M+) | 2 orders of magnitude faster | Pose-free; handles degradation | Yes | Jan 2026 |
| **4D-LRM** | Sparse posed views at any time | <1.5s on A100 | First large-scale 4D recon model | Project page | — |
| **L4GM** (NVIDIA) | Single-view video | ~1 second | Object-centric 4D | Yes | NeurIPS 2024 |
| **Instant4D** | Casual monocular video | 2-8 minutes | 30x speedup over prior methods | Project page | NeurIPS 2025 |
| **Splat4D** | Text, image, or monocular video | Optimization-based | 4D humans, text-guided editing | Project page | SIGGRAPH 2025 |
| **DreamGaussian4D** | Text or single image | ~5 minutes | Animated objects | Yes | — |

### Key Pipeline: NVIDIA GEN3C → Lyra

NVIDIA has assembled the most complete open-source text/image/video-to-3D/4D pipeline:

1. **GEN3C** (CVPR 2025 Highlight): Generates 3D-consistent video from a single image + camera trajectory. Uses a "3D cache" of point clouds from depth prediction to condition next-frame generation. Built on NVIDIA Cosmos. Apache 2.0.

2. **Lyra** (ICLR 2026): Self-distillation framework that distills implicit 3D knowledge from video diffusion models (GEN3C/Cosmos) into explicit 3DGS representation. No multi-view training data required. Extends to 4D via temporal conditioning. Apache 2.0.

### Feed-Forward 3D Reconstruction

| Model | Input | Speed | Key Innovation | Venue |
|-------|-------|-------|---------------|-------|
| **VGGT** | 1 to hundreds of images | <1 second | CVPR 2025 **Best Paper Award** | CVPR 2025 |
| **Fast3R** | Up to 1,000+ images | 251 FPS | 1,500 views per pass on single A100 | CVPR 2025 |
| **MapAnything** | Images + optional calibration | Single pass | Universal transformer, 12+ reconstruction tasks | 3DV 2026 |
| **Google D4RT** | Video (mono or multi-view) | 5s for 1-min video | 18-300x faster; query-based encoder-decoder | Dec 2025 |

### Gaussian Splatting Avatars

| System | Input | Performance | Key Feature |
|--------|-------|------------|-------------|
| **SplattingAvatar** | Monocular video | 300+ FPS desktop, 30 FPS mobile | Mesh-embedded Gaussians |
| **GAvatar** (NVIDIA) | Text prompts | 100 FPS @ 1K | Text-to-animatable avatar |
| **SqueezeMe** | Multi-view capture | 72 FPS on Quest 3 (3 avatars) | Standalone VR avatars |
| **GASPACHO** | 3D scenes | — | Human-object interaction with contact constraints |

### World Models

- **World Labs Marble:** Generates full 3D worlds as Gaussian splats from text/image/multiple images. Exportable as .ply/.spz. Used with NVIDIA Isaac Sim for robotic simulation.
- **GWM (Gaussian World Model, ICCV 2025):** Predicts future scene states as Gaussian Splatting for model-based reinforcement learning.

---

## 7. VR/XR Platform Status

### Steam Frame (Valve, Q2 2026)
- Wi-Fi 7, 2160×2160/eye, 144 Hz, eye tracking
- Built for wireless server-side rendering
- No confirmed GS-specific performance data yet
- **Status:** Dev kits available; public launch Q2 2026

### Meta Quest 3/3S
- **Hyperscape Capture:** Scan real rooms in ~5 minutes on standalone headset; cloud-processed; GS rendered/streamed back to Quest — no PC required
- **Gracia:** First 4DGS volumetric video app on Quest Store
- **Into the Scaniverse:** GS viewing app for Quest
- **Splatara:** Added in-app GS training (Feb 2026) using FastGS and Difix3D+
- **Practical limit:** ~400,000 Gaussians for standalone performance
- **Meta Spatial SDK:** Official documentation for Gaussian splat integration

### Apple Vision Pro (M5, Oct 2025)
- **visionOS 26:** One-click "Spatial Scenes" — any photo becomes volumetric via on-device GS
- **SHARP model:** Open source, <1 second, on-device; 100+ FPS rendering
- **MetalSplatter 1.1:** AR mixed immersive mode via Metal over passthrough
- **Spatial Fields:** Cross-platform GS viewing (Vision Pro, iPhone/iPad, Mac, Apple TV)

---

## 8. Novel Architectures & Representations (2025-2026)

### Dynamic Scene Representations

| Method | Key Innovation | Performance | Venue |
|--------|---------------|-------------|-------|
| **Anchored 4DGS** | 4D anchor-based framework binding Gaussians to strategically distributed anchor points | Storage-efficient + expressive | SIGGRAPH Asia 2025 |
| **Hybrid 3D-4DGS** | Converts temporally invariant Gaussians to 3D; reserves 4D for dynamic elements only | 32.25 dB PSNR; 12 min vs 5.5 hr training | ArXiv May 2025 |
| **HAIF-GS** | Sparse anchor-driven deformation; hierarchical propagation adapts to motion complexity | Significant quality improvement | NeurIPS 2025 |
| **Physics-Informed DGS** | Models Gaussians as Lagrangian particles; recovers intrinsic physical properties | Physically plausible dynamics | AAAI 2026 |
| **FAGS** | Frequency-differentiated kernels; Fourier-Deformation Network for motion blur | Enhanced motion expressiveness | Under review ICLR 2026 |
| **Spike4DGS** | Spike camera arrays (20 kHz) instead of RGB; outperforms event cameras | Superior high-speed scenes | NeurIPS 2025 |
| **Prior-Enhanced GS** | Automatic pipeline enhancing priors for Dynamic GS from casual monocular video | Fully automatic | SIGGRAPH Asia 2025 |

### Ray Tracing + Gaussian Splatting Hybrid

| Method | Key Feature | Venue |
|--------|------------|-------|
| **3DGUT/3DGRUT** (NVIDIA) | Unscented Transform; supports nonlinear cameras, reflections, refractions | CVPR 2025 Oral |
| **HybridSplat** | Baked view-dependent reflections; diffuse via splatting, specular via ray tracing | — |
| **RaySplats** | Pure ray-tracing on confidence ellipses; mesh/lighting integration | — |
| **IRGS** | Inter-reflective GS with 2D Gaussian ray tracing | CVPR 2025 |

---

## 9. Implications for vZero

### What This Means for the Technical Architecture

#### Capture Pipeline
The vZero technical architecture specifies 64-96 synchronized cameras with 4K+ resolution and global shutter at 30 FPS. This is validated by Superman's 192-camera rig, but new options emerge:

1. **Canon's 20-camera prototype** suggests that sparse-view methods + Gaussian splatting may dramatically reduce camera requirements in 18-24 months. For initial production, a full 64-96 camera rig remains prudent, but the capture studio design should accommodate future sparse-view upgrades.

2. **4DViews HOLOSYS** (32-48 cameras, fully transportable) is a strong candidate for the Miraval Weekly Sessions studio — it's already in commercial use and outputs both mesh+texture AND Gaussian splatting from the same capture.

3. **DEGS (Eyeline Labs)** offers an intriguing two-rig approach: a Scene Rig for multi-actor capture and a Face Rig for high-fidelity facial detail, with diffusion-based enhancement. This could enable 4K facial closeups from lower-resolution multi-camera capture.

#### Rendering Engine
The architecture targets 8-12 NVIDIA Rubin GPUs per venue. Current developments strongly support this:

1. **4DGS-1K at 1,000+ FPS** on an RTX 3090 means a single Rubin GPU could render multiple viewpoints simultaneously. The 12-viewer target (each at 2160×2160/eye, 144 Hz) may require fewer GPUs than projected.

2. **TC-GS Tensor Core acceleration** (2-4.76x) is directly applicable to the venue rendering engine and should be integrated from day one.

3. **Hybrid rendering (3DGUT)** enables reflections, refractions, and complex camera effects. For interactive elements composited via Unreal Engine, this bridges the gap between 4DGS performers and real-time environment rendering.

4. **Foveated rendering** combined with the Steam Frame's eye tracking could reduce per-viewer rendering cost by 40-60%, potentially allowing more viewers per room or higher quality.

#### Compression & Streaming
The original architecture targets ~4 MB per frame for wireless streaming. Current compression advances change the calculus:

1. **P-4DGS at 90x compression** with >260 FPS rendering means raw 4DGS frames (which can be large) are compressible to network-friendly sizes while maintaining real-time performance.

2. **AirGS streaming** (December 2025) achieves >30 dB PSNR with 50% per-frame size reduction — purpose-built for free-viewpoint video delivery.

3. **GIFStream** achieves real-time rendering at 30 Mbps — well within Wi-Fi 7 capacity (up to 46 Gbps theoretical). Even accounting for 12 simultaneous streams at 30 Mbps each (360 Mbps total), this is comfortably within a single Wi-Fi 7 AP's capacity.

4. **PCGS progressive compression** enables quality adaptation based on per-viewer network conditions — important for the home Neural Pass subscription where bandwidth varies.

5. **MPEG Part 45 standardization** and the **glTF extension** mean vZero's content pipeline will converge with industry standards rather than being proprietary. This significantly de-risks the technology choice.

#### Content Pipeline
1. **NVIDIA Lyra + GEN3C** (both Apache 2.0) could augment the content pipeline: generate 3D environments from concept art, create background elements, or pre-visualize shows before expensive capture sessions.

2. **AnySplat and feed-forward methods** could accelerate iterative content development — capture a quick take, reconstruct in seconds, iterate on staging and lighting without full pipeline processing.

3. **Postshot** (used for Superman's VFX) is a proven desktop tool for training Gaussian splats from captured footage, with After Effects integration for the editorial pipeline.

4. **OctaneRender 2026** with path-traced relightable Gaussian splats enables the content team to relight 4DGS performers in post — adding creative flexibility beyond what was captured.

#### Home / Neural Pass
1. **Mobile-GS at 116 FPS** on edge devices suggests that future Steam Frame generations (or competitor headsets) may be able to render locally rather than requiring server-side streaming. This doesn't change the venue architecture but opens options for the home subscription.

2. **The glTF standard** means Neural Pass content can potentially be consumed across platforms (Quest, Vision Pro, future headsets) without format conversion — expanding the addressable market beyond Steam Frame owners.

3. **Apple SHARP and visionOS 26** demonstrate that Apple is building native Gaussian splatting into its platform. If vZero content is available in glTF+GS format, Vision Pro users become a potential subscriber base.

#### Risk Mitigation
1. **The M11 gate** (can 4DGS stream wirelessly to a Steam Frame at <25ms latency?) is **substantially de-risked** by AirGS, GIFStream, and the broader streaming research. Multiple independent teams have demonstrated real-time 4DGS streaming.

2. **Visual quality** (>28 dB PSNR target) is routinely exceeded by current methods. 4DGS-1K achieves +0.38 dB above vanilla 4DGS which was already at 32+ dB. Superman proved cinema-quality results.

3. **The DBS/UBS-7D dependency** may be reconsidered. While it remains strong, the proliferation of alternative approaches (Anchored 4DGS, Hybrid 3D-4DGS, Physics-Informed DGS) means the team should evaluate multiple reconstruction methods during Phase 1, not commit to a single pipeline prematurely.

### Recommended Technical Architecture Updates

| Original Spec | Recommended Update | Rationale |
|--------------|-------------------|-----------|
| DBS/UBS-7D exclusively | Evaluate multiple reconstruction methods in Phase 1 | Field has diversified; several methods exceed 32 dB PSNR with better efficiency |
| ~4 MB/frame streaming target | Adopt AirGS or GIFStream architecture (30 Mbps, adaptive quality) | Purpose-built streaming solutions now exist with better quality/bandwidth tradeoffs |
| Custom compression pipeline | Build on PCGS progressive compression + SPZ/glTF formats | Progressive streaming adapts to bandwidth; standards-based formats de-risk IP |
| 8-12 Rubin GPUs per venue | Likely 6-8 sufficient with TC-GS + foveated rendering | 1,000+ FPS per GPU + Tensor Core acceleration + foveation reduces requirements |
| 64-96 capture cameras | Start at 64 for AAA; evaluate 32-48 (4DViews HOLOSYS) for Weekly Sessions | Different quality tiers need different capture density |
| Proprietary content format | Adopt glTF KHR_gaussian_splatting as interchange format | Industry standard imminent; enables cross-platform distribution |
| Home: PC-rendered degraded version | Home: server-streamed via GIFStream/AirGS OR local rendering on capable hardware | Streaming has matured enough for home delivery; local rendering approaching viability |

---

## 10. Key Academic & Industry Events (2026)

| Event | Date | Relevance |
|-------|------|-----------|
| **AAAI 2026** | Feb-Mar 2026 | PCGS (progressive compression), Physics-Informed DGS, SparseSurf |
| **ICLR 2026** | Apr 2026 | Mobile-GS, NVIDIA Lyra, ExGS, FAGS, GASPACHO |
| **MPEG Part 45 CfP** | May 15, 2026 | Gaussian Splat Coding standardization |
| **Khronos glTF Ratification** | Q2 2026 | KHR_gaussian_splatting becomes official |
| **CVPR 2026** | Jun 2026 | NVIDIA 4D Digital Twins workshop; expect major new papers |
| **Steam Frame Launch** | Q2 2026 | First hardware for vZero development |
| **NVIDIA Rubin Launch** | Mid 2026 | Venue server target GPU |
| **SIGGRAPH 2026** | Aug 2026 | Major venue for production-ready GS tools |
| **MPEG Part 45 Registration** | Aug 2026 | Formal proposals for GS coding standard |

---

## Appendix A: All Papers Referenced

### NeurIPS 2025
- 4DGS-1K: 1000+ FPS Dynamic Scene Rendering
- HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting
- Spike4DGS: High-Speed Dynamic Scene Rendering via Spike Camera Array
- Instant4D: Casual Monocular 4D Reconstruction

### ICCV 2025
- MEGA: Memory-Efficient 4D Gaussian Splatting
- CodecGS: Compression with Feature Planes and Standard Video Codecs
- AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
- DiffusionGS: 3D Gaussian Point Cloud Generation via Diffusion
- Gaussian World Model

### CVPR 2025
- FlashGS (with TC-GS extension)
- GIFStream: 4D Gaussian-based Immersive Video with Feature Stream
- DepthSplat
- VGGT (Best Paper Award)
- Fast3R
- GaussianCity
- GEN3C (Highlight)
- 3DGUT/3DGRUT (NVIDIA, Oral)
- IRGS: Inter-Reflective Gaussian Splatting
- SplatAD: Real-Time Rendering for Autonomous Driving
- DropGaussian / DropoutGS
- Generative Sparse-View GS

### SIGGRAPH 2025
- Splat4D (Diffusion-Enhanced 4DGS)
- VRSplat
- NVIDIA NuRec Libraries

### SIGGRAPH Asia 2025
- Anchored 4DGS
- TC-GS: Tensor Core Gaussian Splatting
- AnySplat
- Prior-Enhanced GS for Dynamic Scenes
- DEGS (Detail Enhanced GS)

### AAAI 2026
- PCGS: Progressive Compression (Oral)
- Physics-Informed Deformable GS
- SparseSurf

### ICLR 2026
- NVIDIA Lyra
- Mobile-GS
- ExGS: Extreme Compression with Diffusion Priors
- FAGS: Frequency-Aware Dynamic GS
- GSCV: Video Codec GS Compression
- GASPACHO: Controllable Humans and Objects

### 3DV 2026
- PM-Loss: Improved Depth for Feed-Forward 3DGS
- MapAnything

### Other Notable
- AirGS (ArXiv Dec 2025)
- P-4DGS (ArXiv Oct 2025)
- Light4GS (ArXiv Jan 2026 rev)
- OMG4 (ArXiv Oct 2025)
- Hybrid 3D-4DGS (ArXiv May 2025)
- Disentangled4DGS (ArXiv Oct 2025)
- SpeeDe3DGS (ArXiv Jun 2025)
- Faster-GS (ArXiv Feb 2026)
- EVolSplat4D (ArXiv Jan 2026)
- U-4DGS (ArXiv Feb 2026)
- NeoVerse (Jan 2026)
- EcoSplat (Dec 2025)
- IDESplat (Jan 2026)
- LGSCV (ArXiv Dec 2025)
- GSICO (ArXiv Jan 2026)
- SV-GS (Jan 2026)
- Google D4RT (Dec 2025)

### Production/Industry
- Superman (Framestore + Infinite Realities, Feb 2026)
- Perfume music video (4DViews HOLOSYS + Crescent Inc., 2025)
- A$AP Rocky "Helicopter" (Dynamic GS for nearly every person)
- Usher "Ruin" music video (Postshot)
- An Ark immersive theater (The Shed, NYC)

---

*This document was compiled on February 21, 2026, from web research across academic papers (ArXiv, conference proceedings), industry announcements, press coverage, and product documentation. It is part of the vZero Strategic Document Suite.*
