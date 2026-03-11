# vZero: Neural Presence

## Technical Architecture & Production Pipeline

**Classification:** Technical White Paper — Confidential
**Date:** February 2026
**Version:** 1.0

---

## I. Architecture Overview

vZero's technical stack has three domains: **Capture** (producing the volumetric asset), **Venue** (rendering and delivering the experience), and **Home** (degraded-fidelity consumer distribution). Each domain has distinct hardware, software, and performance requirements.

The fundamental architectural decision is **server-side rendering with wireless headset delivery**. The volumetric performer and environment are rendered on GPU racks in the venue's server room and streamed as compressed video frames to untethered VR headsets via Wi-Fi 7. The headsets handle display, tracking, and lightweight local compositing (UI overlays, interactive game elements). This architecture is only viable because the Valve Steam Frame was designed from the ground up for this use case.

---

## II. The Capture Pipeline

### 1. The Volumetric Stage

The capture facility is a controlled environment where artists perform for recording into 4D Gaussian Splatting format.

**Camera Array:**
- **64–96 synchronized cameras** in a hemispherical or cylindrical arrangement
- Resolution: 4K minimum per camera (8K preferred for close-up detail)
- Global shutter required (rolling shutter produces temporal artifacts in multi-view reconstruction)
- Frame rate: 30 FPS capture (interpolated to 60–90 FPS in post via temporal models)
- Synchronization: Hardware genlock (sub-frame accuracy across all cameras)

**Why 64–96 cameras?** Current production volumetric studios operate at 32 cameras (4DViews HOLOSYS+) to 106 cameras (Metastage/MRCS). 4D Gaussian Splatting is fundamentally more data-efficient than mesh-based reconstruction — modern regularization techniques enable robust reconstruction from sparser camera arrays than legacy systems required. A 64–96 camera array provides sufficient angular coverage for high-quality 4DGS while remaining within proven production engineering bounds.

**Lighting:**
- Full Light Stage capability: programmable LED sphere surrounding the capture volume
- Enables per-frame environment lighting bake into the Gaussian kernels
- The captured 4DGS asset inherits physically accurate lighting, reducing the need for real-time relighting at playback

**Audio Capture:**
- 32-channel ambisonic microphone array capturing the artist's vocal field from all angles
- Separate close-mic and room-mic feeds for the spatial audio mix
- The ambisonic capture is processed into Wave Field Synthesis "sound objects" in post-production

### 2. The 4DGS Processing Pipeline

Raw multi-view video is processed into a streamable 4D Gaussian Splatting representation through a multi-stage pipeline.

**Stage 1: Multi-View Calibration & Feature Extraction**
- Camera intrinsics/extrinsics calibrated via structure-from-motion (COLMAP or equivalent)
- Per-frame feature extraction for Gaussian initialization

**Stage 2: Beta Kernel Reconstruction**
- We use the **Deformable Beta Splatting (DBS)** framework rather than standard Gaussian kernels
- Beta kernels provide bounded support (compact spatial extent) versus Gaussian kernels (infinite tails). This produces sharper edges on performers — critical for close-up viewing in VR
- DBS achieved SIGGRAPH 2025 publication and has open-source code availability
- The 7-dimensional extension (UBS-7D) adds temporal and view-dependent dimensions, currently achieving state-of-the-art quality at 32.22 dB PSNR on the Neural 3D Video benchmark

**Stage 3: Temporal Compression & Streaming Preparation**
- **Keyframe-guided streaming** based on the Instant Gaussian Stream (IGS) architecture
- Full reconstruction on keyframes (every 10–30 frames); intermediate frames use motion-predicted deltas via AGM-Net
- Per-frame storage target: ~4 MB (achievable with current 4DGS-1K compression techniques, which demonstrate 41x storage reduction)
- A 20-minute show at 30 FPS = ~36,000 frames = ~144 GB uncompressed, ~3.5 GB with temporal compression

**Stage 4: Interactive Layer Authoring**
- The 4DGS performer asset is a passive recording — it does not respond to the audience
- Interactive elements (environmental reactions, game mechanics, audience-responsive lighting) are authored separately in a real-time engine (Unreal Engine 5 or equivalent)
- The interactive layer is rendered locally on the Steam Frame's Snapdragon 8 Gen 3 and composited over the server-streamed 4DGS feed
- Occlusion between the interactive layer and the 4DGS performer is handled via depth buffer sharing between the server and headset

**Production timeline per show:** 3–4 months of capture sessions (multiple performances, wardrobe changes, angles), 4–6 months of 4DGS processing and quality iteration, 3–4 months of interactive layer and spatial audio authoring. Total: 12–16 months from first capture to deliverable asset.

---

## III. The Venue Rendering Architecture

### 1. The Server Rack

Each 2-room Flagship is powered by a central edge-compute cluster.

**Hardware (2028 deployment):**

| Component | Specification | Quantity | Notes |
|-----------|--------------|----------|-------|
| GPU Nodes | NVIDIA Rubin R200 or Rubin Ultra | 8–12 | 288GB HBM4 per GPU, NVLink 6 interconnect |
| CPU | AMD EPYC or equivalent server-class | 4 | Host processing, network management |
| Storage | NVMe SSD array | 20 TB | Full show library loaded to local storage |
| Network | 100GbE internal fabric | — | GPU-to-AP backhaul |
| Cooling | Liquid cooling loop | — | Required for sustained GPU load |

**Why 8–12 GPUs for 24 headsets (12 per room)?** Each viewer requires an independent render — they are in different positions, looking in different directions. However, the 4DGS scene data is shared across all viewers (loaded once into pooled VRAM via NVLink). The per-viewer cost is the rasterization pass, not the data loading. With DLSS 4.5 providing up to 6x frame multiplication, the server renders at 24 FPS per viewer and the headset receives 144 FPS. At 24 FPS per viewer x 24 viewers = 576 FPS aggregate rendering load, distributed across 8–12 GPUs. This is within the performance envelope of Rubin-class hardware based on current Blackwell scaling data.

**Foveated Rendering:** The Steam Frame's 240Hz eye tracking identifies the viewer's gaze point. The server renders the gaze region at full resolution (2160x2160 per eye) and the periphery at 1/4 resolution. This reduces per-viewer rendering cost by approximately 3–4x, making the 24-viewer load feasible.

### 2. Wireless Transport

The server-to-headset link uses Wi-Fi 7 (802.11be) enterprise access points.

**Specifications:**
- 4x enterprise Wi-Fi 7 access points per room (dedicated 6GHz band for VR streaming)
- Real-world per-AP throughput: 2–7 Gbps (verified in 2025 commercial deployments)
- Per-headset bandwidth requirement: ~150 Mbps (H.265/AV1 compressed video at 2160x2160 per eye, foveated)
- 12 headsets x 150 Mbps = 1.8 Gbps total — well within the capacity of 4x Wi-Fi 7 APs
- Latency target: <20ms motion-to-photon (comparable to SteamVR Link performance)

**The Valve Protocol:** The Steam Frame uses Valve's proprietary SteamVR wireless transport, which prioritizes foveated data packets. If network congestion occurs, the gaze-center resolution is maintained while peripheral resolution degrades — a perceptually invisible tradeoff.

### 3. The Steam Frame in Venue Configuration

The retail Steam Frame (Snapdragon 8 Gen 3, 16GB LPDDR5X, 2160x2160/eye, 144Hz) is used with minimal modification for the venue:

- **Ruggedized face gasket:** Sanitizable, swappable between sessions
- **External battery pack:** Belt-mounted to extend session life and reduce headset weight (the retail Frame is 440g with integrated battery)
- **Software lock:** Venue units run a locked SteamOS configuration with vZero's streaming client, disabling consumer features

No custom PCIe module or hardware modification is required. Audio-visual synchronization is achieved through network time protocol (NTP) synchronization between the rendering server and the WFS audio processor, both on the same local 100GbE fabric. The Steam Frame's built-in Wi-Fi 7 transport introduces <20ms latency; the WFS processor buffers audio by the same margin to maintain temporal coherence.

---

## IV. The Audio & Haptic Infrastructure

This is the venue's physical differentiator — the component that cannot be replicated at home.

### 1. Wave Field Synthesis (WFS) Array

**Technology:** Wave Field Synthesis creates "holographic" sound sources in physical space. Unlike channel-based audio (5.1, 7.1, Atmos), WFS calculates the wavefront needed to make sound appear to originate from any arbitrary point in the room. If the virtual performer walks to the center of the space, their voice physically originates from that empty point in the air.

**Hardware:**
- **1,200–1,500 discrete small-format drivers** embedded in the room's walls and ceiling (300 m² room with ~70m of wall perimeter requires higher driver density than smaller installations)
- **Holoplot X1 Matrix Array** or equivalent (Holoplot operates the Sphere's 1,900-array, 157,000-channel system)
- **Dedicated Spatial Audio Processor (SAP):** 2x Dante-enabled multicore processors calculating per-driver wavefronts in real-time
- **Cost estimate:** $3M–$5M per room (based on Holoplot commercial installation scaling from the Sphere's system, sized for 300 m²)

**Performance:**
- Sound source positioning accuracy: sub-10cm
- Frequency response: 80Hz–20kHz (WFS is effective above ~80Hz; sub-bass is handled by the haptic system)
- Latency: <5ms from audio processor to driver output

### 2. Infrasonic Haptic Floor

**Hardware:**
- 96–128 electromagnetic transducers (e.g., Powersoft Mover) embedded in a modular floor grid covering 300 m²
- Frequency range: 20Hz–80Hz (below WFS effective range)
- Function: Translates the sub-bass frequencies of the music into physical vibration felt through the audience's feet and body
- At 300 m², the room's fundamental mode drops below 10 Hz — well below the haptic floor's 20 Hz operating range, eliminating problematic standing waves in the sub-bass

**Effect:** The haptic floor provides the "weight" of a live concert. When the kick drum hits, the floor vibrates in sync with the WFS audio. This physical grounding reduces VR motion sickness (by providing vestibular reference) and dramatically increases the sense of presence.

**Cost estimate:** $500K–$800K per room (96–128 transducers + 300 m² modular floor construction + amplification)

### 3. Audio Production Workflow

The spatial audio mix is produced in post-production alongside the 4DGS visual asset:

1. The 32-channel ambisonic capture is decomposed into discrete "sound objects" (vocal, instruments, ambience)
2. Each sound object is positioned in 3D space relative to the 4DGS performer's position at each frame
3. The WFS processor renders these sound objects in real-time, adapting the wavefront to each viewer's position (tracked by the Steam Frame)
4. The haptic floor receives a separate sub-bass feed, synchronized to the WFS output

---

## V. Current State of Technology: Honest Assessment

### What works today (February 2026):

- **4DGS quality** is production-viable for controlled captures. UBS-7D achieves 32+ dB PSNR — visually convincing at arm's length in VR.
- **4DGS rendering performance** is solved. 1000+ FPS on a single GPU (4DGS-1K) means server-side rendering for 12+ viewers is feasible with current hardware, let alone Rubin.
- **The Steam Frame** ships Q2 2026. Its architecture (Wi-Fi 7 streaming, eye tracking, 144Hz) matches our requirements.
- **WFS commercial installations** are proven at scale (the Sphere operates 157,000 channels nightly).
- **Volumetric capture studios** exist commercially (4DViews, Metastage, Dimension Studio) with 32–106 camera arrays.

### What requires development (by Q4 2028):

- **4DGS streaming to multiple simultaneous VR headsets** has not been demonstrated in production. The IGS architecture handles single-viewer streaming at 2.67s per-frame latency. Multi-viewer rendering from a shared scene graph is a systems engineering challenge, not a research problem — but it has not been built.
- **Integration between 4DGS rendering and WFS audio** does not exist as a product. The synchronization between the visual server and the audio processor requires custom middleware.
- **The interactive overlay system** (real-time game elements composited with server-streamed 4DGS) requires a custom SDK bridging the Steam Frame's local rendering with the server's depth buffer.
- **Content production tooling** for authoring 20-minute 4DGS shows (editorial tools, spatial audio mixing for WFS, interactive scripting) does not exist. This is green-field development.

### What is a genuine risk:

- **Wi-Fi 7 reliability** in a 300 m² room with 12 active headsets, 1,200–1,500 active speakers, and 96–128 haptic transducers creating electromagnetic interference. Enterprise Wi-Fi 7 real-world testing has not occurred in this specific RF environment.
- **Motion-to-photon latency** for server-streamed 4DGS via Wi-Fi 7. Valve's target is <20ms. Adding 4DGS rendering overhead to the pipeline may push this above the comfort threshold (~25ms). Foveated transport helps, but this must be measured, not assumed.
- **Content production cost and timeline.** The $10M budget and 12–16 month production timeline are estimates based on scaling from existing volumetric studio economics. A true AAA 4DGS show with interactive elements has never been produced. The first show will almost certainly exceed budget and timeline.

---

## VI. The Home Version: What Degrades

The vZero Home experience delivers the same 4DGS visual asset to Steam Frame owners at home, with significant reductions:

| Dimension | Flagship | Home |
|-----------|----------|------|
| **Visual** | Server-rendered, full resolution, path-traced environment | PC-rendered (RTX 4070+ required) or degraded standalone, simplified environment |
| **Audio** | 1,200+ driver WFS creating physical sound sources | Headset speakers or consumer headphones — binaural HRTF approximation |
| **Haptic** | Infrasonic floor, full-body acoustic pressure | Optional haptic vest (Woojer/Subpac class, ~$250) |
| **Social** | 12 people in shared physical space, passthrough silhouettes | Solo or 2–3 person online session |
| **Interactive** | Collective audience mechanics, spatial voting | Simplified single-player interaction |

The home version is a $29.99 purchase or subscription item. It is deliberately inferior to the Flagship experience — the venue must remain the premium tier. However, the 4DGS visual quality (the core Neural Presence of seeing a lifelike performer at arm's length) is preserved. The degradation is in the physical and social dimensions, not the visual.

---

## VII. Technology Partnerships

### Build vs. Buy

vZero does not need to build a volumetric capture studio or develop a 4DGS rendering engine from scratch. The technology landscape in 2026 provides viable partners for each pipeline stage:

| Pipeline Stage | Build | Buy/Partner | Recommendation |
|---------------|-------|-------------|----------------|
| **Capture Studio** | Build custom 64–96 camera rig | Lease time at 4DViews, Metastage, or Dimension Studio | **Partner initially**, build proprietary studio in Year 2 |
| **4DGS Processing** | Fork UBS-7D / DBS open-source code | License from Gracia AI or 4DViews | **Build on open-source** (DBS/UBS-7D has permissive license, SIGGRAPH-validated quality) |
| **Rendering Engine** | Custom multi-viewer 4DGS renderer | Extend FlashGS or existing engines | **Build custom** (no existing engine handles multi-viewer streaming to VR headsets) |
| **WFS Audio** | — | Holoplot or equivalent | **Buy** (Holoplot is the proven commercial provider) |
| **Content Tooling** | Custom editorial/authoring suite | — | **Build** (nothing exists for this workflow) |
| **Headset** | — | Valve Steam Frame | **Buy** (this is the platform partnership) |

The core engineering investment is the **multi-viewer rendering server** and the **content authoring toolchain**. Everything else can be sourced from established partners.

---

## VIII. Technical Team Requirements

The engineering team required to build the vZero platform (not the content — that is a separate production team):

| Role | Count | Focus |
|------|-------|-------|
| **4DGS Rendering Engineers** | 3–4 | Multi-viewer renderer, FlashGS integration, Rubin optimization |
| **Systems/Infrastructure** | 2 | Server architecture, GPU cluster management, Wi-Fi 7 networking |
| **Spatial Audio Engineers** | 2 | WFS integration, Holoplot API, audio-visual sync middleware |
| **VR/Steam Frame Engineers** | 2 | SteamVR SDK, foveated transport, headset-side compositing |
| **Tools/Pipeline Engineers** | 2–3 | Content authoring tools, editorial workflow, asset management |
| **Interactive/Game Engineers** | 2 | Unreal/Unity overlay, audience mechanics, haptic scripting |
| **Technical Director** | 1 | Architecture ownership, vendor coordination |
| **Total** | **15–18** | |

This team is hired in Phase 0–1 (2026) and ramps through Phase 2 (2027).

---

*This document is part of the vZero Strategic Document Suite. See also: Strategic Vision & Market Opportunity, Business Model & Financial Architecture, and Execution Roadmap.*
