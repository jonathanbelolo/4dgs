# 4D Volumetric Video Pipeline Research
## Gemini Conversation - March 3, 2026

---

## 1. Rendering Primitives: Gaussians vs Beta Splatting

### Disentangled 4DGS (Feng et al., arXiv:2503.22159)
- **Core idea**: "Projection-first" pipeline instead of "slice-first"
- Projects 4D Gaussian spatial component to 2D first, applies temporal deformation after
- **343 FPS** at 1352x1014 on RTX 3090
- Disentangles spatial params (position, rotation, scale, opacity, color) from temporal params (time scaling, velocity)
- Flow-Gradient Guided Consistency Loss + Temporal Splitting for plausible motion
- ~4.5% storage reduction over 4DGS baselines

### Beta Splatting (Deformable Beta Splatting / Universal Beta Splatting)
- **Papers**: DBS (arXiv:2501.18630, SIGGRAPH 2025), UBS (arXiv:2510.03312, ICLR 2026)
- **Authors**: Rong Liu, Zhongpai Gao, Andrew Feng (USC / United Imaging Intelligence)
- **Code**: github.com/RongLiu-Leo/beta-splatting (Apache 2.0)
- Replaces Gaussian kernel with **Beta kernel** controlled by learnable parameter `b`:
  - b < 0: flat/box-like (hard surfaces, sharp edges)
  - b > 0: peaked (fine details)
  - b = 0: approximates standard Gaussian (backward compatible)
- **Spherical Beta** replaces Spherical Harmonics for color: 31% of SH3 params, better specular highlights
- UBS extends to N-dimensions (time, space, viewing angle)
- **+8.27 dB PSNR** over 3DGS on static scenes, **+2.78 dB** on dynamic scenes
- ~45% fewer parameters than standard 3DGS/4DGS
- Compression via `compress.py` to PNG-based format (5-6x smaller than .ply)

### Other SOTA Methods (2025-2026)
- **EvoGS**: Neural ODE velocity field, best for temporal extrapolation / sparse training data
- **OMG4**: Minimalist, < 10MB per scene, Gaussian pruning/merging
- **Hybrid 3D-4DGS** (ICLR 2026): Freezes static areas as 3D, reserves 4D for moving objects, 3-5x faster training
- **7DGS**: 7-dimensional (x,y,z,t + view-dependent appearance), better for reflections
- **MeshSplatting / Triangle Splatting+**: Returns to triangle meshes, game engine compatible, physics-ready, ~30.5 dB PSNR (lower than splatting but better for production)

### Combining Disentangled + Beta
- Swap Gaussian math for Beta math while keeping Projection-First pipeline
- Beta's bounded nature makes projection even more efficient
- Requires re-deriving equivalence proof for Beta distribution (Taylor expansion or lookup table)

---

## 2. Chosen Approach: Universal Beta Splatting for 6DOF VR

### Why UBS for VR
- Sharp edges even at extreme viewing angles (Beta can flatten into true surfaces)
- N-dimensional kernel integrates view-dependent component natively
- Better temporal precision (sharp movements modeled cleanly)
- 73% fewer parameters than 4DGS
- 5-6x storage reduction with PNG compression

### AI Upscaling Integration
- **During training** (on primitives): SRSplat / SRGS densifies splats from low-res to high-res
- **During playback** (on 2D render): DLSS 4.5 / FSR 4.0 boosts FPS and resolution
- **Frame generation**: Multi-Frame Gen interpolates to 240Hz for headset

### Resolution Scheduling (Training Strategy)
- **Phase 1** (2K): Global structure, SMPL-X skeleton, velocity
- **Phase 2** (4K): Beta shape optimization, edge sharpening, densification
- **Phase 3** (8K): Color/SH only, patch-based (512x512 random patches), geometry locked

---

## 3. Human Performance Optimization

### Parametric Body Models
- **SMPL-X**: Industry standard but linear, low vertex density, joints collapse
- **SUPR**: Successor to SMPL-X, higher vertex count, better neck/jaw/shoulder transitions
- **GHUM/GHUML** (Google): Non-linear (VAE-based), anatomical joint limits, better skin deformation
- **SKEL** (2025/2026): Biomechanically accurate skeleton, models scapula sliding and radius/ulna crossing
- **FLAME 2025**: Updated expressive head model

### Recommended: SKEL skeleton + SUPR surface density
- Use SKEL for movement (biomechanical accuracy, no joint collapse)
- Subdivide SKEL surface to match SUPR vertex density
- Best of both worlds for dancing/singing performers

### Constraining Splats to Body
- **Initialization**: Distribute Beta kernels on parametric mesh vertices (NOT from SfM)
- **Position formula**: `Position = Mesh(vertex) + Delta_offset`
- **Neural Linear Blend Skinning (N-LBS)**: Each splat assigned to SKEL bones with weights
- **Elastic Surface Loss (L_surf)**:
  - Skin: High-penalty L2 distance from SUPR surface
  - Clothing/Hair: Laplacian Smoothness (allows volume, prevents explosion)
- **As-Rigid-As-Possible (ARAP)**: Local splat clusters maintain relative distance/orientation
- **Normal-Alignment**: Splat normals forced to align with mesh normals
- **Bone-Centric Regularization**: Prevents splats jumping to wrong body part
- **Gaussian Shell Maps**: Extrude mesh into thin 3D shell, keep all splats inside

### Constraint Schedule
| Stage | Data | Constraint Strength | Goal |
|-------|------|-------------------|------|
| 1 | 2K | Maximum | Lock to SKEL bones |
| 2 | 4K | Medium | Allow inflation for clothing/hair |
| 3 | 8K | Low spatial / High normal | Lock positions, refine normals |

### Advanced Techniques for Human Performances
- **Dual-Layer Motion**: Rigid (LBS body) + Non-Rigid (MLP for clothing/hair residuals)
- **Adaptive Gabor / Frequency-Aware**: Learnable frequency weights, high-freq for sequins/pores, low-freq for smooth areas
- **Uncertainty-Aware 4DGS**: Weights training by visibility confidence, fills occlusion gaps with temporal redundancy
- **Hierarchical LoD**: Pyramid of splats, dense close-up, sparse far away

---

## 4. Semantic Masking & Material Inference

### Background Removal (3-tier)
1. **SAM 2** (Segment Anything Model 2): Click once per camera, propagates mask through video
2. **SMPL-X/SKEL Geometry Pruning**: Delete splats > 20cm from body surface
3. **Cross-view Consensus**: Use multi-camera consistency to identify foreground vs background

### Semantic Segmentation: Grounded SAM 2
- **Best stack**: Grounding DINO-X + SAM 2
- Text-prompted ("sheer lace trim", "flyaway hair strands")
- Temporal coherence via Memory Attention (no flickering)
- Native 8K support
- Run on center cameras first, propagate to peripheral cameras (40% time savings)

### Automatic PBR Inference
- **SAMa** (Adobe Research): Takes SAM 2 masks, lifts to 3D, applies Material-Specific Priors
- **COREA / GS-IR**: Inverse rendering frameworks for Albedo/Normal/Roughness/Metallic extraction
- **Material-Informed GS**: Semantic label -> PBR property mapping (leather roughness range, skin SSS, etc.)
- **Semantic-Material Loss**: `If Label == "Skin" -> apply SSS_Prior; If Label == "Metal" -> apply High_Metallic_Prior`

### Semantic Scene Decomposition (Stage Capture)
- Layer 1: Performer (4D Beta Splats, Dynamic)
- Layer 2: Floor (3D Static, Reflective PBR)
- Layer 3: Props (3D Static, Diffuse/Metallic)
- Layer 4: Background (Low-density scaffold)
- **Contact Constraint**: Feet splats interact with floor splats (prevents sliding)
- **Hybrid 3D-4DGS**: Auto-freezes static elements, reserves 4D for performer only

---

## 5. Relighting & Environment Compositing

### Inverse Rendering for Relighting
- Decompose 8K pixels into **Albedo, Normals, Roughness** during Stage 2-3
- Each Beta kernel gets: Albedo (rho), Normal (n), Roughness (r), Metallic (m)
- Optimize HDR Environment Map alongside dancer (100-camera coverage helps)
- **Spherical Beta functions** for specular lobes (narrow for sharp glints, wide for matte)
- Pre-calculate **Splat-based Visibility Field** for self-shadowing

### Hybrid Relighting Architecture
- **H100/B300 (server)**: Global Illumination, ray-traced shadows, PBR, SSS
- **Steam Frame (headset)**: Local specular glints only (simplified SG/SB shader)
- Server sends "pre-lit" splats; headset adds view-dependent micro-details
- If network hiccups, headset continues rendering from cached 3D data

### OctaneRender 2026 Integration
- Octane 2026.1 treats splats as **native path-traced primitives**
- Supports Beta kernels via custom vertex attributes + Neural Radiance Cache
- **SPZ format**: ~10x smaller than PLY, preserves Beta fidelity
- OSL shaders map Beta `b` parameter to PBR roughness
- Can calculate GI, caustics, refractions through glass/water
- **Octane 2027 EA**: "Bake-to-Beta" exports CGI assets (trees, water) as Dynamic Beta Splats
- Color bleeding, contact shadows, self-shadowing all computed physically

### Dynamic Environments
- CGI environments converted to splats (unified stream to headset)
- Static elements: 3D Gaussians, cached on headset (sent once)
- Dynamic elements (trees, water, clouds): 4D with Deformation Field
  - Canonical splats + temporal deformation coefficients
  - Transient Splatting for fluids (opacity decay, velocity buffers)
- **Static-Dynamic Decomposition**: 90% bandwidth reserved for dynamic content

### Stage Lighting Handling
- **Live stage (flashing lights)**: Per-frame Light Embedding (latent code l_t), Lumina-4DGS exposure normalization
- **Studio (controlled)**: Uniform flat lighting recommended for best material decomposition
  - Relighting done in CGI (Octane) for infinite re-usability
  - Hybrid compromise: uniform + synchronized color LED tubes for skin light response data

---

## 6. Streaming & Delivery Architecture

### Primitive Streaming (NOT 2D video streaming)
- Stream Beta kernel parameters + deformation deltas to headset
- Headset reconstructs images locally using its GPU
- **Advantages over 2D streaming**:
  - Zero-latency local parallax (< 10ms motion-to-photon vs 30-50ms)
  - View-dependent effects computed from actual eye position
  - Bandwidth-efficient (only send deltas, prioritize face/hands)
  - Rock-solid stability even with network jitter

### Neural Bitstream Encoding
- Stream one "Canonical" Beta model at session start
- Continuous stream of Deformation Field offsets + Beta shape changes
- Wi-Fi 7 MLO: geometry on 6GHz band, color/texture on 5GHz band
- Hierarchical LoD: always maintain base layer, subdivide as packets arrive

### Split-Render Architecture (Server + Headset)
- **Server (B300)**: 4D deformations, Spherical Beta lighting, 8K foveal patches, AI offsets
- **Headset (Steam Frame)**: 6DOF position, depth testing, peripheral rendering, ATW, DLSS/FSR
- **Foveated Streaming**: Server only sends high-fidelity data for eye-tracked gaze cone
- Eye tracking < 5ms, server "beats your eyes" to next location

---

## 7. Hardware Specifications

### Target Headset: Valve Steam Frame (2026)
- **SoC**: Snapdragon 8 Gen 3
- **RAM**: 16 GB LPDDR5X
- **Connectivity**: Wi-Fi 7 (6GHz) + Dual Antennas + dedicated USB 6GHz receiver
- **Display**: 2160x2160 per eye (Pancake Optics)
- **Eye Tracking**: Dual interior cameras (enables foveated rendering/streaming)
- **OS**: SteamOS (Linux, open architecture for custom Vulkan shaders)

### Cameras: Machine Vision over Cinema
- **Recommendation**: Emergent Vision Technologies Zenith HZ-21000-G
  - 21MP (5120x4096, ~Cinema 8K)
  - Sony Pregius S sensor (IMX530) or Gpixel GSPRINT4521
  - Up to 542 fps at full resolution
  - 100GigE QSFP28 fiber interface
  - IEEE 1588 PTP sync (< 1 microsecond across 100 cameras)
  - Global Shutter
  - GPIO for strobe triggering
- **Lenses**: Sigma Art or Zeiss CP3 primes with adapters (cinema quality on industrial body)
- **Why not RED V-Raptor**: Sync nightmare (BNC genlock), sneakernet data management (CFexpress cards), heat/noise/size, cost

### Lighting: Synchronized Strobed LED
- GPIO-triggered strobe via Gardasoft RT-Series or CCS controllers
- Industrial LED bars (SmartView, Metaphase)
- Exposure at ~1/2000s freezes all motion
- Low heat on talent, low power draw
- For uniform studio: high-CRI continuous LED panels acceptable if enough power

### GPU Compute: NVIDIA DGX B300 (Blackwell Ultra)
- **VRAM**: 288 GB HBM3e per GPU, 2.3 TB total (8 GPUs)
- **Memory BW**: 8.0 TB/s
- **FP4 Compute**: 112 PFLOPS
- **Price**: ~$720-750K per unit
- **Power**: ~15kW, requires dedicated cooling (50k BTU/hr)
- **Deployment**: Remote data center via dedicated dark fiber (DWDM/100G)
  - RDMA / GPUDirect for zero-CPU data path
  - ~0.2ms round-trip latency at 20km
- **Second unit consideration**: Enables pipeline parallelism
  - Unit A: Ingest + Transcoding (always ready for next take)
  - Unit B: Training + Optimization (processes previous take)
  - 16 GPUs = 4.6 TB VRAM, NVLink multi-node as single super-node

### Storage
- NVMe-oF array (WD OpenFlex Data24 or Pure Storage FlashBlade//S)
- 90 GB/s+ sustained sequential writes
- GPUDirect Storage for real-time 8K -> 4K -> 2K downscale + write

### Network
- 100GigE Aggregation Switch (Mellanox/NVIDIA Spectrum-4) in studio
- Dedicated dark fiber uplink to data center
- ConnectX-8 RDMA-enabled NICs

---

## 8. Complete Production Pipeline Summary

### Capture
1. 100x Emergent Zenith 8K cameras (half-circle or full-circle)
2. GPIO-synchronized LED strobes (1/2000s freeze)
3. 100GigE fiber -> Aggregation switch -> Dark fiber -> Data center

### Ingest (B300 Unit A)
1. GPUDirect: Camera data -> GPU VRAM (bypass CPU)
2. Real-time downscale: 8K -> 4K -> 2K (NPP/NVENC)
3. GPUDirect Storage: Write all resolutions to NVMe array

### Pre-processing
1. COLMAP/SfM on 2K footage -> camera poses + sparse point cloud (filter out rig)
2. Grounded SAM 2: Semantic masks (performer, floor, props, rig)
3. SKEL + SUPR estimation from masked 2K video -> body pose + high-res mesh

### Optimization (B300 Unit B)
1. **Stage 1 (2K)**: Initialize Beta splats on SKEL/SUPR surface, LBS constraints, max stiffness
2. **Stage 2 (4K)**: Beta shape refinement, densification, relax constraints for clothing/hair
3. **Stage 3 (8K)**: Patch-based color/SH refinement, normal alignment, geometry locked
4. Inverse rendering: Decompose into Albedo/Normal/Roughness/Metallic

### Compositing (Octane 2026)
1. Import dancer Beta splats into CGI environment
2. Convert CGI environment to splats (static 3D + dynamic 4D)
3. Path-traced GI, shadows, caustics, SSS
4. Export as unified SPZ bitstream

### Delivery
1. Stream canonical model + deformation deltas via Wi-Fi 7
2. Server computes GI/shadows, sends pre-lit splats
3. Headset renders local specular glints + ATW/DLSS
4. Foveated streaming based on eye tracking

---

## Key Papers & Resources
- Disentangled 4DGS: arXiv:2503.22159
- Deformable Beta Splatting: arXiv:2501.18630
- Universal Beta Splatting: arXiv:2510.03312
- Beta Splatting Code: github.com/RongLiu-Leo/beta-splatting
- UBS Project Page: rongliu-leo.github.io/universal-beta-splatting
- SKEL: Biomechanically accurate parametric model
- SUPR: High-resolution parametric human model
- Grounded SAM 2: Grounding DINO-X + SAM 2
- SAMa: Select Any Material (Adobe Research)
- GS-IR / COREA: Gaussian Splatting Inverse Rendering
- Lumina-4DGS: Adaptive curve adjustment for exposure normalization
- SRSplat / SRGS: Super-resolution for splats
- StreamSTGS: Neural bitstream encoding for splat streaming
