# Plan: Splat4D — Native macOS 4D Gaussian Splatting Viewer

## Context

We have a 4D Gaussian Splatting training pipeline that produces PLY files with per-Gaussian temporal parameters. Currently we view results using a Python viser-based viewer that runs on a remote GPU and streams JPEGs to the browser. We want a standalone native Mac app that renders locally on Apple Silicon, with a future path to visionOS / Vision Pro.

## Foundation: MetalSplatter (Fork)

**MetalSplatter** (github.com/scier/MetalSplatter) — MIT license, Swift + Metal, 476 stars, actively maintained.
- Already supports macOS, iOS, visionOS
- PLYIO module for PLY parsing, SplatIO for splat data interpretation
- Tile-based Gaussian splatting rasterizer in Metal compute shaders
- SH evaluation up to degree 3

**Why fork (not SPM dependency):** MetalSplatter pre-bakes covariance from rotation + scale at load time. For 4D, covariance must be recomputed every frame because rotation and scale vary with time. This requires modifying GPU structs, Metal shaders, and the renderer's buffer management — too deep for extension.

## Architecture

```
Splat4D/
  Splat4D.xcodeproj
  Splat4D/
    App/
      Splat4DApp.swift              -- @main, document-based (accepts .ply)
      Splat4DDocument.swift         -- Document model
    Model/
      GaussianModel4D.swift         -- Holds all GPU buffers
      TemporalCompute.swift         -- Dispatches temporal update compute kernel
      PLYLoader4D.swift             -- Reads 4D PLY, builds GPU buffers
    View/
      ContentView.swift             -- Metal view + controls overlay
      MetalView.swift               -- NSViewRepresentable wrapping MTKView
      TimelineControls.swift        -- Time slider, play/pause, speed
      CameraController.swift        -- Orbit/pan/zoom via trackpad + mouse
    Shaders/
      TemporalUpdate.metal          -- Compute kernel for per-frame Gaussian update
  MetalSplatter/                    -- Forked, local Swift Package
```

## GPU Buffer Layout

Two immutable source-of-truth buffers + two mutable scratch buffers per frame:

**Immutable (loaded once from PLY):**

| Buffer | Per Gaussian | Content |
|--------|-------------|---------|
| `RawSplatPoint` | 48 bytes | position(3), sh_dc(3)+opacity_logit(1) as half4, log_scale(3), quaternion(4) |
| `TemporalSplatParams` | 56 bytes | mu_t, s_t, velocity(3), d_scale(3), d_rotation(4), d_sh0(3) |
| SH coefficients | 90 bytes | f_rest_0..44 as Float16 |
| SH derivatives | 90 bytes | d_f_rest_0..44 as Float16 |

**Mutable (rewritten each frame by compute kernel):**

| Buffer | Per Gaussian | Content |
|--------|-------------|---------|
| `EncodedSplatPoint` | ~32 bytes | position(3), color+opacity(half4), covA(half3), covB(half3) |
| Output SH | 90 bytes | time-interpolated SH coefficients |

**Memory at 1M Gaussians: ~410 MB** — fits comfortably on M1 (16 GB unified).

## Temporal Update Compute Kernel

`TemporalUpdate.metal` — dispatched once per frame before rasterization:

```metal
kernel void temporalUpdate(
    device const RawSplat *rawSplats [[buffer(0)]],
    device const TemporalParams *temporalParams [[buffer(1)]],
    device Splat *outputSplats [[buffer(2)]],
    // SH buffers...
    constant TemporalUniforms &uniforms [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    float dt = tp.mu_t - uniforms.timestamp;

    // Position: pos + velocity * dt
    // Opacity:  sigmoid(logit) * exp(-0.5 * (dt/exp(s_t))^2)
    // Scale:    exp(log_scale + d_scale * dt)
    // Rotation: normalize(quat + d_quat * dt)
    // SH:       sh + d_sh * dt
    // Covariance: recompute from time-varying rotation + scale

    // Write EncodedSplatPoint with baked covariance
}
```

Cost: <0.5ms for 1M Gaussians on M1. The existing rasterization pipeline then reads the scratch buffer unchanged.

## Per-Frame Render Loop

```
1. Update timestamp (if playing, advance by deltaTime * speed)
2. Dispatch temporal compute kernel → writes scratch buffers
3. Sort (throttled: every 4-8 frames during playback, on camera change)
4. Rasterize (existing MetalSplatter pipeline, reads scratch buffers)
5. Present
```

Sorting is CPU-based in MetalSplatter. During playback, positions change slightly per frame — sort every few frames is sufficient.

## MetalSplatter Modifications (4 files)

| File | Change |
|------|--------|
| `EncodedSplatPoint.swift` | Add `RawSplatPoint` and `TemporalSplatParams` Swift structs |
| `ShaderCommon.h` | Add Metal-side `RawSplat`, `TemporalParams`, `TemporalUniforms` structs |
| `SplatChunk.swift` | Add optional temporal buffer properties, `is4D` flag |
| `SplatRenderer.swift` | Add `dispatchTemporalUpdate()`, compute pipeline state |

All other MetalSplatter files used unchanged (shaders, sorter, buffer pool, PLYIO).

## UI

```
+--------------------------------------------------+
|  [Open File]    Splat4D            [Settings]     |
+--------------------------------------------------+
|                                                   |
|            Metal Rendering View                   |
|           (orbit/pan/zoom via trackpad)           |
|                                                   |
+--------------------------------------------------+
|  |<  [Play/Pause]  >|   [====o==========] 0.42   |
|  Speed: [1.0x]      Gaussians: 378,098    60 FPS |
+--------------------------------------------------+
```

- Time slider: scrub [0, 1]
- Playback: CVDisplayLink-driven, configurable speed (0.25x–4x)
- Keyboard: Space=play/pause, Left/Right=frame step

## Implementation Phases

### Phase 1: Static Viewer
Fork MetalSplatter, set up Xcode project, create MTKView + orbit camera, load and render a standard 3DGS PLY file. Validates build pipeline and rendering.

### Phase 2: 4D PLY Loading
Create PLYLoader4D using PLYIO's PLYReader. Parse all 76 fields per Gaussian. Initially compute covariance on CPU at t=0.5 to verify loading is correct.

### Phase 3: Temporal Compute Shader
Write TemporalUpdate.metal. Add temporal buffers to SplatChunk. Wire timestamp from UI slider → compute dispatch → render. Verify output matches Python viewer.

### Phase 4: Playback & Polish
TimelineControls with play/pause/speed/scrub. CVDisplayLink-driven animation. Sort throttling. FPS counter.

### Phase 5: visionOS Preparation (Future)
MetalSplatter already has visionOS support. Our temporal compute kernel is platform-agnostic Metal. Port requires: replace MTKView with CompositorServices, stereo cameras, head-tracked viewpoint.

## Technical Risks

1. **CPU sorting bottleneck during playback** — Mitigate with sort throttling (every 4-8 frames). GPU radix sort as future optimization.
2. **Quaternion convention** — Our PLY uses wxyz, Metal's simd_quatf uses xyzw. Handle conversion in PLY loader, use raw float4 in compute kernel.
3. **Float16 covariance precision** — Compute in float32 in kernel, cast to half at write. Same as MetalSplatter's existing approach.
