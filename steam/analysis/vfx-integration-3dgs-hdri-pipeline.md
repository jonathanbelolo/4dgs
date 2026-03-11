# VFX Integration Pipeline: 3DGS + HDRI Lighting

**Classification:** Technical Pipeline — Confidential
**Date:** February 2026
**Version:** 1.0

---

## Overview

This document describes a VFX integration pipeline that replaces the traditional LIDAR + proxy geometry workflow with 3D Gaussian Splatting (3DGS) scene reconstruction, combined with on-set HDRI capture for physically accurate lighting. The result is faster set digitization, better bounce lighting, and native compatibility with modern GPU renderers like Octane.

---

## The Problem

Integrating CGI assets into live-action plates requires two things:

1. **Accurate lighting** — so the CGI asset is lit consistently with the real environment
2. **Nearby geometry** — so light bounces off set surfaces onto the CGI asset, and the asset casts correct contact shadows and occlusion

The traditional pipeline handles these separately:

| Need | Traditional Solution | Limitation |
|------|---------------------|------------|
| Lighting | HDRI chrome ball on set | Treats all light as infinitely distant — no parallax |
| Geometry | LIDAR scan + manual cleanup | Slow, expensive, untextured, requires manual surfacing |
| Surface color | Reference photos → hand-painted textures on proxy geo | Labor-intensive, approximate |

This works, but the geometry step is the bottleneck — hours of scanning, cleanup, and manual texturing for proxy geometry that exists only to catch shadows and bounce light.

---

## The Pipeline

### On-Set Capture

Two assets are captured on set:

| Asset | Equipment | Time | Purpose |
|-------|-----------|------|---------|
| **HDRI** | Chrome ball + bracketed exposures (or dedicated HDRI camera) | ~5 minutes | True radiance values for primary lighting (20+ stops of dynamic range) |
| **Reference photos** | Any camera, handheld, ~20-50 photos around the set | ~10 minutes | Input for 3DGS reconstruction |

No LIDAR scanner required.

### Reconstruction

The reference photos are processed through a feed-forward multi-view 3DGS method:

| Method | Input | Output | Speed |
|--------|-------|--------|-------|
| **NoPoSplat** | Sparse unposed images | Consistent 3DGS in canonical space | Seconds (feed-forward) |
| **AnySplat** | Unconstrained views | 3DGS from arbitrary camera configurations | Seconds (feed-forward) |
| **Traditional 3DGS** | Posed images (via COLMAP) | Per-scene optimized 3DGS | Minutes (optimization) |

For highest quality, a brief per-scene optimization pass can follow the feed-forward initialization (adds 2-5 dB PSNR, takes minutes not hours).

The output is a 3DGS `.ply` file representing the full set environment with geometry and textured appearance.

### Integration in Renderer

Using a renderer with native Gaussian support (e.g., Octane):

1. **Import the 3DGS** — the set environment loads as native Gaussian geometry
2. **Apply the HDRI** — as the lighting environment (image-based lighting)
3. **Place the CGI asset** — positioned within the Gaussian set
4. **Ray trace** — the renderer bounces HDRI light off the Gaussian set surfaces onto the CGI asset

The renderer handles:
- **Primary illumination** from the HDRI (correct radiance, full dynamic range)
- **Bounce light** off Gaussian surfaces (correct color tint from actual set materials, correct intensity from HDRI)
- **Contact shadows** where the CGI asset meets Gaussian geometry
- **Ambient occlusion** from surrounding Gaussian surfaces
- **Reflections** with correct spatial parallax (not infinitely distant like HDRI-only)

---

## Why This Is Better

### vs. HDRI Alone

HDRI treats the entire environment as infinitely distant. A red wall 2 meters from the CGI character bounces the same as if it were at infinity — wrong parallax, wrong solid angle, wrong occlusion. The 3DGS provides the nearby geometry that makes bounce light physically correct.

### vs. HDRI + LIDAR

LIDAR gives geometry but no appearance. The resulting proxy mesh must be manually textured to produce correct bounce light color. 3DGS captures geometry and surface appearance simultaneously — the color tint of bounced light is inherently correct because the Gaussians encode the actual surface colors from the reference photos.

### vs. HDRI + Manual Proxy Geometry

Manual proxy construction takes hours of artist time and produces approximate geometry. 3DGS reconstruction from photos takes seconds to minutes and produces geometry that is photogrammetrically accurate.

### Comparison Table

| Capability | HDRI Only | HDRI + LIDAR | HDRI + Manual Proxy | **HDRI + 3DGS** |
|------------|-----------|-------------|--------------------|--------------------|
| Primary lighting accuracy | Excellent | Excellent | Excellent | **Excellent** |
| Bounce light direction | None | Correct | Approximate | **Correct** |
| Bounce light color | None | Manual texturing | Approximate | **Automatic** |
| Contact shadows | None | Correct | Correct | **Correct** |
| Reflection parallax | None | Correct | Approximate | **Correct** |
| Set capture time | 5 min | 1-2 hours | 5 min (photos) | **15 min** |
| Post-processing time | Minimal | Hours (cleanup) | Hours (modeling) | **Minutes** |
| Equipment cost | Low | High (scanner) | Low | **Low** |

---

## Why HDRI Remains Essential

3DGS alone cannot replace HDRI for lighting because Gaussian reconstructions are **radiometrically inaccurate**:

- **LDR appearance** — Gaussians learn tone-mapped, clipped camera output (~8 stops), not physical radiance (20+ stops)
- **Baked exposure** — a bright window blown to (255, 255, 255) in source photos reads the same as a white wall, when the window may be 100x brighter
- **View-dependent baking** — spherical harmonic coefficients encode specular as seen from training views, not actual light transport

The HDRI provides what the Gaussians cannot: true luminance values across the full dynamic range of the scene. The Gaussians provide what the HDRI cannot: spatially accurate nearby geometry with surface color.

**Together they are complementary, not redundant.**

---

## Renderer Compatibility

### Octane Render

Octane supports native Gaussian Splatting geometry, enabling direct ray tracing against imported 3DGS assets without intermediate mesh conversion. This eliminates the Poisson reconstruction or SuGaR meshing step that would otherwise be required.

### Other Renderers

For renderers without native Gaussian support, a meshing step is required:

| Method | Output | Quality |
|--------|--------|---------|
| **SuGaR** | Textured mesh from 3DGS | High quality, preserves detail |
| **2DGS** | Mesh-aligned Gaussians | Clean surfaces |
| **Poisson reconstruction** | Watertight mesh from Gaussian centers | Standard, well-understood |

---

## Practical Considerations

### Capture Guidelines for On-Set Reference Photos

- **Coverage:** Photograph all surfaces that could bounce light onto the CGI asset's position — floors, walls, nearby furniture, ceilings
- **Overlap:** ~60-70% overlap between adjacent photos for robust reconstruction
- **Lighting consistency:** Capture under the same lighting conditions as the plate (same time of day, same practicals on/off)
- **Resolution:** Standard production stills camera is sufficient — 3DGS quality is driven more by view coverage than per-image resolution
- **Count:** 20-50 images is typically sufficient for a single set; complex environments may benefit from more

### Limitations

- **Dynamic set elements** (moving curtains, flickering practicals) will ghost in the reconstruction — capture with set locked off
- **Highly specular surfaces** (mirrors, chrome) are challenging for 3DGS — these are better handled by the HDRI reflection environment
- **Transparent surfaces** (glass, water) are not well-represented by Gaussians — use traditional proxy geometry for these elements
- **The Gaussian set is for lighting only** — it does not appear in the final render; the live-action plate remains the background

### Quality vs. Speed Tradeoffs

| Approach | Reconstruction Time | Bounce Light Quality |
|----------|-------------------|---------------------|
| Feed-forward only (NoPoSplat) | ~5 seconds | Good — usable for previz and most shots |
| Feed-forward + brief optimization | ~5-15 minutes | High — production quality |
| Full per-scene optimization (traditional 3DGS) | ~30-60 minutes | Highest — hero shots |

For most VFX work, feed-forward + brief optimization provides the best quality-to-time ratio.

---

## Future: Relightable Gaussian Splatting

Emerging research methods decompose Gaussian scenes into physically-based material properties (albedo, roughness, metallic, normals) rather than baked appearance:

- **GaRe, ROS-GS** — inverse rendering from Gaussians
- **Relightable 3DGS** — separates lighting from material

When these mature, the HDRI may become optional — the Gaussian reconstruction itself would contain enough physical information to serve as both geometry and light source. But this is research-stage, not production-ready today.

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Shoot HDRI on set | 5 min |
| 2 | Capture 20-50 reference photos of set | 10 min |
| 3 | Run through NoPoSplat / 3DGS reconstruction | Seconds to minutes |
| 4 | Import 3DGS + HDRI into Octane | Minutes |
| 5 | Place CGI asset, ray trace | Standard render time |

**Total set digitization time: ~15-20 minutes, no LIDAR, no manual modeling.**

The result: physically accurate bounce light with correct color, contact shadows, occlusion, and parallax-correct reflections — at a fraction of the cost and time of traditional set reconstruction.

---

*This document is part of the vZero Technical Document Suite.*
