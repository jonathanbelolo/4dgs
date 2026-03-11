# 360° Stereoscopic Camera Rig Design

## Overview

This document outlines the design for a multi-camera circular array capable of capturing 360° video with stereoscopic depth perception for VR applications. The rig uses 6-8 cameras arranged in a circular pattern to provide full spherical coverage while maintaining proper interpupillary distance (IPD) for natural depth perception.

## The Fundamental Challenge

Creating 360° video with stereo depth perception presents a unique challenge:

- **Stereo vision requires two cameras** separated by the human interpupillary distance (~63-65mm)
- **360° coverage requires cameras pointing in all directions**
- **A single stereo pair** only provides depth for one viewing direction
- **Solution**: Multiple stereo pairs arranged in a circle, with the viewer seeing the appropriate pair based on their head orientation

## Recommended Configuration: 8-Camera Circular Array

### Why 8 Cameras?

The 8-camera configuration provides the optimal balance of:

- **Coverage**: 45° per stereo pair (360° / 8 = 45°)
- **Overlap**: Sufficient for seamless stitching between adjacent camera views
- **Depth quality**: Maintains proper ~65mm baseline between adjacent cameras
- **Complexity**: Manageable number of cameras for sync, storage, and processing
- **Cost**: More affordable than 16-camera rigs while better than 4-6 camera setups

### Camera Array Geometry

```
Top-Down View (looking down at rig):

            C1 (0°)
              |
    C8 (315°) + C2 (45°)
              |
C7 (270°) ----+---- C3 (90°)
              |
    C6 (225°) + C4 (135°)
              |
            C5 (180°)

Circle radius: ~100-150mm (optimized for minimal parallax issues)
Camera spacing: 65mm arc distance between adjacent cameras
```

### Stereo Pairs Formed

The configuration creates **8 overlapping stereo pairs**:

1. C1-C2: Covers 0° - 45° viewing direction
2. C2-C3: Covers 45° - 90°
3. C3-C4: Covers 90° - 135°
4. C4-C5: Covers 135° - 180°
5. C5-C6: Covers 180° - 225°
6. C6-C7: Covers 225° - 270°
7. C7-C8: Covers 270° - 315°
8. C8-C1: Covers 315° - 360°

## Hardware Requirements

### Camera Selection Criteria

Choose cameras based on:

1. **Sensor synchronization**: Frame-level sync across all cameras (critical)
2. **Wide field of view**: 180°+ FOV lenses for maximum overlap
3. **Resolution**: Minimum 4K per camera (8K preferred)
4. **Frame rate**: 30fps minimum, 60fps ideal for smooth VR
5. **Global shutter**: Prevents rolling shutter artifacts during motion
6. **Compact form factor**: Allows tight spacing in circular array

### Recommended Camera Options

**Budget-Friendly (< $5,000 total)**:
- 8x GoPro Hero 12/13 Black
- Hardware sync via external trigger
- 5.3K resolution, up to 60fps
- ~$400 per camera

**Professional (< $20,000 total)**:
- 8x Machine vision cameras (e.g., FLIR Blackfly S with wide-angle lenses)
- Precision hardware sync
- Global shutter
- ~$1,500-2,500 per camera + lenses

**High-End (> $50,000)**:
- Custom Micro Four Thirds sensor array (Meta Four approach)
- 8x MFT sensors with ultra-wide lenses
- Professional color science and dynamic range

### Rig Structure

**Material**: Aluminum extrusion or precision 3D-printed rigid frame

**Key Design Requirements**:
- **Rigidity**: No flex or vibration that could misalign cameras
- **Precise spacing**: ±0.5mm tolerance on camera positions
- **Mounting**: Standard tripod mount (1/4"-20 thread)
- **Cable management**: Clean routing for all camera cables
- **Cooling**: Adequate airflow if cameras generate heat during long takes

**Dimensions**:
- Outer diameter: ~300-400mm
- Height: ~150-200mm (depends on camera size)
- Weight: 2-4kg (excluding cameras)

### Synchronization Hardware

**Critical**: All cameras must capture frames at exactly the same time.

**Options**:
1. **Genlock signal**: Professional solution, requires cameras with genlock input
2. **External trigger**: GPIO/trigger input to all cameras simultaneously
3. **Software sync**: Post-processing alignment (least reliable, not recommended)

**Recommended**: Arduino or Raspberry Pi triggering all cameras via GPIO

## Camera Placement Calculations

### Circular Array Math

For an 8-camera rig with radius R from center:

```
Camera positions (in degrees from 0°):
θ = [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]

Cartesian coordinates:
x_i = R × cos(θ_i)
y_i = R × sin(θ_i)

Arc distance between adjacent cameras:
s = (2πR) / 8 = πR / 4

To achieve 65mm spacing:
R = (4 × 65mm) / π ≈ 82.8mm

Recommended R = 100mm for practical mounting
Actual spacing = (π × 100mm) / 4 ≈ 78.5mm
```

### Vertical Stacking Consideration

For scenes with vertical viewing, consider a **two-tier design**:
- Top ring: 8 cameras pointing horizontally or slightly upward
- Bottom ring: 8 cameras pointing horizontally or slightly downward
- Provides stereo depth for vertical head movements
- Doubles camera count to 16 (significant complexity increase)

**Recommendation**: Start with single-tier horizontal ring for initial prototype.

## Optical Considerations

### Lens Selection

**Field of View Requirements**:
- Minimum 180° horizontal FOV per camera
- Ideally 200°+ for better overlap
- Fisheye lenses are acceptable (distortion corrected in software)

**Lens Considerations**:
- **Entrance pupil position**: Critical for parallax-free stitching
- **Chromatic aberration**: Minimize color fringing at stitch boundaries
- **Distortion profiles**: Must be well-characterized for calibration

### Overlap Zones

Adjacent cameras should overlap by **20-40%** of their FOV:
- Allows accurate feature matching for stitching
- Provides redundancy if one camera fails
- Enables better depth estimation through multi-view stereo

## Software Pipeline

### Capture Phase

1. **Synchronize all cameras** via hardware trigger
2. **Record simultaneously** to high-speed storage (8x 4K streams = ~800 MB/s)
3. **Timecode embedding** for frame alignment verification
4. **Monitor storage** - 1 minute of 8K footage = ~48GB

### Calibration Phase (one-time per rig setup)

1. **Intrinsic calibration**: Determine each camera's internal parameters
   - Focal length
   - Principal point
   - Lens distortion coefficients
   - Use checkerboard pattern

2. **Extrinsic calibration**: Determine relative camera positions
   - Rotation matrices between cameras
   - Translation vectors
   - Use structure-from-motion or known target

3. **Stereo calibration**: Compute rectification for each stereo pair
   - Epipolar geometry
   - Rectification transforms
   - Disparity-to-depth mapping

### Processing Pipeline

```
Raw Footage (8 streams)
    ↓
Frame Synchronization Check
    ↓
Lens Distortion Correction
    ↓
Color Matching (normalize across all cameras)
    ↓
Stereo Rectification (for each adjacent pair)
    ↓
Depth Map Generation (optional, for advanced effects)
    ↓
Equirectangular Projection & Stitching
    ↓
Stereo Panorama Output (Left Eye + Right Eye)
    ↓
VR Video File (e.g., 8K side-by-side or top-bottom stereo)
```

### Output Format

**Standard VR Video Format**:
- **Resolution**: 8K x 4K (7680 x 3840) for each eye
- **Projection**: Equirectangular
- **Stereo layout**: Side-by-side or top-bottom
- **Codec**: H.265/HEVC for file size efficiency
- **Metadata**: Spatial audio metadata if capturing 360° audio

**VR Playback**:
- When viewer looks in direction θ, display the stereo pair from cameras at θ ± 22.5°
- Smooth transition between adjacent pairs as viewer rotates head
- Interpupillary distance should match camera baseline (~65mm)

## Technical Challenges & Solutions

### Challenge 1: Parallax at Center

**Problem**: All cameras radiate from center point, creating parallax errors for close objects.

**Solution**:
- Keep minimum subject distance > 1 meter
- Use larger radius (R = 150mm+) for scenes with closer subjects
- Apply computational correction using depth maps

### Challenge 2: Stitching Artifacts

**Problem**: Visible seams where camera views join, especially with motion.

**Solution**:
- Maximize overlap between cameras
- Use optical flow-based stitching algorithms
- Blend in overlap regions using multi-band blending
- Ensure accurate calibration

### Challenge 3: Storage & Processing

**Problem**: 8 camera streams = massive data volume.

**Solution**:
- Use high-speed NVMe RAID array for capture
- Consider on-camera compression (if latency acceptable)
- Parallel processing pipeline (GPU acceleration)
- Cloud rendering for final output

### Challenge 4: IPD Mismatch

**Problem**: Fixed 65mm camera spacing doesn't match all viewers' IPD (ranges 54-74mm).

**Solution**:
- 65mm is median, acceptable for 90% of users
- Advanced: Capture with wider baseline, computationally adjust in post
- Use depth maps to synthesize different IPD views

### Challenge 5: Synchronization Drift

**Problem**: Cameras falling out of sync over long recordings.

**Solution**:
- Use genlock-capable cameras with master clock
- Monitor sync status during recording
- Post-processing sync verification using audio or visual cues

## Power & Data Management

### Power Requirements

**Per camera**: ~5-8W during recording

**Total**: 8 cameras × 8W = 64W continuous

**Power Options**:
1. **AC power**: Tethered to wall power (studio use)
2. **Battery pack**: High-capacity USB-C PD battery (portable use)
   - Recommended: 100Wh+ battery for 1+ hour runtime
3. **Hybrid**: AC with battery backup

### Data Storage

**Storage calculations** (4K per camera, H.264):
```
Single camera: ~100 Mbps @ 4K30
8 cameras: 800 Mbps = 100 MB/s
Per minute: 6 GB
Per hour: 360 GB

Recommended: 1TB+ high-speed storage for 2+ hours
```

**Storage medium**:
- NVMe SSD array (RAID 0 for speed)
- Multiple SD cards (if cameras support)
- Direct to computer capture

## Bill of Materials (Example Build)

### 8-Camera GoPro Configuration (~$5,000)

| Item | Quantity | Unit Price | Total |
|------|----------|-----------|-------|
| GoPro Hero 12 Black | 8 | $400 | $3,200 |
| Custom aluminum rig | 1 | $500 | $500 |
| Sync trigger (Arduino-based) | 1 | $100 | $100 |
| High-speed USB hub | 1 | $150 | $150 |
| NVMe SSD 2TB | 2 | $200 | $400 |
| Cables & mounting | - | $200 | $200 |
| Power distribution | 1 | $150 | $150 |
| **Total** | | | **$4,700** |

## Development Roadmap

### Phase 1: Proof of Concept (2-4 weeks)
- [ ] Design and build basic 4-camera rig (simpler for testing)
- [ ] Implement hardware sync
- [ ] Capture test footage
- [ ] Basic manual stitching test

### Phase 2: Full Rig Build (4-6 weeks)
- [ ] Expand to 8-camera configuration
- [ ] Precision rig fabrication
- [ ] Camera calibration
- [ ] Automated stitching pipeline (basic)

### Phase 3: Software Refinement (6-8 weeks)
- [ ] Advanced stitching algorithms
- [ ] Depth map generation
- [ ] VR playback optimization
- [ ] Color correction automation

### Phase 4: Production Ready (2-4 weeks)
- [ ] User documentation
- [ ] Capture workflow optimization
- [ ] Quality control procedures
- [ ] Example content creation

## Software Tools & Libraries

### Calibration
- **OpenCV**: Camera calibration, stereo rectification
- **Agisoft Metashape**: Professional photogrammetry calibration

### Stitching
- **PTGui**: Professional panorama stitching (paid)
- **Hugin**: Open-source panorama stitching
- **Mistika VR**: Professional 360 VR stitching (paid)
- **Custom**: OpenCV-based custom stitching pipeline

### VR Playback
- **Unity**: Game engine with VR support
- **Unreal Engine**: Real-time VR rendering
- **GoPro VR Player**: 360 video playback testing

### Processing
- **FFmpeg**: Video encoding/transcoding
- **DaVinci Resolve**: Color correction, final output
- **Nuke**: Compositing and advanced stitching

## Alternative Configurations

### 6-Camera Configuration
- **Coverage**: 60° per pair
- **Pros**: Lower cost, simpler sync
- **Cons**: Less overlap, larger gaps between pairs
- **Best for**: Budget-conscious builds, static scenes

### 12-Camera Configuration
- **Coverage**: 30° per pair
- **Pros**: Better overlap, smoother transitions
- **Cons**: Higher cost, more complex processing
- **Best for**: Professional productions requiring highest quality

### Dual-Ring (16-Camera) Configuration
- **Coverage**: Horizontal + vertical stereo
- **Pros**: Full spherical stereo including up/down
- **Cons**: Double cost, extreme processing requirements
- **Best for**: Premium VR experiences, large budgets

## References & Further Reading

1. **Meta Four Camera**: Professional 360 stereo rig design
2. **GoPro Odyssey**: 16-camera array case study
3. **Research Papers**:
   - "Omnidirectional Stereo Vision Systems" (IEEE)
   - "360MVSNet: Deep Multi-view Stereo Network"
   - "Dual-fisheye omnidirectional stereo"
4. **IPD Studies**: Human interpupillary distance research
5. **VR Video Standards**: Spatial Media specifications (Google/YouTube)

## Conclusion

The 8-camera circular array represents the optimal balance of quality, complexity, and cost for 360° stereoscopic VR capture. With proper calibration and processing, this configuration can deliver immersive VR experiences with natural depth perception across the full 360° viewing sphere.

**Next Steps**:
1. Select camera hardware based on budget and use case
2. Design and fabricate custom rig structure
3. Implement synchronization system
4. Develop or adapt stitching software pipeline
5. Test and iterate on real-world content

This technology enables unprecedented immersive storytelling, virtual tourism, live event capture, and VR content creation.
