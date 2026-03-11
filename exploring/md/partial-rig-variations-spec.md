# Partial Coverage Rig Variations - Technical Specification

## Executive Summary

This document specifies optimized camera rig configurations with reduced camera counts for deployment positions where full 360° coverage is unnecessary or wasteful. By tailoring rig configurations to specific mounting locations, we can achieve:

- **21-35% reduction in camera hardware costs**
- **20-50% reduction in storage and processing requirements**
- **Faster setup and maintenance**
- **Equivalent user experience quality** in captured directions

## The Optimization Opportunity

### Problem Statement

In multi-rig venue deployments, many rig positions have significant coverage areas that capture only static, uninteresting content:

| Position Type | Wasted Coverage | Example |
|--------------|----------------|---------|
| Balcony rail mount | 180° behind (balcony wall) | 50% waste |
| Against wall | 90-180° behind (wall) | 25-50% waste |
| Ceiling mount looking down | 180° upward (ceiling/lights) | 50% waste |
| Proscenium arch | 180° behind (backstage) | 50% waste |
| Corner position | 90° behind (two walls) | 25% waste |

**Current approach**: Deploy full 8-camera 360° rigs everywhere

**Optimized approach**: Deploy camera count based on position requirements

## Rig Configuration Specifications

### Configuration 1: Full Coverage (360°)

**Camera Count**: 8 cameras

**Coverage**: Complete 360° horizontal stereo

**Geometry**:
```
Top-Down View:

        C1 (0°)
          |
  C8 ----●---- C2
  (315°) |   (45°)
         |
  C7 ----+---- C3
  (270°) |   (90°)
         |
  C6 ----●---- C4
  (225°) |   (135°)
         |
        C5 (180°)

Circle radius: 100mm
Camera spacing: 45° angular (78.5mm arc distance)
Stereo pairs: 8 overlapping pairs
```

**Specifications**:
- **Cameras**: 8× GoPro Hero 12 or equivalent
- **Diameter**: 300-400mm
- **Height**: 150-200mm
- **Weight**: 4-6kg with cameras
- **Data rate**: 100 MB/s (8 cameras × 12.5 MB/s)
- **Storage**: 360 GB/hour
- **Cost**: ~$5,000 (GoPro configuration)

**Use Cases**:
- Floor center positions (prime audience viewpoint)
- Photo pit (close to stage)
- Crowd immersion positions
- Any position where viewer may look in any direction

**Coverage Quality**:
- Full stereo: 360°
- Overlap zones: 20-30% between adjacent pairs
- Seamless stitching throughout

---

### Configuration 2: Three-Quarter Coverage (270°)

**Camera Count**: 6 cameras

**Coverage**: 270° horizontal stereo (three-quarters of circle)

**Geometry**:
```
Top-Down View:

        C1 (0°)
          |
       ---●--- C2
          |   (45°)
          |
       ---+--- C3
          |   (90°)
    (135°)|
        C4|
          |
    (180°)|
        C5|
          |
    (225°)|
        C6

[WALL/OBSTRUCTION: 270° - 360°]

Arc radius: 100mm
Camera spacing: 45° angular
Stereo pairs: 5 overlapping pairs
Uncaptured zone: 90° (270° - 360°)
```

**Specifications**:
- **Cameras**: 6× GoPro Hero 12 or equivalent
- **Diameter**: 300mm (asymmetric design possible)
- **Height**: 150-200mm
- **Weight**: 3.5-5kg with cameras
- **Data rate**: 75 MB/s (6 cameras × 12.5 MB/s)
- **Storage**: 270 GB/hour
- **Cost**: ~$3,800 (GoPro configuration)
- **Savings vs. full rig**: $1,200 (24%)

**Use Cases**:
- Corner stage positions
- Side-stage mounts
- Near-wall positions with minimal rear interest
- Positions with 90° obstruction behind

**Coverage Quality**:
- Full stereo: 270° (front three-quarters)
- Fallback coverage: 90° (static panorama or visual boundary)
- Transition zone: 15-20° blend between live and fallback

**Fallback Strategy**:
- Static 360° panorama captured pre-show
- Smooth blend from live stereo to static mono
- Visual indicator when in static zone (subtle vignette)

---

### Configuration 3: Half Coverage (180°)

**Camera Count**: 4 cameras

**Coverage**: 180° horizontal stereo (half circle, forward-facing)

**Geometry**:
```
Top-Down View:

    C1 (0°)     C2 (45°)
        \       /
         \     /
          \   /
           \ /
            ●
           / \
          /   \
         /     \
        /       \
    C4 (135°)  C3 (90°)

[WALL/OBSTRUCTION: 180° - 360° behind]

Arc width: 200mm
Camera spacing: 45° angular
Stereo pairs: 3 overlapping pairs
Uncaptured zone: 180° (behind)
```

**Specifications**:
- **Cameras**: 4× GoPro Hero 12 or equivalent
- **Diameter**: 250mm (half-circle or linear array)
- **Height**: 150-200mm
- **Weight**: 2.5-4kg with cameras
- **Data rate**: 50 MB/s (4 cameras × 12.5 MB/s)
- **Storage**: 180 GB/hour
- **Cost**: ~$2,600 (GoPro configuration)
- **Savings vs. full rig**: $2,400 (48%)

**Use Cases**:
- Balcony rail mounts (looking forward to stage)
- Wall-mounted positions
- Proscenium arch (looking at stage only)
- Overhead positions where rear view is ceiling/rigging
- Behind-stage positions (looking at performers, not backstage)

**Coverage Quality**:
- Full stereo: 180° (forward half)
- Fallback coverage: 180° (static panorama or boundary)
- Transition zones: 15° blend on each edge

**Fallback Strategy Options**:
1. **Static panorama**: Pre-captured 360° still image
2. **Visual boundary**: Artistic frame showing "live window"
3. **Low-res backup**: One 360° camera (Insta360) for mono coverage
4. **Rig-to-rig sharing**: Borrow coverage from other rig's view (if overlapping)

---

### Configuration 4: Hemisphere Coverage (Down-Facing)

**Camera Count**: 4-6 cameras

**Coverage**: 180° hemisphere (looking down only, no upward coverage)

**Geometry**:
```
Side View:

    [CEILING MOUNT]
         ●
        /|\
       / | \
      /  |  \
     C1 C2 C3
     (pointing downward at angles)

     [Additional cameras C4, C5, C6
      fill hemisphere coverage]

Top-Down View:

        C1
       /  \
      C6  C2
      |    |
      C5  C3
       \  /
        C4

All cameras angled downward 45-60°
```

**Specifications**:
- **Cameras**: 4-6× GoPro Hero 12 or equivalent (with wide-angle lenses)
- **Diameter**: 200-300mm
- **Height**: 100-150mm (compact)
- **Weight**: 2.5-5kg with cameras
- **Data rate**: 50-75 MB/s
- **Storage**: 180-270 GB/hour
- **Cost**: ~$2,600-3,800
- **Savings vs. full rig**: $1,200-2,400 (24-48%)

**Use Cases**:
- Ceiling suspension (looking down at stage/audience)
- Overhead truss mounts
- High positions where upward view is infrastructure/lights
- Bird's-eye overview positions

**Coverage Quality**:
- Full stereo: 180° hemisphere (downward)
- No upward coverage (ceiling not interesting)
- Optimized for overhead perspective

**Special Considerations**:
- Cameras mounted at angles (not horizontal)
- May use fisheye lenses for wider FOV per camera
- Calibration accounts for non-horizontal mounting
- Stitching adapted for hemispherical projection

---

## Uncaptured Direction Solutions

### Solution 1: Static Environment Shell (Recommended)

**Concept**: Fill uncaptured directions with pre-recorded static 360° panorama

**Implementation**:
1. **Pre-show capture**: Before event, capture one 360° still panorama from each partial rig position
2. **Method**: Use portable 360° camera (Insta360 X3, Ricoh Theta Z1)
3. **Quality**: 8K equirectangular still image
4. **Processing**: Basic stitching and color correction
5. **Integration**: Composite with live footage during playback

**Playback Behavior**:
```
User head angle: θ

Example for 180° rig (0° - 180° live):

if 0° ≤ θ ≤ 165°:
    Display live stereo footage (high quality)

elif 165° < θ < 195°:
    Blend live → static (30° transition zone)
    Stereo depth gradually reduces
    Visual quality remains high

elif 195° ≤ θ ≤ 360°:
    Display static panorama (mono, no depth)
    Subtle visual indicator (slight vignette)
    Optional UI hint: "Static view"
```

**User Experience**:
- Seamless visual transition
- Environment "freezes" when looking at uncaptured area
- Spatial audio continues normally
- Minimal immersion break

**Advantages**:
- Low cost (one-time panorama capture)
- Simple implementation
- Honest about coverage
- Provides spatial context

**Disadvantages**:
- No motion in uncaptured areas
- No stereo depth in static zones
- Requires pre-show time (5 min per position)

**Cost**: ~$400 for Insta360 X3 (used for all rigs)

---

### Solution 2: Visual Boundary Portal

**Concept**: Frame live coverage as a "window" with artistic boundary

**Implementation**:
1. **Live zone**: High-quality stereo video in captured directions
2. **Boundary**: Visible frame/portal around live coverage edge
3. **Background**: Stylized gradient, venue branding, or abstract pattern
4. **UI**: Clear indication of "live" vs "non-coverage" areas

**Visual Design**:
```
When looking at captured area (0° - 180°):
┌─────────────────────────────────┐
│                                 │
│    [Full Stereo Live Video]     │
│     Immersive, high quality     │
│                                 │
└─────────────────────────────────┘

When looking at uncaptured area (180° - 360°):
┌─────────────────────────────────┐
│  ╔═══════════════════════╗      │  ← Portal frame visible
│  ║                       ║      │
│  ║   [Live Video]        ║      │  ← Can still see edge
│  ║   visible at edge     ║      │     of live coverage
│  ║                       ║      │
│  ╚═══════════════════════╝      │
│                                 │
│    [Branded Background]         │  ← Venue logo, gradient,
│    "Limited Coverage Area"      │     or abstract visual
└─────────────────────────────────┘
```

**User Experience**:
- Clear communication of coverage limits
- Artistic/branded presentation
- Can be experiential (e.g., portal through space/time)
- No attempt to fake uncaptured content

**Advantages**:
- Honest and clear
- Branding opportunity
- Creative design possibilities
- No pre-capture required

**Disadvantages**:
- Breaks immersion more than static panorama
- Requires UI design work
- May feel "incomplete" to users

**Cost**: Design time only (no hardware)

---

### Solution 3: Hybrid Low-Res Backup

**Concept**: Add ONE 360° camera for mono backup coverage

**Implementation**:
1. **Primary**: 4-6 cameras for high-quality stereo in important directions
2. **Backup**: 1× 360° camera (Insta360 X3, GoPro Max) at rig center
3. **Processing**: Dual-layer video
   - High-res stereo layer for main coverage
   - Low-res mono layer for full 360° fallback

**Coverage Map**:
```
Priority Coverage (180° example):

High-quality stereo (0° - 180°):
  - 8K resolution per eye
  - Full stereo depth
  - Primary 4-camera rig

Low-quality mono (180° - 360°):
  - 4K resolution mono
  - No depth (single 360° camera)
  - Backup Insta360 camera

Transition handled automatically
```

**User Experience**:
- Full 360° coverage (no gaps)
- Quality gracefully degrades in less important directions
- Stereo depth only where it matters
- Seamless (user may not notice quality change)

**Advantages**:
- Complete coverage
- Better than static panorama (has motion)
- Relatively low cost
- Simple fallback mechanism

**Disadvantages**:
- Additional camera per rig (+$450)
- Extra data stream (lower bitrate, but still adds ~25 MB/s)
- Slight complexity in processing
- No stereo in backup areas

**Cost**: +$450 per rig (Insta360 X3)

**When to use**: Premium positions where complete coverage is important but full stereo everywhere is overkill

---

### Solution 4: Rig-to-Rig Coverage Sharing

**Concept**: When multiple rigs can see each other's positions, share coverage

**Implementation**:
```
Example: Theater with 3 rigs

Rig A: Balcony (180° forward, wall behind)
Rig B: Orchestra (360° full coverage)
Rig C: Stage-left (270° coverage)

When user at Rig A looks backward:
→ Blend to Rig B's view of the balcony area
→ User sees balcony from orchestra perspective
→ Creates spatial continuity

When user at Rig C looks to blocked corner:
→ Blend to Rig B's view of that corner
→ OR blend to static panorama
```

**User Experience**:
- Seamless coverage by "borrowing" views
- Can create interesting multi-perspective moments
- Complex but potentially powerful

**Advantages**:
- Maximizes value of multi-rig deployment
- Can create unique spatial experiences
- No additional hardware needed

**Disadvantages**:
- Very complex to implement
- Only works where rigs have overlapping views
- May be disorienting (perspective jump)
- Requires sophisticated processing pipeline

**Cost**: Development time only

**When to use**: Advanced feature for premium multi-rig deployments

---

### Solution 5: Soft Boundary with Visual Feedback

**Concept**: Allow full rotation, provide gentle feedback in uncaptured areas

**Implementation**:
1. **Full freedom**: User can look anywhere (no hard boundaries)
2. **Captured area**: Full quality stereo, no indicators
3. **Uncaptured area**:
   - Progressive vignette (darkening at edges)
   - Text overlay: "No coverage in this direction"
   - Optional: Abstract pattern or blur
4. **Audio**: Spatial audio continues (captured separately via ambisonic mic)

**Visual Treatment**:
```
Entering uncaptured zone:

Frame 1 (θ = 170°, still in live coverage):
  [Full quality, no indicators]

Frame 2 (θ = 185°, entering uncaptured):
  [Slight vignette starts]
  [Subtle text: "Coverage boundary"]

Frame 3 (θ = 200°, fully uncaptured):
  [Strong vignette or dark background]
  [Text: "No camera coverage"]
  [Spatial audio still accurate]
```

**User Experience**:
- Honest about limitations
- Doesn't try to fake content
- Maintains spatial audio for context
- Clear communication

**Advantages**:
- Simple to implement
- Honest user communication
- No pre-capture needed
- Works for any partial rig

**Disadvantages**:
- Least immersive solution
- Breaks presence in uncaptured areas
- May frustrate users expecting full coverage

**Cost**: Minimal (UI design only)

**When to use**: Budget deployments, technical demos, or when coverage transparency is important

---

## Comparison Matrix

| Solution | Cost per Rig | Pre-Capture Time | Coverage | Immersion | Complexity |
|----------|-------------|------------------|----------|-----------|------------|
| **Static Panorama** | $400 (one-time) | 5 min | 360° mono | High → Medium | Low |
| **Visual Boundary** | $0 | 0 | Partial only | Medium | Low |
| **Hybrid Backup** | +$450 | 0 | 360° (varying quality) | High | Medium |
| **Rig Sharing** | $0 | 0 | Depends on rig placement | Medium → High | High |
| **Soft Boundary** | $0 | 0 | Partial only | Low → Medium | Low |

**Recommended Default**: Static Panorama (best balance of cost, immersion, and simplicity)

---

## Pipeline Adaptations

### 1. Rig Configuration System

Each rig needs a configuration file defining its coverage:

**Example: `balcony_center_config.json`**
```json
{
  "rig_id": "balcony_center_001",
  "rig_type": "180_degree",
  "deployment_position": {
    "venue_section": "Balcony",
    "position": "Center rail",
    "height_meters": 8.0,
    "coordinates": [0, 0, 8]
  },
  "hardware": {
    "camera_count": 4,
    "camera_model": "GoPro Hero 12 Black",
    "cameras": [
      {
        "id": "CAM_01",
        "angle_degrees": 0,
        "position": [100, 0, 0]
      },
      {
        "id": "CAM_02",
        "angle_degrees": 45,
        "position": [70.7, 70.7, 0]
      },
      {
        "id": "CAM_03",
        "angle_degrees": 90,
        "position": [0, 100, 0]
      },
      {
        "id": "CAM_04",
        "angle_degrees": 135,
        "position": [-70.7, 70.7, 0]
      }
    ]
  },
  "coverage": {
    "live_stereo": {
      "start_angle": 0,
      "end_angle": 180,
      "quality": "8K_stereo",
      "stereo_pairs": [
        ["CAM_01", "CAM_02"],
        ["CAM_02", "CAM_03"],
        ["CAM_03", "CAM_04"]
      ]
    },
    "fallback": {
      "type": "static_panorama",
      "start_angle": 180,
      "end_angle": 360,
      "file": "balcony_center_static_8K.jpg",
      "quality": "8K_mono",
      "transition_zone_degrees": 30
    }
  },
  "sync": {
    "timecode_source": "tentacle_sync_01",
    "frame_rate": 30
  },
  "storage": {
    "data_rate_mbps": 50,
    "estimated_gb_per_hour": 180
  }
}
```

### 2. Calibration Workflow Updates

**Standard Calibration** (same process, fewer cameras):

1. **Intrinsic Calibration**:
   - Calibrate each camera individually (same as 360° rig)
   - Use checkerboard pattern
   - Compute lens distortion coefficients

2. **Extrinsic Calibration**:
   - Determine relative positions of active cameras only
   - Same structure-from-motion techniques
   - Handles 4, 6, or 8 camera configurations equally

3. **Stereo Calibration**:
   - Compute rectification for each active stereo pair
   - 4-camera rig: 3 pairs
   - 6-camera rig: 5 pairs
   - 8-camera rig: 8 pairs

**No fundamental changes** to calibration algorithms—just fewer cameras to process.

**Tools**: OpenCV, Agisoft Metashape (both handle variable camera counts)

### 3. Stitching Pipeline Modifications

**Input Detection**:
```python
# Pseudocode for rig-aware stitching

def stitch_footage(rig_config, camera_feeds):
    camera_count = rig_config['hardware']['camera_count']
    coverage_type = rig_config['rig_type']

    if coverage_type == "360_degree":
        output = stitch_360(camera_feeds, camera_count=8)

    elif coverage_type == "270_degree":
        live_output = stitch_270(camera_feeds, camera_count=6)
        static_bg = load_static_panorama(rig_config['coverage']['fallback']['file'])
        output = composite_live_and_static(live_output, static_bg,
                                           transition_zone=30)

    elif coverage_type == "180_degree":
        live_output = stitch_180(camera_feeds, camera_count=4)
        static_bg = load_static_panorama(rig_config['coverage']['fallback']['file'])
        output = composite_live_and_static(live_output, static_bg,
                                           transition_zone=30)

    elif coverage_type == "hemisphere":
        output = stitch_hemisphere(camera_feeds, camera_count=camera_count)

    return output
```

**Compositing Live + Static**:
```python
def composite_live_and_static(live_footage, static_panorama, transition_zone):
    """
    Blend live stereo footage with static mono panorama

    Args:
        live_footage: Partial coverage (e.g., 0° - 180°)
        static_panorama: Full 360° static image
        transition_zone: Degrees of blending (e.g., 30°)
    """
    output = create_stereo_frame()

    for angle in range(360):
        if angle in live_coverage_zone:
            # Use live footage at full quality
            output[angle] = live_footage[angle]

        elif angle in transition_zone:
            # Blend live → static
            blend_factor = calculate_blend(angle, transition_zone)
            output[angle] = blend(
                live_footage[angle] * blend_factor,
                static_panorama[angle] * (1 - blend_factor)
            )
            # Gradually reduce stereo depth
            output[angle].depth = reduce_depth(output[angle].depth, blend_factor)

        else:
            # Use static panorama (mono)
            output[angle] = static_panorama[angle]
            output[angle].depth = 0  # No stereo in static zones

    return output
```

### 4. VR Application Metadata

**Viewpoint Descriptor** for VR app:

```json
{
  "experience_name": "Broadway Show - Multi-Viewpoint",
  "viewpoints": [
    {
      "id": "orchestra_center",
      "name": "Orchestra Center",
      "description": "Premium audience perspective",
      "rig_type": "360_degree",
      "position": [0, 5, 1.5],
      "video_file": "orchestra_center_8K_stereo.mp4",
      "coverage": {
        "live_stereo": {
          "range": "0° - 360°",
          "quality": "8K_stereo"
        }
      },
      "thumbnail": "orchestra_thumb.jpg"
    },
    {
      "id": "balcony_center",
      "name": "Balcony Overview",
      "description": "Elevated view of full stage",
      "rig_type": "180_degree",
      "position": [0, 20, 8],
      "video_file": "balcony_center_8K_partial.mp4",
      "coverage": {
        "live_stereo": {
          "range": "0° - 180°",
          "quality": "8K_stereo"
        },
        "static_fallback": {
          "range": "180° - 360°",
          "quality": "8K_mono",
          "method": "static_panorama",
          "transition_zone_degrees": 30
        }
      },
      "ui_hints": {
        "show_coverage_indicator": true,
        "indicator_type": "subtle_vignette"
      },
      "thumbnail": "balcony_thumb.jpg"
    },
    {
      "id": "stage_left",
      "name": "Stage Left",
      "description": "Side stage performer view",
      "rig_type": "270_degree",
      "position": [-5, 2, 1.5],
      "video_file": "stage_left_8K_partial.mp4",
      "coverage": {
        "live_stereo": {
          "range": "0° - 270°",
          "quality": "8K_stereo"
        },
        "static_fallback": {
          "range": "270° - 360°",
          "quality": "8K_mono",
          "method": "static_panorama",
          "transition_zone_degrees": 30
        }
      },
      "thumbnail": "stage_left_thumb.jpg"
    }
  ]
}
```

### 5. Real-Time Playback Logic

**VR App Head Tracking Handler**:

```javascript
// Unity/Unreal pseudocode for VR playback

function onHeadRotation(headAngle, currentViewpoint) {
    const config = currentViewpoint.coverage;
    const liveRange = config.live_stereo.range;
    const fallbackRange = config.static_fallback?.range;

    if (isInRange(headAngle, liveRange)) {
        // Full quality live stereo
        setVideoLayer('live_stereo', opacity: 1.0);
        setVideoLayer('static_fallback', opacity: 0.0);
        setStereoDepth(1.0);
        hideUIIndicator();

    } else if (isInTransitionZone(headAngle, config.static_fallback.transition_zone_degrees)) {
        // Blend zone
        const blendFactor = calculateBlendFactor(headAngle, liveRange, fallbackRange);
        setVideoLayer('live_stereo', opacity: blendFactor);
        setVideoLayer('static_fallback', opacity: 1.0 - blendFactor);
        setStereoDepth(blendFactor); // Gradually reduce depth
        showUIIndicator(opacity: 1.0 - blendFactor); // Fade in indicator

    } else if (fallbackRange && isInRange(headAngle, fallbackRange)) {
        // Static panorama zone
        setVideoLayer('live_stereo', opacity: 0.0);
        setVideoLayer('static_fallback', opacity: 1.0);
        setStereoDepth(0.0); // No stereo in static
        showUIIndicator(opacity: 0.3); // Subtle vignette

    }
}

function showUIIndicator(opacity) {
    // Subtle visual feedback in uncaptured zones
    // Options: vignette, corner icon, subtle text
    uiIndicator.setOpacity(opacity);
}
```

---

## Cost-Benefit Analysis

### Single Rig Comparison

| Rig Type | Cameras | Hardware Cost | Storage (GB/hr) | Savings vs. 360° | Use Case |
|----------|---------|---------------|-----------------|------------------|----------|
| **360° Full** | 8 | $5,000 | 360 | — (baseline) | Floor center, premium positions |
| **270° Partial** | 6 | $3,800 | 270 | $1,200 (24%) | Corners, side-stage |
| **180° Half** | 4 | $2,600 | 180 | $2,400 (48%) | Balcony, walls, overhead |
| **Hemisphere** | 4-6 | $2,600-3,800 | 180-270 | $1,200-2,400 | Ceiling mounts |

### Multi-Rig Deployment Scenarios

#### Scenario 1: Medium Venue (7 rigs total)

**All 360° Rigs (Current)**:
- 7 rigs × 8 cameras = 56 cameras
- Total cost: $35,000
- Storage: 2.5 TB/hour
- Processing: Full stitching for 56 cameras

**Optimized Mixed Rigs**:
- 3× 360° (floor positions): 24 cameras → $15,000
- 2× 270° (side positions): 12 cameras → $7,600
- 2× 180° (balcony/wall): 8 cameras → $5,200
- **Total: 44 cameras → $27,800**
- **Savings: $7,200 (21%)**
- Storage: 1.98 TB/hour (21% reduction)
- Processing: 21% faster stitching

#### Scenario 2: Large Venue (12 rigs total)

**All 360° Rigs**:
- 12 rigs × 8 cameras = 96 cameras
- Total cost: $60,000
- Storage: 4.3 TB/hour
- Processing: Extreme compute requirements

**Optimized Mixed Rigs**:
- 5× 360° (premium floor): 40 cameras → $25,000
- 3× 270° (sides/corners): 18 cameras → $11,400
- 3× 180° (balcony/walls): 12 cameras → $7,800
- 1× Hemisphere (ceiling): 6 cameras → $3,800
- **Total: 76 cameras → $48,000**
- **Savings: $12,000 (20%)**
- Storage: 3.27 TB/hour (24% reduction)
- Processing: 21% fewer cameras to stitch

#### Scenario 3: Theater (4 rigs, careful placement)

**All 360° Rigs**:
- 4 rigs × 8 cameras = 32 cameras
- Total cost: $20,000
- Storage: 1.4 TB/hour

**Optimized Mixed Rigs**:
- 2× 360° (orchestra floor): 16 cameras → $10,000
- 1× 180° (balcony front): 4 cameras → $2,600
- 1× 270° (stage side): 6 cameras → $3,800
- **Total: 26 cameras → $16,400**
- **Savings: $3,600 (18%)**
- Storage: 1.1 TB/hour (21% reduction)

### Long-Term Savings

**Per-Event Operating Costs** (7-rig deployment):

| Cost Category | All 360° | Optimized Mix | Savings |
|--------------|----------|---------------|---------|
| Storage media | $1,400 | $1,100 | $300 |
| Processing time (labor) | $2,000 | $1,600 | $400 |
| Data transfer | $200 | $160 | $40 |
| Equipment maintenance | $350 | $275 | $75 |
| **Per-event total** | **$3,950** | **$3,135** | **$815** |

**Over 20 events**: $815 × 20 = **$16,300 additional savings**

**Total 20-event savings**: $7,200 (hardware) + $16,300 (operating) = **$23,500**

---

## Recommended Deployment Strategy

### Decision Matrix: Which Rig Type for Which Position?

```
Position Assessment Checklist:

1. Can viewers realistically look in all directions?
   YES → Consider 360° rig
   NO → Evaluate partial rig

2. Is there a wall, ceiling, or obstruction behind position?
   YES → Consider 180° or 270° rig
   NO → Consider 360° rig

3. Is this a premium "main viewpoint" position?
   YES → Use 360° for best quality
   NO → Partial rig acceptable

4. Is the position overhead/ceiling-mounted?
   YES → Use hemisphere rig
   NO → Use standard horizontal config

5. What percentage of coverage is interesting content?
   >80% → 360° rig
   60-80% → 270° rig
   <60% → 180° rig

6. Budget constraints significant?
   YES → Maximize use of partial rigs
   NO → Use 360° for flexibility
```

### Venue-Specific Recommendations

**Concert Venue**:
- Photo pit: 360° (premium view)
- Front of house: 360° (professional mix position)
- Stage sides: 270° (limited rear interest)
- Balcony: 180° (wall behind)
- Ceiling: Hemisphere (overhead view)

**Broadway Theater**:
- Orchestra center: 360° (premium audience view)
- Mezzanine: 180° (looking forward only)
- Balcony: 180° (looking at stage)
- Proscenium: 180° (looking at stage, not backstage)

**Stadium/Arena**:
- Floor center: 360° (full immersion)
- Corner positions: 270° (two walls nearby)
- Upper deck: 180° (looking at field)
- Suspended overhead: Hemisphere (looking down)

---

## Implementation Checklist

### Pre-Deployment

- [ ] Venue site survey
- [ ] Identify rig positions
- [ ] Assess coverage requirements per position
- [ ] Choose rig configurations (360°, 270°, 180°, hemisphere)
- [ ] Create rig config files for each position
- [ ] Determine fallback strategy (static panorama, visual boundary, etc.)

### Static Panorama Capture (if using)

- [ ] Bring portable 360° camera (Insta360 X3)
- [ ] Capture panorama from each partial rig position (5 min each)
- [ ] Capture panorama with similar lighting to show (if possible)
- [ ] Transfer panoramas for processing
- [ ] Stitch panoramas to 8K equirectangular
- [ ] Color correct to match live cameras
- [ ] Store with rig config files

### Live Capture

- [ ] Deploy rigs according to configuration
- [ ] Verify sync across all cameras
- [ ] Test storage (ensure adequate capacity per rig)
- [ ] Monitor capture during show
- [ ] Verify all feeds recording correctly

### Post-Production

- [ ] Calibrate each rig (intrinsic + extrinsic)
- [ ] Stitch live footage per rig config
- [ ] Composite live + static (for partial rigs)
- [ ] Color match across all viewpoints
- [ ] Generate stereo pairs (left/right eye)
- [ ] Encode to VR format
- [ ] Create viewpoint metadata JSON
- [ ] Build VR app with viewpoint switching
- [ ] QA test coverage transitions

### Quality Control

- [ ] Verify coverage zones match config
- [ ] Test transitions between live and static
- [ ] Check for visual artifacts at blend zones
- [ ] Verify stereo depth quality in live zones
- [ ] Test on multiple VR headsets
- [ ] User testing for immersion quality

---

## Technical Specifications Summary

### Hardware per Rig Type

| Component | 360° Rig | 270° Rig | 180° Rig | Hemisphere |
|-----------|----------|----------|----------|------------|
| Cameras | 8 | 6 | 4 | 4-6 |
| Diameter | 400mm | 350mm | 300mm | 250mm |
| Weight | 6kg | 5kg | 4kg | 4-5kg |
| Power | 64W | 48W | 32W | 32-48W |
| Data rate | 100 MB/s | 75 MB/s | 50 MB/s | 50-75 MB/s |
| Storage/hr | 360GB | 270GB | 180GB | 180-270GB |
| Cost (GoPro) | $5,000 | $3,800 | $2,600 | $2,600-3,800 |

### Fallback Solutions Cost Comparison

| Solution | Cost/Rig | Capture Time | Quality | Complexity |
|----------|----------|--------------|---------|------------|
| Static panorama | $0* | 5 min | High | Low |
| Visual boundary | $0 | 0 | Medium | Low |
| Hybrid backup | $450 | 0 | High | Medium |
| Rig sharing | $0 | 0 | Variable | High |
| Soft boundary | $0 | 0 | Low | Low |

*Assumes one Insta360 X3 ($400) used for all rigs

---

## Future Enhancements

### Advanced Features

1. **AI-Driven Coverage Optimization**
   - Analyze venue geometry automatically
   - Suggest optimal rig types per position
   - Predict viewer attention zones

2. **Dynamic Resolution Scaling**
   - Higher bitrate for frequently viewed directions
   - Lower bitrate for rarely viewed zones
   - Based on user analytics

3. **Depth Map Enhancement**
   - Use depth maps to synthesize new viewing angles
   - Fill coverage gaps computationally
   - Create "impossible" viewpoints

4. **Volumetric Hybrid**
   - Combine camera rigs with volumetric capture
   - Camera rigs for wide coverage
   - Volumetric for key performance areas

### Software Pipeline Improvements

1. **Automated Rig Detection**
   - Auto-detect camera count on startup
   - Load appropriate stitching profile
   - Reduce manual configuration

2. **Real-Time Preview**
   - Live preview of stitched output
   - Coverage zone visualization
   - Immediate quality feedback

3. **Cloud Processing**
   - Parallel processing of multiple rigs
   - Faster turnaround for events
   - Scalable infrastructure

---

## Conclusion

Partial coverage rig variations offer significant cost and operational advantages for multi-rig deployments without compromising user experience quality. By matching rig configuration to position requirements, productions can:

- **Reduce hardware costs by 20-35%**
- **Decrease storage and processing requirements by 20-50%**
- **Simplify deployment and operation**
- **Maintain high quality in important viewing directions**

**Recommended approach**:
1. Use full 360° rigs for premium, central positions
2. Deploy 180° or 270° rigs for edge positions (balconies, walls, corners)
3. Implement static panorama fallback for uncaptured directions
4. Use hemisphere rigs for overhead suspension

This optimization makes large-scale multi-rig deployments significantly more economically viable while preserving the immersive VR experience where it matters most.

---

## References

- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Equirectangular Projection: https://en.wikipedia.org/wiki/Equirectangular_projection
- VR Video Standards: https://github.com/google/spatial-media/blob/master/docs/spherical-video-rfc.md
- GoPro Hero 12 Specifications: https://gopro.com/en/us/shop/cameras/hero12-black/
- Insta360 X3 Specifications: https://www.insta360.com/product/insta360-x3

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Author**: Technical Specifications Team
**Status**: Production Ready