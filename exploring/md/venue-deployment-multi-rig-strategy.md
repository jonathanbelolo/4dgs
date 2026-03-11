# Multi-Rig 360° Stereo Deployment for Live Events

## Overview

This document analyzes the physical footprint, audience impact, and multi-rig deployment strategies for capturing concerts, Broadway shows, and other live events using 360° stereo camera rigs. It covers single-rig placement, multi-viewpoint systems for teleportation between positions, the infrastructure required for synchronized multi-rig capture, and **optimized partial-coverage rig configurations** that reduce costs by 20-48% for edge positions.

**Key optimization**: Not all deployment positions require full 360° coverage. Rigs mounted against walls, on balconies, or overhead can use 4-6 cameras instead of 8, saving significant hardware, storage, and processing costs while maintaining quality in captured directions.

> **See also**: [Partial Rig Variations Specification](partial-rig-variations-spec.md) for detailed technical specifications of 360°, 270°, 180°, and hemisphere rig configurations.

## Single Rig Physical Footprint

### Core Rig Dimensions

**Rig Unit Only**:
- Diameter: 300-400mm (12-16 inches)
- Height: 150-200mm (6-8 inches)
- Weight: 2-4kg (4-9 lbs) without cameras
- Total weight with cameras: 4-6kg (9-13 lbs)

**Mounted Configuration**:

The rig requires mounting infrastructure that significantly increases the footprint:

```
Option 1: Tripod Mount (Most Common)
- Footprint: ~600mm diameter circle (tripod legs)
- Total height: 1.5-2m from ground (eye-level viewing)
- Stability: Good, requires sandbag weights
- Portability: Medium

Option 2: Monopod/Pole Mount
- Footprint: Single point, ~100mm base
- Total height: 1.5-2m from ground
- Stability: Requires guy-wires or weighted base
- Portability: High

Option 3: Overhead Rigging
- Footprint: Zero floor space
- Height: Suspended from ceiling/truss
- Stability: Excellent (no vibration)
- Portability: Low, requires venue infrastructure

Option 4: Custom Stand
- Footprint: ~400mm square base
- Total height: Adjustable 1-2m
- Stability: Excellent with proper design
- Portability: Medium
```

### Visual Obstruction Analysis

**Horizontal Obstruction**:

The rig creates a circular obstruction of approximately **400mm diameter** at mounting height.

From audience member's perspective:
- At 5m distance: Rig appears ~4.6° wide (minimal obstruction)
- At 10m distance: Rig appears ~2.3° wide (barely noticeable)
- At 20m distance: Rig appears ~1.1° wide (negligible)

**Vertical Obstruction**:

Key consideration: **Mounting height**

```
Scenario 1: Eye-level mount (1.6m height)
- Obstructs view for anyone directly behind within 3-5m
- Impact: HIGH for seated audiences
- Recommendation: Avoid for ticketed seating areas

Scenario 2: Above-head mount (2.5m height)
- Minimal obstruction for standing audiences
- Slight obstruction for balcony/upper seating looking down
- Impact: LOW to MEDIUM
- Recommendation: Good for general admission concerts

Scenario 3: Overhead suspension (4m+ height)
- No obstruction for ground-level audiences
- Cameras may be too far for intimate perspective
- Impact: MINIMAL
- Recommendation: Best for Broadway/theater with rigging
```

### Sight Line Analysis for Typical Venues

#### Concert Venue (3,000 capacity)

```
Floor Plan View:

        [STAGE]
    ═════════════════

Row 1:  X X X X X X X    ← Premium seats (2m from stage)
Row 5:  X X X X X X X    ← 6m from stage
Row 10: X X X X X X X    ← 12m from stage

Rig Placement Options:

Position A: Front-center (4m from stage, 2.5m height)
  - Obstructs: Rows 1-3 center seats (6-10 seats)
  - Benefits: Best stage view
  - Verdict: HIGH IMPACT - avoid

Position B: Side-stage (2m from stage edge, 2m height)
  - Obstructs: Side-view seats (2-4 seats)
  - Benefits: Unique angle, minimal center impact
  - Verdict: MEDIUM IMPACT - acceptable

Position C: Elevated platform at sound booth (25m from stage, 3m height)
  - Obstructs: Nobody (behind audience)
  - Benefits: Zero impact, overview perspective
  - Verdict: MINIMAL IMPACT - ideal for overview

Position D: Suspended from ceiling (center, 5m height)
  - Obstructs: Nobody
  - Benefits: Perfect center position
  - Verdict: NO IMPACT - requires venue support
```

#### Broadway Theater (1,000 capacity)

```
Side View:

[STAGE]
   |
   | Orchestra Pit (2m below stage)
   ↓
[Rig Position 1] ← Stage level, 3m from front

   Row A (2m from rig)  ← Premium orchestra seats
   Row E (6m from rig)
   Row M (12m from rig)

[Mezzanine Level] (4m above orchestra)

[Rig Position 2] ← Suspended from proscenium

[Balcony Level] (8m above orchestra)

[Rig Position 3] ← Balcony rail mount


Recommended Positions:

Position 1: Orchestra center aisle (eye-level)
  - Impact: Obstructs 0 seats if in aisle
  - Requires: Aisle placement permission
  - View: Authentic audience perspective

Position 2: Proscenium suspension (4m high)
  - Impact: Zero obstruction
  - Requires: Theater rigging crew
  - View: Elevated, professional angle

Position 3: Balcony rail
  - Impact: Obstructs balcony front row sight line
  - Requires: Rail mounting bracket
  - View: High overview perspective
```

### Cables & Infrastructure Visibility

Beyond the rig itself, consider:

**Power cables**:
- 8 cameras = 8+ cables to power distribution
- Risk: Trip hazard if floor-mounted
- Solution: Overhead cable runs, cable ramps, gaff tape

**Data cables**:
- USB or Ethernet to central recorder
- May require 5-10m cable runs
- Solution: Wireless transmission (adds latency) or tidy cable management

**Control station**:
- Computer/recorder nearby (within 5m typically)
- Requires: Small table/rack space
- Footprint: 600mm × 600mm area
- Can be hidden behind equipment cases

## Rig Configuration Optimization

### The Partial Coverage Opportunity

Multi-rig deployments often place cameras in positions where **full 360° coverage is wasteful**:

| Position Type | Wasted Coverage | Solution |
|--------------|----------------|----------|
| **Balcony rail** | 180° behind = balcony wall | Use 180° (4-camera) rig |
| **Against wall** | 90-180° behind = wall | Use 180-270° rig |
| **Ceiling mount** | 180° upward = ceiling/lights | Use hemisphere (4-6 camera) rig |
| **Proscenium** | 180° behind = backstage | Use 180° (4-camera) rig |
| **Corner position** | 90° behind = two walls | Use 270° (6-camera) rig |

### Rig Type Configurations

**Full Coverage (360°) - 8 cameras**:
- **Cost**: ~$5,000 (GoPro configuration)
- **Storage**: 360 GB/hour
- **Use for**: Floor center, photo pit, crowd positions
- **Coverage**: Complete 360° horizontal stereo

**Three-Quarter Coverage (270°) - 6 cameras**:
- **Cost**: ~$3,800 (24% savings)
- **Storage**: 270 GB/hour
- **Use for**: Corners, side-stage positions
- **Coverage**: 270° stereo + 90° static panorama fallback

**Half Coverage (180°) - 4 cameras**:
- **Cost**: ~$2,600 (48% savings)
- **Storage**: 180 GB/hour
- **Use for**: Balcony rails, walls, proscenium
- **Coverage**: 180° stereo + 180° static panorama fallback

**Hemisphere (Down-Facing) - 4-6 cameras**:
- **Cost**: ~$2,600-3,800 (24-48% savings)
- **Storage**: 180-270 GB/hour
- **Use for**: Ceiling suspension, overhead views
- **Coverage**: 180° hemisphere downward only

### Uncaptured Direction Solutions

For partial rigs, uncaptured directions are handled via:

1. **Static Panorama (Recommended)**: Pre-captured 360° still image fills gaps
   - Captured before show (5 min per position)
   - Seamless blend from live stereo to static mono
   - Maintains spatial context
   - Cost: $400 one-time (Insta360 X3 for all rigs)

2. **Visual Boundary**: Artistic frame showing "live window"
   - Clear communication of coverage limits
   - Branding opportunity
   - Cost: $0 (design only)

3. **Hybrid Backup**: Add one 360° camera for low-res mono coverage
   - Complete 360° coverage (varying quality)
   - Cost: +$450 per rig

> **See**: [Partial Rig Variations Spec](partial-rig-variations-spec.md) for detailed technical implementation, pipeline adaptations, and fallback solution comparisons.

### Decision Matrix: Choosing Rig Type by Position

```
Position Assessment:

1. Is there a wall/ceiling/obstruction behind?
   YES → Consider 180° or 270° rig
   NO → Consider 360° rig

2. Is this a premium "main viewpoint" position?
   YES → Use 360° for maximum flexibility
   NO → Partial rig acceptable

3. What % of coverage is interesting content?
   >80% → 360° rig
   60-80% → 270° rig
   <60% → 180° rig

4. Is position overhead/ceiling-mounted?
   YES → Use hemisphere rig
   NO → Use horizontal config

5. Budget constraints significant?
   YES → Maximize partial rigs
   NO → Use 360° for flexibility
```

## Multi-Rig Deployment Strategy

### Concept: Multi-Viewpoint VR Experience

Instead of a single fixed perspective, deploy **multiple synchronized rigs** throughout the venue. Users can teleport between viewpoints while watching in VR, choosing their preferred perspective.

**User Experience**:
```
User wearing VR headset sees:

  Current View: Front Row Center

  [Hotspots visible in VR:]
  → "Teleport to Stage Left"
  → "Teleport to Balcony Overview"
  → "Teleport to Behind Drummer"

  User selects hotspot → Smooth fade transition → New 360° viewpoint

  All positions maintain full 360° stereo vision
```

### Optimal Rig Quantity by Venue Type

#### Small Venue (500 capacity club)

**Recommended: 3 rigs**

```
    [STAGE]
    ═══════

Rig 1: Front Center (2m from stage, audience POV)
Rig 2: Stage Right (1m from stage edge, performer POV)
Rig 3: Rear Center (8m from stage, room overview)

Coverage: 3 distinct perspectives
Infrastructure: Manageable for single operator
Data rate: 2.4 Gbps total (300 MB/s)
Cost impact: 3× rig cost (~$15K for GoPro setup)
```

#### Medium Venue (3,000 capacity arena)

**Recommended: 5-7 rigs (optimized configuration)**

```
           [STAGE]
        ═════════════

Rig 1: Front Left (near stage) → 360° rig
Rig 2: Front Center (premium seat view) → 360° rig
Rig 3: Front Right (near stage) → 360° rig
Rig 4: Mid-house Left (side perspective) → 270° rig
Rig 5: Mid-house Center (sound booth area) → 360° rig
Rig 6: Mid-house Right (side perspective) → 270° rig
Rig 7: Overhead/Balcony (bird's eye) → 180° or Hemisphere rig

Coverage: Comprehensive venue coverage
Infrastructure: Requires dedicated crew (2-3 people)

Optimized cost comparison:
- All 360° rigs: 56 cameras, $35,000, 700 MB/s
- Mixed configuration: 44 cameras, $27,800, 550 MB/s
- Savings: $7,200 (21%), 150 MB/s less data
```

#### Large Venue (10,000+ capacity stadium)

**Recommended: 8-12 rigs (optimized configuration)**

```
Strategic positions:
- Floor front (3 rigs): Left, Center, Right close to stage → 360° rigs
- Floor mid (2 rigs): House left/right at mix position → 360° rigs
- Floor rear (1 rig): Back of venue overview → 180° rig (wall behind)
- Stage positions (3 rigs): On-stage perspectives → 270° rigs (if permitted)
- Elevated (3 rigs): Balcony levels, scaffolding → 180° or Hemisphere rigs

Coverage: Full venue, multiple height levels
Infrastructure: Dedicated crew of 4-6 people

Optimized cost comparison:
- All 360° rigs: 96 cameras, $60,000, 1,200 MB/s
- Mixed configuration: 76 cameras, $48,000, 950 MB/s
- Savings: $12,000 (20%), 250 MB/s less data
```

#### Broadway Theater (1,000 capacity)

**Recommended: 4-5 rigs (optimized configuration)**

```
[STAGE]
   ↓

Rig 1: Orchestra Row F Center (audience perspective) → 360° rig
Rig 2: Orchestra Row A Left (front row angle) → 360° rig
Rig 3: Mezzanine Center (elevated view) → 180° rig (looking at stage)
Rig 4: Balcony Center (upper level) → 180° rig (wall behind)
Rig 5: On-stage (performer POV, if show permits) → 270° rig (limited backstage)

Coverage: Multiple seating tiers
Infrastructure: Requires theater technical crew

Optimized cost comparison:
- All 360° rigs: 40 cameras, $25,000, 500 MB/s
- Mixed configuration: 26 cameras, $16,400, 325 MB/s
- Savings: $8,600 (34%), 175 MB/s less data
```

## Multi-Rig Technical Requirements

### Synchronization

**Critical**: All rigs must capture frames at exactly the same time.

**Genlock System**:

```
Master Clock
    ↓
    ├─→ Rig 1 (all 8 cameras synced)
    ├─→ Rig 2 (all 8 cameras synced)
    ├─→ Rig 3 (all 8 cameras synced)
    ├─→ Rig 4 (all 8 cameras synced)
    └─→ Rig 5 (all 8 cameras synced)

Total synchronized cameras: 5 rigs × 8 cameras = 40 cameras

Sync method options:
1. Wired genlock signal (professional, most reliable)
2. Network time protocol (NTP) with GPS sync
3. Wireless timecode (Tentacle Sync or similar)
4. SMPTE timecode from venue audio system
```

**Recommended Solution**:
- Timecode generator (Tentacle Sync Studio or similar)
- Wireless timecode receivers at each rig
- ±1 frame accuracy (33ms @ 30fps)
- Cost: ~$200 per rig + $500 master unit

### Network Infrastructure

For live monitoring and control of multiple rigs:

```
Network Topology:

[Internet] (optional, for streaming)
    ↓
[Main Router/Switch] 10 Gbps backbone
    ↓
    ├─→ [Rig 1 Recorder] 1 Gbps
    ├─→ [Rig 2 Recorder] 1 Gbps
    ├─→ [Rig 3 Recorder] 1 Gbps
    ├─→ [Rig 4 Recorder] 1 Gbps
    ├─→ [Rig 5 Recorder] 1 Gbps
    └─→ [Master Control Station] 1 Gbps

Allows:
- Remote monitoring of all rigs
- Centralized start/stop recording
- Live preview streams
- Status monitoring (storage, battery, sync)
```

**Equipment needed**:
- Managed network switch (10 Gbps): $500-1,000
- CAT6 cables (up to 50m runs): $200
- Network-attached storage (NAS) for backup: $2,000-5,000

### Data Storage Scaling

**Storage requirements scale linearly with rig count**:

```
Single rig: 100 MB/s (8 cameras × 12.5 MB/s each)
5 rigs: 500 MB/s total
10 rigs: 1,000 MB/s total = 1 GB/s

For a 2-hour show:

1 rig:  720 GB
5 rigs: 3.6 TB
10 rigs: 7.2 TB

Recommended storage strategy:
- Local NVMe SSD at each rig (2TB per rig)
- Network backup to central NAS (optional during capture)
- Post-show transfer to central server for processing
```

**Cost scaling**:
- 2TB NVMe SSD per rig: ~$200 × number of rigs
- Central 20TB NAS: ~$3,000-5,000
- Backup drives: 2× capacity, ~$500 per 10TB

### Power Distribution

**Per rig power consumption**: ~80-100W

```
5 rigs × 100W = 500W total
10 rigs × 100W = 1,000W total

Power solutions:

Option 1: AC power distribution
- Run AC to each rig location
- Requires venue power access
- No runtime limit
- Cables may be trip hazard

Option 2: Battery packs per rig
- 150Wh battery = ~90 min runtime per rig
- Fully portable, no cables
- Limited by show duration
- Cost: ~$200 per battery × number of rigs

Option 3: Hybrid
- AC where possible (fixed positions)
- Battery for mobile/stage positions
- Best flexibility

Recommended:
- Use venue AC power for fixed positions
- Battery backup for reliability
- Hot-swap batteries for shows >2 hours
```

### Crew Requirements

**Minimal setup (1-3 rigs)**:
- 1 technical operator
- Setup time: 1-2 hours
- Can manage monitoring during show

**Medium setup (4-7 rigs)**:
- 2-3 technical operators
- Setup time: 3-4 hours
- Dedicated monitor operator during show

**Large setup (8+ rigs)**:
- 4-6 technical operators
- Setup time: 4-6 hours
- Multiple monitor operators
- Dedicated troubleshooting crew member

## Placement Strategies by Venue Type

### Rock/Pop Concert Arena

**Priorities**: Energy, crowd atmosphere, multiple perspectives

**Ideal positions**:

1. **Photo Pit** (Front center, 2m from stage, 1.5m height)
   - Captures: Performer close-ups, stage energy
   - Obstruction: Minimal (photographers already there)
   - Permission: Requires venue/artist approval
   - **Rig type**: 360° (premium viewpoint)

2. **Front of House** (25m from stage, 2.5m height)
   - Captures: Professional mixer perspective
   - Obstruction: Zero (behind audience)
   - Permission: Easy (sound booth area)
   - **Rig type**: 180° (wall/booth behind)

3. **Stage Left/Right** (On riser at stage edge)
   - Captures: Side stage view, performer interactions
   - Obstruction: None if on elevated platform
   - Permission: Requires production approval
   - **Rig type**: 270° (limited backstage interest)

4. **Crowd Center** (15m from stage, 2m height on platform)
   - Captures: Audience immersion, crowd energy
   - Obstruction: Medium (requires small platform)
   - Permission: Requires GA floor space allocation
   - **Rig type**: 360° (full immersion)

5. **Upper Level** (Balcony rail, 8m above floor)
   - Captures: Overview, venue scale
   - Obstruction: Minimal
   - Permission: Easy if venue has balcony
   - **Rig type**: 180° (balcony wall behind)

### Broadway Theater

**Priorities**: Intimacy, sightlines, artistic perspective

**Ideal positions**:

1. **Orchestra Center** (Row H center aisle, 1.5m height)
   - Captures: Traditional theater audience perspective
   - Obstruction: Zero if mounted in aisle
   - Permission: Requires theater management approval
   - Note: This is the "premium seat" view
   - **Rig type**: 360° (premium position)

2. **Front Row Offset** (Row A, left/right aisle, 1.3m height)
   - Captures: Intimate performer perspective
   - Obstruction: Minimal
   - Permission: Moderate difficulty
   - **Rig type**: 360° (full audience immersion)

3. **Mezzanine Front** (Mezzanine Row A, 4m above orchestra)
   - Captures: Elevated view of stage and full set
   - Obstruction: Only for mezzanine front row
   - Permission: Easy
   - **Rig type**: 180° (looking forward at stage only)

4. **Proscenium Mount** (Suspended from arch, 5m high, center)
   - Captures: Perfect framing of stage
   - Obstruction: Zero
   - Permission: Difficult (requires rigging crew)
   - **Rig type**: 180° (backstage not interesting) or Hemisphere (looking down)

**Not recommended**:
- On-stage placement (breaks fourth wall)
- Orchestra pit (unless documenting musicians)

### Classical Concert / Symphony

**Priorities**: Acoustics, conductor view, instrument sections

**Ideal positions**:

1. **Conductor's Perspective** (On-stage behind conductor podium, 2m height)
   - Captures: Looking at orchestra, immersive musician view
   - Obstruction: None (on stage)
   - Permission: Requires orchestra approval
   - Note: Most unique perspective

2. **Front Center Audience** (Row 10 center, 1.5m height)
   - Captures: Traditional concert hall experience
   - Obstruction: Minimal if in aisle
   - Permission: Moderate

3. **Balcony Center** (First balcony front row)
   - Captures: Full hall acoustic sweet spot
   - Obstruction: Minimal
   - Permission: Easy

4. **Stage Left/Right** (Side boxes or stage level)
   - Captures: Sectional perspective (strings vs brass)
   - Obstruction: Low
   - Permission: Moderate

### Sporting Event (Arena/Stadium)

**Priorities**: Action following, multiple angles, replay value

**Ideal positions**:

1. **Courtside/Rinkside** (2m from playing surface)
   - Captures: Player-level intensity
   - Obstruction: May block premium seats
   - Permission: Difficult (premium location)

2. **Behind Goal/Net** (Elevated, 3m high)
   - Captures: Scoring action perspective
   - Obstruction: Low if elevated
   - Permission: Moderate

3. **Center Court/Ice** (Upper level, suspended if possible)
   - Captures: Broadcast angle, full field view
   - Obstruction: Zero if suspended
   - Permission: Easy (press box area)

4. **Corner Positions** (Mid-level, 4 corners)
   - Captures: Strategic overview
   - Obstruction: Low
   - Permission: Moderate

## Audience Impact Mitigation Strategies

### Strategy 1: Aisle Mounting

Place rigs in aisles rather than blocking seating:
- Uses existing walkways
- Zero seat loss
- May require narrower rig design or offset mounting
- Requires venue approval (fire code compliance)

### Strategy 2: Elevated Platforms

Small platforms (500mm × 500mm × 500mm high):
- Raises rig above seated audience heads
- People can see under/around it
- Requires ~1 seat space worth of floor area
- More stable than tripods

### Strategy 3: Overhead Suspension

Suspend from venue infrastructure:
- Requires rigging points rated for load
- Professional riggers needed
- Zero floor obstruction
- Most expensive option (~$500-1,000 per suspension point)

### Strategy 4: "Dead Space" Utilization

Use venue spaces that don't impact ticketed audience:
- Behind sound booth
- In lighting towers
- On stage (if production allows)
- In press boxes
- On balcony railings

### Strategy 5: Transparent Communication

Work with venue to:
- Identify "obstructed view" seats already selling at discount
- Place rigs in those locations
- Further discount those seats or comp them
- Invite those ticket holders to exclusive VR preview

### Strategy 6: Compact Rig Design

Optimize rig for minimal profile:
- Use slim-profile cameras (machine vision vs. action cameras)
- Vertical mounting (tall, narrow vs. wide)
- Matte black finish (less visually obvious)
- Reduces apparent size by 30-40%

## Viewpoint Switching User Experience

### Interface Design (VR Headset)

**Method 1: Gaze-based Hotspots**

```
User looks around in 360°
→ Semi-transparent markers appear at other rig locations
→ User gazes at marker for 2 seconds
→ Smooth fade transition (1 second)
→ New viewpoint loads
→ User maintains head orientation (looking same direction)
```

**Method 2: Controller Menu**

```
User presses menu button
→ Overhead map of venue appears
→ Icons show all available viewpoints
→ User selects with controller
→ Instant transition or fade
→ Landing orientation: facing stage/action
```

**Method 3: Automatic Switching**

```
AI director chooses viewpoint based on:
→ Current action (score in sports, solo in concert)
→ User viewing history
→ Dramatic timing
→ User can override at any time
```

### Technical Implementation

**Live Streaming Multi-Rig**:

Extremely bandwidth-intensive. For 5 rigs:

```
5 rigs × 8K stereo × 30fps × compression = ~150 Mbps per rig

Total: 750 Mbps downstream to viewer
Impractical for most consumer internet

Solutions:
1. Lower resolution per viewpoint (4K instead of 8K)
2. Adaptive bitrate (only stream active viewpoint at full quality)
3. Pre-buffer adjacent viewpoints at lower quality
```

**Recommended for Live**: Stream only 2-3 viewpoints at full quality, viewer chooses at start.

**Post-Produced Multi-Rig**:

Full quality available, user switches at will:

```
All 5 rigs processed to 8K stereo
→ Packaged in VR app with hotspot navigation
→ User downloads entire experience (~50-100GB)
→ Local playback, instant viewpoint switching
→ Best quality, zero latency
```

## Cost Analysis: Multi-Rig Deployments

### 3-Rig Setup (Small Venue)

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| Camera rigs (GoPro-based) | 3 | $5,000 | $15,000 |
| Timecode sync system | 1 | $1,200 | $1,200 |
| Network switch & cables | 1 | $800 | $800 |
| Storage (2TB SSD per rig) | 3 | $200 | $600 |
| Mounting hardware | 3 | $400 | $1,200 |
| Power distribution | 1 | $500 | $500 |
| **Subtotal** | | | **$19,300** |
| Crew (setup + show) | 2 people | $500/day | $1,000 |
| **Total per event** | | | **$20,300** |

### 7-Rig Setup (Medium Venue)

**Configuration A: All 360° Rigs (Original)**

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| Camera rigs (GoPro-based) | 7 × 360° | $5,000 | $35,000 |
| Timecode sync system | 1 | $1,500 | $1,500 |
| Network switch & cables | 1 | $1,500 | $1,500 |
| Storage (2TB SSD per rig) | 7 | $200 | $1,400 |
| Mounting hardware | 7 | $400 | $2,800 |
| Power distribution | 1 | $800 | $800 |
| Central NAS storage | 1 | $4,000 | $4,000 |
| **Subtotal** | | | **$47,000** |
| Crew (setup + show) | 3 people | $600/day | $1,800 |
| **Total per event** | | | **$48,800** |

**Configuration B: Optimized Mixed Rigs (Recommended)**

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| 360° rigs (floor positions) | 3 | $5,000 | $15,000 |
| 270° rigs (side positions) | 2 | $3,800 | $7,600 |
| 180° rigs (balcony/wall) | 2 | $2,600 | $5,200 |
| Insta360 X3 (static panoramas) | 1 | $400 | $400 |
| Timecode sync system | 1 | $1,500 | $1,500 |
| Network switch & cables | 1 | $1,500 | $1,500 |
| Storage (reduced per rig) | 7 | $150 avg | $1,050 |
| Mounting hardware | 7 | $400 | $2,800 |
| Power distribution | 1 | $800 | $800 |
| Central NAS storage | 1 | $3,000 | $3,000 |
| **Subtotal** | | | **$38,850** |
| Crew (setup + show) | 3 people | $600/day | $1,800 |
| **Total per event** | | | **$40,650** |

**Savings with optimized configuration**: $8,150 (17% reduction)
**Ongoing savings**: 21% less storage, 21% less processing time per event

### Amortization Over Multiple Events

If producing a series of events:

**All 360° Rigs (Configuration A)**:
```
7-rig system cost: $47,000 (one-time)
Crew cost per event: $1,800
Storage/processing per event: $500

Event 1: $48,800 total ($48,800 per event)
Event 5: $56,000 total ($11,200 per event)
Event 10: $65,000 total ($6,500 per event)
Event 20: $83,000 total ($4,150 per event)
```

**Optimized Mixed Rigs (Configuration B - Recommended)**:
```
7-rig system cost: $38,850 (one-time)
Crew cost per event: $1,800
Storage/processing per event: $395 (21% less)

Event 1: $40,650 total ($40,650 per event)
Event 5: $48,430 total ($9,686 per event)
Event 10: $57,800 total ($5,780 per event)
Event 20: $76,540 total ($3,827 per event)
```

**Total savings over 20 events**: $6,460 (8% overall reduction)

Break-even: After ~10-15 events, equipment cost is amortized. Optimized configuration maintains cost advantage throughout lifecycle.

## Distribution Strategy

### Option 1: VR App Download

Users download app with all viewpoints:
- **File size**: 10-20GB per hour of content (all viewpoints)
- **Platform**: Meta Quest, SteamVR, PlayStation VR
- **Price point**: $5-20 per event
- **Pros**: Highest quality, unlimited viewpoint switching
- **Cons**: Large download, requires VR headset

### Option 2: Streaming Service

Users stream selected viewpoints:
- **Bandwidth**: 50-100 Mbps per stream
- **Platform**: Web-based or dedicated app
- **Price point**: $15-30 per event (pay-per-view)
- **Pros**: No large download, accessible on multiple devices
- **Cons**: Requires high-speed internet, limited viewpoint switching

### Option 3: Hybrid

Users stream initial view, download full experience:
- Start watching immediately in primary viewpoint
- Background download of other viewpoints
- Full quality available after buffer period
- Best user experience

## Regulatory & Venue Considerations

### Permissions Required

**Venue Management**:
- Floor plan approval for rig placement
- Load capacity approval for suspended rigs
- Fire code compliance (aisle obstruction)
- Insurance liability

**Production/Artist**:
- Recording rights
- Distribution rights
- Creative approval of viewpoints
- On-stage access (if applicable)

**Union/Labor**:
- Camera crew union agreements (if venue is union)
- Rigging crew requirements
- Electrical crew for power distribution

### Insurance

Typical requirements for multi-rig deployment:
- General liability: $1-2 million coverage
- Equipment insurance: Full replacement value
- Riggers liability (for suspended equipment)

**Cost**: ~$1,000-3,000 per event for coverage

## Case Study Examples

### Example 1: Indie Band at 500-capacity Club

**Setup**: 3 rigs
- Front center (audience perspective)
- Stage left (on riser)
- Back of room (overview)

**Impact**:
- Obstruction: 2 tickets lost (stage riser placement)
- Crew: 1 operator
- Cost: $20,000 initial + $500 per show

**Result**: Minimal audience impact, unique fan experience

### Example 2: Broadway Musical (8 shows/week)

**Setup**: 4 rigs
- Orchestra Row G center aisle
- Orchestra Row B left aisle
- Mezzanine front center
- Proscenium suspension

**Impact**:
- Obstruction: 0 tickets lost (all in aisles or suspended)
- Crew: 2 operators (rotate shifts)
- Cost: $25,000 initial + $1,000 per week

**Result**: Zero audience impact, premium VR product

### Example 3: Stadium Concert (50,000 capacity)

**Setup**: 10 rigs
- 3 in photo pit
- 2 at FOH mix position
- 3 on stage truss
- 2 in upper decks

**Impact**:
- Obstruction: ~20 tickets lost (platform positions)
- Revenue loss: ~$2,000 in tickets
- Crew: 5 operators
- Cost: $60,000 initial + $3,000 per show

**Result**: Minimal impact relative to capacity, global streaming opportunity

## Recommendations Summary

### For Concert Venues:
✅ **Use 5-7 rigs** for medium venues (3,000-5,000 cap)
✅ **Prioritize**: Photo pit, FOH, balcony, one crowd position
✅ **Mounting**: Mix of tripods and overhead suspension
✅ **Impact mitigation**: Dead space placement, elevated platforms
✅ **Rig optimization**:
  - 360° rigs: Photo pit, crowd center (3 rigs)
  - 270° rigs: Stage sides (2 rigs)
  - 180° rigs: FOH booth, balcony (2 rigs)
  - **Savings**: ~$7,200 (21%)

### For Broadway/Theater:
✅ **Use 3-4 rigs** for traditional proscenium theaters
✅ **Prioritize**: Orchestra center, mezzanine, suspended proscenium
✅ **Mounting**: Aisle placement and rigging
✅ **Impact mitigation**: Zero seat loss strategy
✅ **Rig optimization**:
  - 360° rigs: Orchestra center (1-2 rigs)
  - 180° rigs: Mezzanine, balcony, proscenium (2-3 rigs)
  - **Savings**: ~$8,600 (34%)

### For Large Festivals/Stadiums:
✅ **Use 8-12 rigs** for full coverage
✅ **Prioritize**: Multiple height levels, on-stage if possible
✅ **Mounting**: Professional rigging crew required
✅ **Impact mitigation**: Minimal % of capacity affected
✅ **Rig optimization**:
  - 360° rigs: Floor positions, premium locations (5-6 rigs)
  - 270° rigs: Stage/corner positions (2-3 rigs)
  - 180°/Hemisphere rigs: Balconies, overhead (3-4 rigs)
  - **Savings**: ~$12,000 (20%)

### Universal Optimization Guidelines:
✅ **Default strategy**: Use static panorama fallback for partial rigs
✅ **Pre-show checklist**: Capture 360° panoramas from each partial rig position (5 min each)
✅ **Processing**: Blend live stereo footage with static backgrounds during post-production
✅ **User experience**: Subtle visual indicators when viewing uncaptured zones
✅ **Long-term**: Optimized configurations save 20-48% per rig while maintaining quality

## Next Steps

1. **Venue Scout**: Walk venue with floor plan, identify placement locations
2. **Rig Type Selection**: For each position, determine optimal rig configuration (360°, 270°, 180°, or hemisphere)
3. **Coverage Assessment**: Identify which positions have obstructions/wasted coverage
4. **Sight Line Test**: Use cardboard mockup to test obstruction before event
5. **Venue Approval**: Present plan to management with impact analysis
6. **Crew Planning**: Hire appropriate crew size for rig count
7. **Power Survey**: Map power access at each rig location
8. **Network Design**: Plan cable runs or wireless infrastructure
9. **Static Panorama Planning**: Schedule pre-show time to capture fallback panoramas (if using partial rigs)
10. **Rehearsal**: Test setup day before (if possible)
11. **Contingency**: Backup plan if rig position becomes problematic

With careful planning and optimized rig configurations, multi-rig 360° stereo capture can provide extraordinary immersive experiences while maintaining minimal impact on live audiences **and reducing deployment costs by 20-35%**.

---

## Additional Resources

**For detailed technical specifications on partial rig configurations**, see:
- [Partial Rig Variations Specification](partial-rig-variations-spec.md) - Complete technical specs for 360°, 270°, 180°, and hemisphere rigs, including hardware requirements, pipeline adaptations, fallback solutions, and cost-benefit analysis.

**Related Documents**:
- [360° Stereo Camera Rig Design](360-stereo-camera-rig-design.md) - Core 8-camera rig design and fundamentals
- [GoPro Prototype Rig BOM](gopro-prototype-rig-bom.md) - Bill of materials for GoPro-based implementation
- [Software Pipeline Complete Workflow](software-pipeline-complete-workflow.md) - Processing pipeline for stitching and VR output
