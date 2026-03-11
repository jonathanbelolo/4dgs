# GoPro Prototype Rig - Bill of Materials (BOM)

**Configuration**: 8-Camera Circular Array for 360° Stereoscopic VR Capture
**Target Use**: Broadway/Theater proof of concept
**Budget**: ~$5,000 per rig
**Build Time**: 2-4 weeks (1 week parts sourcing, 1 week fabrication, 1-2 weeks testing)

---

## Complete System Cost Summary

| Category | Subtotal |
|----------|----------|
| Cameras & Accessories | $3,940 |
| Synchronization System | $550 |
| Rig Structure & Mounting | $800 |
| Power System | $680 |
| Storage & Recording | $1,200 |
| Cables & Connectors | $280 |
| Calibration & Tools | $150 |
| **TOTAL (Single Rig)** | **$7,600** |
| Contingency (10%) | $760 |
| **TOTAL WITH CONTINGENCY** | **$8,360** |

**Note**: This is higher than the $5K estimate due to including all necessary accessories, power, storage, and sync. Can be reduced by using lower-cost alternatives (see Budget Optimization section below).

---

## 1. CAMERAS & ACCESSORIES

### Primary Cameras

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| GoPro Hero 12 Black | CHDHX-121-XX | 8 | $399 | $3,192 | B&H, Amazon, BestBuy | Latest model with best image quality |
| GoPro Protective Housing | ADDIV-001 (included) | 8 | Included | $0 | Included with camera | Provides mounting points |
| GoPro Curved Adhesive Mounts | AACFT-001 | 2 packs (6/pack) | $15 | $30 | B&H | For rig attachment points |
| GoPro Anti-Fog Inserts | AHDAF-301 | 2 packs (12/pack) | $15 | $30 | B&H | Prevent condensation during long shoots |
| MicroSD Cards (512GB UHS-II) | SanDisk Extreme Pro 512GB | 8 | $85 | $680 | B&H, Amazon | V60 speed class for 5.3K recording |
| MicroSD Card Reader (USB-C) | Anker 2-in-1 SD/MicroSD | 2 | $14 | $28 | Amazon | For offloading footage |

**Subtotal: $3,960**

### Camera Configuration Notes

**GoPro Hero 12 Settings for VR Capture**:
- Resolution: 5.3K @ 60fps (5312 × 2988)
- Aspect Ratio: 16:9
- FOV: SuperView (155° diagonal, ~140° horizontal)
- Bitrate: Maximum (100 Mbps)
- Sharpness: Medium (High causes over-sharpening, harder to stitch)
- Color: Flat (easier to color match in post)
- White Balance: Locked (prevents drift between cameras)
- ISO Min: 100, ISO Max: 1600 (reduce noise)
- Shutter: Auto or 1/120s for 60fps
- EV Comp: 0
- ProTune: ON

**Storage Requirements**:
- 5.3K60 @ 100 Mbps = ~12.5 MB/s per camera
- 8 cameras = 100 MB/s total
- 512GB card = ~85 minutes per camera
- For 2-hour show: Need card swap or continuous offload

---

## 2. SYNCHRONIZATION SYSTEM

### Hardware Sync Solution

**Option A: Camdo Blink Hub + GoPro Labs (RECOMMENDED)**

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|-------------|-------|--------|-------|
| CamDo Blink Hub | Blink Hub 10-pack | 1 | $450 | $450 | camdosolutions.com | Syncs up to 10 GoPros via USB |
| USB-C to USB-C Cables (1m) | Anker PowerLine III | 8 | $10 | $80 | Amazon | Connect GoPros to Blink Hub |
| GoPro Labs Firmware | Free download | - | Free | $0 | gopro.com/labs | Enables external trigger, QR settings |
| Micro HDMI to HDMI Cables (optional) | Cable Matters | 2 | $10 | $20 | Amazon | For live monitoring (2 cameras) |

**Subtotal: $550**

**Option B: DIY Arduino Trigger (Budget Alternative - $150)**

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| Arduino Uno R3 | 1 | $25 | $25 | Microcontroller for sync trigger |
| 8-Channel Relay Module | 1 | $12 | $12 | Triggers GoPro record pins |
| GoPro USB-C Breakout Cables | 8 | $8 | $64 | Custom or DIY (solder USB-C connectors) |
| Power Supply 5V 3A | 1 | $12 | $12 | Powers Arduino + relays |
| Breadboard + Jumper Wires | 1 | $15 | $15 | Prototyping |
| Enclosure | 1 | $22 | $22 | Protect electronics |

**DIY Subtotal: $150** (Saves $400 but less reliable, requires technical skills)

**Recommendation**: Use **Camdo Blink Hub** for reliability. DIY acceptable for initial testing only.

### Sync Strategy

**Camdo Blink Hub Method**:
1. Connect all 8 GoPros to Blink Hub via USB-C
2. Flash GoPro Labs firmware to all cameras (enables USB trigger)
3. Blink Hub sends simultaneous trigger signal to all cameras
4. Cameras start/stop recording in sync (±1 frame accuracy)
5. Verify sync in post using audio or visual cue (clapper board)

**Sync Accuracy**:
- Expected: ±1 frame @ 60fps (±16.7ms)
- Sufficient for stitching (human perception: ~30ms)
- Post-processing can align frames using timecode or audio sync

---

## 3. RIG STRUCTURE & MOUNTING

### Custom Circular Rig (Fabrication Required)

**Option A: Aluminum Extrusion Frame (RECOMMENDED)**

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| 80/20 Aluminum Extrusion (10 series, 1" x 1") | 1010-S (48" length) | 4 | $35 | $140 | 8020.net | Cut to size for octagon frame |
| 80/20 Corner Brackets | 4112 | 16 | $4 | $64 | 8020.net | Connect extrusions |
| 80/20 T-Nuts (25-pack) | 3382 | 2 packs | $12 | $24 | 8020.net | Fastening system |
| M6 Socket Head Screws | Various lengths | 1 pack (50) | $18 | $18 | McMaster-Carr | Assembly hardware |
| Camera Mounting Plates (1/4"-20) | Custom CNC or 3D print | 8 | $15 | $120 | Local machine shop | Attach GoPros to rig |
| Central Hub Plate (8" diameter) | 1/4" aluminum plate | 1 | $45 | $45 | Local metal supplier | Center structure, holds octagon |
| Tripod Mount Adapter (1/4"-20 to 3/8"-16) | Manfrotto 120 | 1 | $12 | $12 | B&H | Mount rig to tripod |
| Laser Cutting / CNC Service | Custom design | 1 | $200 | $200 | SendCutSend, local shop | Cut aluminum plates precisely |
| Hardware (bolts, washers, etc.) | Assorted M6 | 1 kit | $30 | $30 | McMaster-Carr | Assembly |

**Subtotal: $653**

**Option B: 3D Printed Frame (Budget Alternative - $250)**

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| 3D Printing Service (PLA or PETG) | 1 rig | $150 | $150 | Shapeways, local maker space, or own printer |
| Aluminum Center Plate | 1 | $45 | $45 | Metal center for rigidity |
| Camera Mounting Hardware | 8 sets | $5 | $40 | 1/4"-20 screws and spacers |
| Assembly Hardware | 1 kit | $15 | $15 | Screws, nuts, washers |

**3D Printed Subtotal: $250** (Cheaper but less rigid, acceptable for prototype)

**Design Specifications**:
- **Geometry**: Regular octagon, 8 camera mounting points
- **Radius**: 100mm from center to camera lens entrance pupil
- **Camera spacing**: ~78.5mm arc distance between adjacent cameras
- **Camera orientation**: Each camera points outward at 45° intervals (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- **Vertical alignment**: All camera sensors on same horizontal plane (±0.5mm tolerance)
- **Rigidity**: No flex under 5kg load test
- **Material**: 6061-T6 aluminum (recommended) or PETG 3D print (budget)

### Rig Mounting Hardware

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| Heavy-Duty Tripod | Manfrotto MT055XPRO3 | 1 | $280 | $280 | B&H | Carbon fiber, supports 20 lbs |
| Fluid Head | Manfrotto MVH502AH | 1 | $220 | $220 | B&H | Smooth panning for setup (optional) |
| Sandbags (15 lbs each) | Impact Saddle Sandbag | 2 | $25 | $50 | B&H | Stabilize tripod |

**Mounting Subtotal: $550** (Can use cheaper tripod, see budget options)

**Total Rig Structure & Mounting: $1,203** (Aluminum) or **$800** (3D Printed + cheap tripod)

**Budget Tripod Alternative**: Neewer Carbon Fiber Tripod ($120) - reduces cost by $380

---

## 4. POWER SYSTEM

### Camera Power Options

**GoPro Hero 12 Battery Life**:
- 5.3K60: ~40-50 minutes per battery
- For 2-hour show: Need battery swap or continuous power

**Option A: Battery Swap (Simple, Lower Cost)**

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| GoPro Enduro Battery | ADBAT-011 | 24 (3 per camera) | $25 | $600 | B&H, GoPro.com | Extended life in cold, 1720mAh |
| Dual Battery Charger | GoPro Dual Battery Charger | 4 | $20 | $80 | B&H | Charge 8 batteries simultaneously |

**Battery Swap Subtotal: $680**

**Battery Swap Workflow**:
- Install fresh battery in each camera before event
- At ~40 min mark: Pause recording, swap all 8 batteries (2-3 min downtime)
- Resume recording
- Have crew member ready with fresh batteries

**Option B: Continuous USB-C Power (Recommended for Long Events)**

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| USB-C Power Delivery Charger (100W, 8-port) | 1 | $120 | $120 | Powers 8 GoPros continuously |
| USB-C to USB-C Cables (3m, right-angle) | 8 | $15 | $120 | Connect charger to cameras |
| GoPro Battery (keep in camera) | 8 | Included | $0 | Provides redundancy if power fails |
| Cable Management (Velcro straps, clips) | 1 kit | $20 | $20 | Keep cables tidy |

**Continuous Power Subtotal: $260** (Cheaper and no downtime, but requires cables)

**Recommendation**: **Option B (Continuous Power)** for production, **Option A (Batteries)** for quick setup/teardown.

---

## 5. STORAGE & RECORDING

### Primary Storage (In-Camera)

Already included in Camera section (8 × 512GB MicroSD cards = $680)

### Backup Storage & Offload

| Item | Part Number | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| NVMe SSD 2TB (Samsung 980 Pro) | MZ-V8P2T0B/AM | 2 | $180 | $360 | B&H, Amazon | Fast offload via Thunderbolt |
| Thunderbolt 3 NVMe Enclosure | OWC Envoy Pro | 2 | $80 | $160 | B&H, OWC | 1500 MB/s read/write |
| Portable HDD 5TB (Backup) | WD My Passport | 1 | $120 | $120 | Amazon | Redundant backup, slower but cheaper |
| SD Card Case | Pelican 0915 | 1 | $20 | $20 | B&H | Protect cards during transport |

**Storage Subtotal: $660**

### Data Management

**Recording Duration**:
- 8 cameras × 5.3K60 × 100 Mbps = ~800 Mbps total
- 800 Mbps = 100 MB/s
- 1 hour = 360 GB
- 2 hours = 720 GB
- 512GB cards hold ~85 min per camera

**Workflow**:
1. Shoot event (max 90 min per card)
2. Offload all 8 cards to NVMe SSD (~30 min for 400GB)
3. Copy to backup HDD (~60 min)
4. Verify files (check all 8 camera folders)
5. Clear cards for next shoot

**Storage Needs for 1 Event (2 hours)**:
- Primary: 720 GB
- Backup: 720 GB
- Working Storage (post-production): 1-2 TB
- Final Output: 50-100 GB (after stitching & compression)

---

## 6. CABLES & CONNECTORS

| Item | Description | Qty | Unit Price | Total | Vendor | Notes |
|------|-------------|-----|------------|-------|--------|-------|
| USB-C to USB-C Cables (1m) | Anker PowerLine III | 8 | $10 | $80 | Amazon | Sync + power (if using Blink Hub + power simultaneously, need 16 total) |
| USB-C to USB-C Cables (3m) | Anker PowerLine III | 8 | $15 | $120 | Amazon | For continuous power option |
| Cable Ties / Velcro Straps | Assorted | 1 pack (100) | $12 | $12 | Amazon | Cable management |
| HDMI Cables (optional) | For monitoring | 2 | $10 | $20 | Amazon | Connect 2 cameras to monitors for live preview |
| Ethernet Cable (for network control, future) | CAT6, 10m | 1 | $15 | $15 | Amazon | Future: Network control of cameras |
| Cable Labels | Brother P-Touch Labels | 1 roll | $18 | $18 | Amazon | Label each camera (Cam 1-8) |
| Gaffer Tape (Black) | Pro-Gaff 2" × 55 yd | 1 roll | $25 | $25 | B&H | Secure cables to rig/floor |

**Cables Subtotal: $290**

---

## 7. CALIBRATION & TOOLS

### Calibration Equipment

| Item | Qty | Unit Price | Total | Vendor | Notes |
|------|-----|------------|-------|--------|-------|
| Checkerboard Calibration Target (A2 size) | 1 | $30 | $30 | Calib.io, custom print | For camera calibration (OpenCV) |
| LED Panel Light (for even calibration lighting) | 1 | $60 | $60 | Amazon | Ensure consistent exposure during calibration |
| Color Checker Passport | 1 | $100 | $100 | B&H, X-Rite | Color calibration across all 8 cameras |

**Calibration Subtotal: $190**

### Tools & Assembly

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| Allen Key Set (Metric) | 1 | $15 | $15 | For 80/20 assembly |
| Digital Caliper (0-150mm) | 1 | $25 | $25 | Verify camera spacing accuracy |
| Small Level (for rig alignment) | 1 | $12 | $12 | Ensure cameras on same horizontal plane |
| Screwdriver Set | 1 | $20 | $20 | General assembly |
| Portable Work Light | 1 | $25 | $25 | Setup in dark venues |

**Tools Subtotal: $97**

**Total Calibration & Tools: $287**

---

## 8. OPTIONAL ACCESSORIES

### Live Monitoring (Optional but Recommended)

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| Small HDMI Monitor (7") | 2 | $120 | $240 | Monitor 2 cameras during shoot for framing |
| Micro HDMI to HDMI Cables | 2 | $10 | $20 | Connect GoPros to monitors |
| Monitor Mounts (Magic Arms) | 2 | $25 | $50 | Attach monitors to rig or tripod |

**Live Monitoring Subtotal: $310**

### Audio Capture (Spatial Audio)

**Note**: GoPros have built-in mics, but for professional spatial audio, add:

| Item | Qty | Unit Price | Total | Notes |
|------|-----|------------|-------|-------|
| Zoom F6 (6-channel recorder) | 1 | $600 | $600 | Records ambisonic or multi-channel audio |
| Sennheiser Ambeo VR Mic | 1 | $1,600 | $1,600 | Ambisonic 360° audio capture |
| XLR Cables (3m) | 2 | $20 | $40 | Connect mic to recorder |
| Boom Pole & Shock Mount | 1 | $150 | $150 | Position mic above rig or in optimal location |

**Spatial Audio Subtotal: $2,390** (Expensive, but critical for immersive VR experience)

**Budget Alternative**: Use GoPro built-in audio (stereo only) for prototype.

---

## COMPLETE BILL OF MATERIALS - ITEMIZED

### CORE SYSTEM (Minimum Viable Rig)

| # | Category | Item | Qty | Unit Price | Total |
|---|----------|------|-----|------------|-------|
| 1 | Camera | GoPro Hero 12 Black | 8 | $399 | $3,192 |
| 2 | Storage | SanDisk 512GB MicroSD (V60) | 8 | $85 | $680 |
| 3 | Sync | Camdo Blink Hub | 1 | $450 | $450 |
| 4 | Cables | USB-C to USB-C (1m, for sync) | 8 | $10 | $80 |
| 5 | Power | GoPro Enduro Batteries | 24 | $25 | $600 |
| 6 | Power | Dual Battery Chargers | 4 | $20 | $80 |
| 7 | Rig | 3D Printed Frame + Hardware | 1 | $250 | $250 |
| 8 | Mount | Budget Tripod (Neewer Carbon) | 1 | $120 | $120 |
| 9 | Mount | Sandbags (2 × 15 lbs) | 2 | $25 | $50 |
| 10 | Backup Storage | NVMe SSD 2TB | 2 | $180 | $360 |
| 11 | Backup Storage | Portable HDD 5TB | 1 | $120 | $120 |
| 12 | Calibration | Checkerboard Target | 1 | $30 | $30 |
| 13 | Calibration | Color Checker | 1 | $100 | $100 |
| 14 | Accessories | Cables, tape, tools, misc | - | - | $200 |
| | | | | **TOTAL** | **$6,312** |

**Core System Budget: ~$6,300**

### PROFESSIONAL UPGRADE (Add these for production-ready system)

| # | Item | Qty | Unit Price | Total | Notes |
|---|------|-----|------------|-------|-------|
| 1 | Aluminum Rig (instead of 3D print) | 1 | +$400 | +$400 | More rigid |
| 2 | Professional Tripod (Manfrotto) | 1 | +$280 | +$280 | Heavier duty |
| 3 | Continuous Power System | 1 | $260 | $260 | No battery swaps |
| 4 | Live Monitoring (2 screens) | 1 | $310 | $310 | Real-time preview |
| 5 | Spatial Audio System | 1 | $2,390 | $2,390 | Ambisonic mic + recorder |
| | | | **ADDITIONAL** | **+$3,640** |

**Professional System Total: $6,312 + $3,640 = $9,952** (~$10K)

---

## BUDGET OPTIMIZATION

### How to Build for Under $5,000

**Cost Reduction Strategies**:

| Item | Standard Cost | Budget Alternative | Savings |
|------|---------------|-------------------|---------|
| GoPro Hero 12 Black (new) | $3,192 | Refurbished or Hero 11 Black | -$800 |
| Camdo Blink Hub | $450 | DIY Arduino Trigger | -$300 |
| Enduro Batteries (24) | $600 | Standard GoPro batteries (16) | -$200 |
| 512GB MicroSD Cards | $680 | 256GB cards (shorter recording) | -$340 |
| 3D Printed Rig | $250 | DIY wood/PVC frame | -$150 |
| Professional Tripod | $400 | Basic photography tripod | -$280 |
| NVMe SSD 2TB (2×) | $360 | Use existing computer storage | -$360 |
| | | **Total Savings** | **-$2,430** |

**Ultra-Budget Configuration: ~$4,500**

**Trade-offs**:
- Refurbished cameras (risk of failure)
- Less storage capacity (more frequent offloads)
- DIY sync (less reliable, requires technical skill)
- Weaker rig structure (less durable)

**Recommendation**: **Don't cut corners on cameras or sync**. These are mission-critical. Save money on rig structure, tripod, storage instead.

---

## BUILD PHASES

### Phase 1: Core Components (Week 1) - $4,500

**Order immediately**:
1. 8× GoPro Hero 12 Black
2. 8× 512GB MicroSD cards
3. Camdo Blink Hub
4. USB-C cables
5. Batteries & chargers

**Start**: Camera testing, sync validation

### Phase 2: Rig Fabrication (Week 2) - $800

**Design & Build**:
1. CAD design of circular rig (Fusion 360, free)
2. Order 3D printing or aluminum fabrication
3. Assemble rig structure
4. Mount cameras to rig

**Test**: Verify camera spacing, alignment

### Phase 3: Storage & Accessories (Week 3) - $1,500

**Order**:
1. NVMe SSDs
2. Backup HDDs
3. Tripod & sandbags
4. Calibration targets
5. Cables & accessories

**Test**: Full system integration, recording workflow

### Phase 4: Testing & Calibration (Week 4) - $0

**Tasks**:
1. Camera calibration (intrinsic & extrinsic)
2. Sync testing (verify ±1 frame accuracy)
3. Test shoot (capture test footage)
4. Basic stitching test (validate overlap, quality)

**Deliverable**: Working prototype, ready for first pilot customer

---

## VENDORS & SUPPLIERS

### Primary Vendors

| Vendor | Items | Website | Notes |
|--------|-------|---------|-------|
| **B&H Photo** | Cameras, lenses, tripods, most accessories | bhphotovideo.com | Reliable, good support, B2B accounts |
| **Amazon** | Cables, batteries, small accessories | amazon.com | Fast shipping, easy returns |
| **CamDo Solutions** | Blink Hub sync system | camdosolutions.com | Specialized GoPro accessories |
| **GoPro.com** | Cameras, batteries, official accessories | gopro.com | Direct from manufacturer |
| **8020.net** | Aluminum extrusion, hardware | 8020.net | Modular aluminum framing |
| **McMaster-Carr** | Hardware, fasteners, tools | mcmaster.com | Industrial supplier, same-day ship |
| **SendCutSend** | Laser cutting, CNC services | sendcutsend.com | Upload CAD, receive cut parts |
| **Shapeways** | 3D printing services | shapeways.com | High-quality 3D prints |

### Local Suppliers

- **Machine shop**: CNC milling for camera mount plates
- **3D printing**: Local maker space or university lab (often free/cheap)
- **Metal supplier**: Local metal distributor for aluminum plates

---

## ASSEMBLY INSTRUCTIONS (High-Level)

### Step 1: Rig Assembly
1. Cut aluminum extrusions to length (8 sides of octagon, ~117mm each)
2. Assemble octagon frame using corner brackets
3. Attach central hub plate
4. Install camera mounting plates at 45° intervals
5. Verify geometry with digital caliper (100mm radius, ±0.5mm)
6. Attach tripod mount to central hub

### Step 2: Camera Installation
1. Flash GoPro Labs firmware to all 8 cameras
2. Configure identical settings on all cameras (see Camera Configuration section)
3. Label cameras 1-8
4. Mount cameras to rig plates using 1/4"-20 screws
5. Align all cameras on same horizontal plane (use level)
6. Route USB-C cables to central hub

### Step 3: Sync System Setup
1. Connect Camdo Blink Hub to power
2. Connect all 8 GoPros to Blink Hub via USB-C
3. Test sync: Press Blink button, verify all cameras start recording
4. Record 10-second test, verify frame alignment in post

### Step 4: Calibration
1. Position checkerboard target 2m from rig
2. Record 30 seconds while slowly moving target through all camera FOVs
3. Extract frames from all 8 cameras
4. Run OpenCV calibration (estimate intrinsic parameters)
5. Estimate extrinsic parameters (camera positions relative to each other)
6. Save calibration files for stitching software

### Step 5: Test Shoot
1. Set up rig in test environment (office, garage, etc.)
2. Record 5-minute test scene with movement and detail
3. Offload footage to NVMe SSD
4. Import into stitching software (Mistika VR, PTGui, or custom)
5. Stitch to equirectangular 360° output
6. Review in VR headset for quality, seam alignment, stereo depth

**Deliverable**: Validated prototype rig ready for pilot customer shoot

---

## TIMELINE & MILESTONES

| Week | Milestone | Cost This Week | Cumulative |
|------|-----------|----------------|------------|
| **Week 1** | Order cameras, sync, batteries | $4,500 | $4,500 |
| **Week 2** | Design & fabricate rig | $800 | $5,300 |
| **Week 3** | Order storage, tripod, accessories | $1,500 | $6,800 |
| **Week 4** | Assembly, calibration, testing | $0 | $6,800 |
| **Week 5** | First test shoot, iterate | $0 | $6,800 |
| **Week 6** | Pilot customer shoot (if validated) | $0 | $6,800 |

**Total Time to Working Prototype**: 4-6 weeks
**Total Cost**: $6,800 (or $4,500 ultra-budget)

---

## RISK MITIGATION

### Component Failures

**Risk**: Camera failure during shoot
**Mitigation**:
- Buy 1-2 spare GoPros (refurbished, $250 each)
- Test all cameras extensively before pilot shoot
- Have backup batteries ready

**Risk**: Sync failure
**Mitigation**:
- Camdo Blink Hub has proven reliability
- Test sync before every shoot
- Fallback: Use audio waveform sync in post (clapper board)

**Risk**: Storage failure (corrupted card)
**Mitigation**:
- Use high-quality MicroSD cards (SanDisk Extreme Pro)
- Test all cards before shoot
- Monitor recording status during shoot (check LED indicators)

### Rig Structural Issues

**Risk**: Rig flexes, cameras misalign during shoot
**Mitigation**:
- Load test rig before shoot (hang weights, check for flex)
- Use aluminum instead of 3D print for production rigs
- Re-calibrate after transport to venue

---

## NEXT STEPS AFTER BOM

1. **Order Phase 1 components** ($4,500) - cameras, sync, batteries
2. **Design rig in CAD** (Fusion 360, free for hobbyists/startups)
3. **Test cameras individually** - verify settings, image quality
4. **Test sync system** - ensure Camdo Blink works with all 8 cameras
5. **Fabricate rig** - 3D print or machine aluminum frame
6. **Assemble & calibrate** - mount cameras, run calibration
7. **Test shoot** - capture test footage, stitch, review in VR
8. **Iterate** - fix issues, improve workflow
9. **Pilot customer shoot** - real event capture
10. **Learn & improve** - feedback loop for Production Rig v2

---

## CONCLUSION

**Total Investment for Prototype Rig**: **$6,800** (can reduce to $4,500 with budget options)

**What You Get**:
- ✅ Functional 8-camera 360° stereoscopic capture system
- ✅ Synchronized recording (±1 frame accuracy)
- ✅ 5.3K resolution per camera (stitches to 8K+ output)
- ✅ 90-minute continuous recording (or unlimited with continuous power)
- ✅ Professional-quality output suitable for Broadway/theater pilots

**ROI**: At $6,000/event pricing, you recoup the entire rig cost after **1.1 events**. Every event after that is profit (minus crew and post-production costs).

**Ready to build?** Start with Phase 1 component orders this week.
