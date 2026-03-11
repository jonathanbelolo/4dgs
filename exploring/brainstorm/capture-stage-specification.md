# Capture Stage Hardware Specification
## 120-Camera Volumetric Capture Stage for Dynamic Beta Splatting

**Version:** 1.0
**Date:** March 2026
**Purpose:** Studio capture of human performances (singing, dancing) for 6DOF VR delivery with full CGI relighting

---

## 1. Cameras

### Model: Emergent Vision Technologies Zenith HZ-25000-SBS (120 units)

| Parameter | Value |
|-----------|-------|
| Sensor | Sony IMX947 (Pregius S 4th generation, stacked BSI) |
| Resolution | 26.4 MP, 5136 x 5136 (square format) |
| Pixel size | 5.48 µm |
| Optical format | 2.5" (39.7mm diagonal) |
| Shutter | Global shutter (charge-domain) |
| Bit depth | 8 / 10 / 12-bit selectable |
| Frame rate | 423 fps @ 8-bit, 383 fps @ 10-bit, 277 fps @ 12-bit |
| Target capture rate | 60 fps @ 12-bit (with headroom for 120 fps) |
| Interface | 100GigE QSFP28 (fiber optic) |
| Synchronization | IEEE 1588 PTPv2 (sub-microsecond across all cameras) |
| Protocol | GigE Vision 3.0, RDMA |
| HDR | 1-shot HDR supported |
| Binning | V2xH1, V1xH2, 2x2 modes available |
| Enclosure | E2 (CNC machined) with EF lens mount |
| GPIO | Trigger input/output for strobe synchronization |

**Why this sensor:**
- The 5.48µm pixels collect **4x more photons per pixel** than the 2.74µm alternatives (IMX925/928/929), which is critical when using strobed lighting with 500µs pulse widths
- 26MP x 120 cameras = **3.12 billion pixels per frame** - massive multi-view density
- Square format (5136x5136) provides equal horizontal and vertical coverage for full-body capture
- The IMX947 is effectively the IMX927 (105MP flagship) with 2x2 binning baked into silicon - same 2.5" optical format, same image quality per pixel, but 4x the sensitivity
- 277fps @ 12-bit gives 4.6x headroom above the 60fps target, enabling future slow-motion capture without hardware changes

**Why Emergent:**
- Emergent is the **only manufacturer shipping 100GigE machine vision cameras** as of early 2026
- Proven in volumetric capture deployments (documented case studies with 48+ camera rigs)
- RDMA support enables GPUDirect data path to DGX B300 with near-zero CPU overhead
- EF-mount electronic adapters with software-controlled iris and focus across all cameras

**Manufacturer:** Emergent Vision Technologies, Port Coquitlam, BC, Canada
**Website:** emergentvisiontec.com

---

## 2. Lenses

### Model: Sigma 35mm f/1.4 DG HSM Art (EF mount) - 120 units

| Parameter | Value |
|-----------|-------|
| Focal length | 35mm |
| Maximum aperture | f/1.4 |
| Image circle | 43.3mm (full-frame) |
| Mount | Canon EF |
| Sensor coverage | Full 39.7mm IMX947 diagonal with margin |
| Elements / Groups | 13 elements / 11 groups |
| Minimum focus | 30cm |
| Weight | 665g |
| Street price | ~$800 USD |

**Why this lens:**
- Full-frame 43.3mm image circle fully covers the IMX947's 39.7mm sensor diagonal with no vignetting
- f/1.4 maximum aperture provides sufficient light gathering for 500µs strobed illumination
- 35mm focal length at ~3.5m camera-to-subject distance yields a FOV that frames a ~4m wide, ~2.5m tall capture volume
- Sigma Art line delivers cinema-quality optical performance at 1/6th the cost of cinema primes
- EF mount interfaces with Emergent's electronic adapter for software-controlled iris and focus
- At 120 units, photo primes ($96K total) vs cinema lenses ($600K+) saves ~$500K

**Alternative:** Sigma 24mm f/1.4 DG HSM Art for wider FOV if cameras are placed closer to the subject

**Cross-polarization filter:** Hoya HD CIR-PL (77mm) on each lens - $60-80 each
- **Permanently mounted** — never removed or adjusted during a session
- Polarization switching is controlled at the light source via dual LED banks (see Section 4)
- When paired with polarized lights (Bank A): cross-polarized capture (diffuse only)
- When paired with unpolarized lights (Bank B): CPL acts as ~1 stop ND filter (full appearance captured)
- The ~1 stop light loss during non-polarized capture is compensated by strobe overdrive

---

## 3. Camera Arrangement

### Geometry: Hemisphere (360° horizontal + dome cap)

```
                    ╭─── Dome Cap (16 cameras) ───╮
                   ╱    looking downward at ~60°    ╲
                  ╱                                  ╲
         Ring 5 (16 cameras, 3.0m height, 22.5° spacing)
        ╱                                              ╲
  Ring 4 (20 cameras, 2.4m height, 18° spacing)
  Ring 3 (24 cameras, 1.8m height, 15° spacing)    ← chest/face height
  Ring 2 (24 cameras, 1.2m height, 15° spacing)    ← hip height
  Ring 1 (20 cameras, 0.5m height, 18° spacing)    ← ground level
        ╲______________________________________________╱
                     4m diameter capture volume
```

| Ring | Height | Camera Count | Angular Spacing | Purpose |
|------|--------|-------------|----------------|---------|
| 1 | 0.5m | 20 | 18° | Feet, lower body |
| 2 | 1.2m | 24 | 15° | Hips, hands at sides |
| 3 | 1.8m | 24 | 15° | Chest, face (highest density ring) |
| 4 | 2.4m | 20 | 18° | Head, raised arms |
| 5 | 3.0m | 16 | 22.5° | Above head, jumping |
| Dome | 3.5m+ | 16 | ~22.5° | Top of head, shoulders from above |
| **Total** | | **120** | | |

### Capture Volume
- **Diameter:** ~4m (sufficient for full-body dance, spinning, paired movement)
- **Height:** ~2.5m (standing performer with arms raised)
- **Camera-to-center distance:** ~3.5m (from center of capture volume to camera ring)
- **FOV per camera at 35mm:** ~55° horizontal × 55° vertical (square sensor)

### Structural Rig
- Custom aluminum truss hemisphere, ~7m outer diameter
- Camera mounting plates with 3-axis fine adjustment (pan, tilt, roll)
- Cable management channels integrated into truss structure
- Modular assembly for maintenance access

---

## 4. Lighting

### Architecture: Dual-Bank GPIO-Synchronized Strobed Machine Vision LEDs

The lighting system uses two independent banks of LED fixtures — one with linear polarizing film (Bank A) and one without (Bank B) — to enable electronic switching between cross-polarized and non-polarized capture modes. CPL filters remain permanently mounted on all cameras.

### 4.1 Strobe Controllers

**Model:** Gardasoft RT820F-20 (36 units)

| Parameter | Value |
|-----------|-------|
| Channels | 8 independent per unit |
| Pulsed current | Up to 20A per channel |
| Continuous current | Up to 3A per channel |
| Pulse width range | 1 µs to 999 ms (1 µs steps) |
| Trigger inputs | 8x opto-isolated (3V-24V) |
| Timing repeatability | 0.1 µs for pulses up to 10ms |
| Interface | Ethernet (RT820F) |
| Overdrive | SafeSense technology (up to 10x continuous rating) |
| Price | ~$2,000-3,000 each |

- 36 units × 8 channels = 288 channels (covers both LED banks)
- Bank A controllers (18 units): drive polarized LED fixtures
- Bank B controllers (18 units): drive unpolarized LED fixtures
- Triggered from PTP master camera's GPIO output
- Operator selects active bank via Gardasoft software — no physical changes needed between capture phases
- OLAT mode: Bank A controllers fire individual channels in sequence (one light at a time)

### 4.2 LED Fixtures

**Quantity:** 300-400 units total, arranged as two co-located banks

| Bank | Fixtures | Polarization | Purpose |
|------|----------|-------------|---------|
| **Bank A** | 150-200 | Linear polarizing film on each fixture | Cross-polarized capture (A-pose material reference, OLAT) |
| **Bank B** | 150-200 | No polarization | Non-polarized capture (dynamic performance) |

Both banks are distributed around and above the capture volume in matching positions. Each Bank A fixture is co-located with a Bank B fixture to ensure identical illumination geometry regardless of which bank is active.

**Requirements (both banks):**
- CRI: 95+ (essential for accurate skin tone reproduction and material decomposition)
- CCT: 5600K daylight (matches standard photogrammetry workflows)
- Output: Diffused (mounted behind light diffusion fabric)
- Form factor: Bar lights or compact panels for dense packing on the hemisphere frame

**Recommended options:**
- Smart Vision Lights OverDrive series (integrated strobe driver, SafeStrobe technology)
- CCS high-power LED panels (PF series for strobing)
- Advanced Illumination high-CRI bar lights
- Custom LED arrays using Luminus or Cree high-CRI LEDs with Gardasoft control

**Strobe parameters:**
- Pulse width: **500 µs** (1/2000s) - freezes all human motion including fast dance
- Duty cycle at 60fps: 3% (500µs / 16,667µs per frame)
- Effective brightness: With 10x SafeSense overdrive, a 50W continuous LED delivers ~1,500W equivalent instantaneous output during each pulse

### 4.3 Polarization Architecture

**On cameras (permanent):** Circular polarizing filter (CPL) on each lens, never removed or adjusted during a session.

**Polarization switching is controlled entirely at the light source:**

| Active Bank | Light Polarization | Camera CPL | Result |
|------------|-------------------|-----------|--------|
| Bank A | Linear polarized | Always on | **Cross-polarized:** specular eliminated, diffuse only |
| Bank B | Unpolarized | Always on | **Full appearance:** CPL acts as ~1 stop ND filter only |

**Why CPL on unpolarized light is harmless:** When incoming light is unpolarized (Bank B), the CPL blocks ~50% of photons uniformly across all wavelengths and directions — equivalent to an ND2 neutral density filter. No color shift, no selective filtering of diffuse vs specular, no distortion. The 1-stop loss is trivially compensated by increasing Bank B strobe overdrive from 10x to 12x.

**Key advantage:** Zero manual intervention between capture phases. The operator switches banks electronically via the Gardasoft Ethernet interface. The performer can transition from A-pose to dynamic performance without any crew touching the cameras or lights.

### 4.4 OLAT (One Light At A Time) Capability

The Gardasoft controllers' per-channel addressing enables OLAT sequences:
- Each of the 150-200 Bank A fixtures fires individually in sequence
- One frame per light at 60fps = 2.5-3.3 seconds for a complete sweep
- Combined with 120 cameras = 18,000-24,000 unique light-view pairs per sweep
- Used during static A-pose capture for comprehensive BRDF/BCSDF measurement
- Provides near-complete light transport matrix of the performer (comparable to USC Light Stage methodology)

### 4.5 Thermal & Power

| Metric | Strobed (this design) | Continuous equivalent |
|--------|----------------------|----------------------|
| Average power draw | 5-8 kW (both banks installed) | 50-80 kW |
| Heat load on talent | Near zero | Uncomfortable |
| HVAC requirements | Standard studio | Dedicated cooling |
| Makeup/sweat impact | None | Significant |
| Session duration limit | Unlimited | 30-60 min before talent fatigue |

Note: Only one bank fires at a time, so active power draw during capture remains 3-5 kW despite the doubled fixture count.

---

## 5. Background

- **Material:** Matte gray painted cyclorama (seamless floor-to-wall curve)
- **Color:** 18% neutral gray (Munsell N5 or equivalent)
- **Why gray, not green:**
  - Green screen causes color spill on skin and clothing edges
  - Gray provides a neutral reference for color calibration
  - With 120 cameras and Grounded SAM 2 segmentation, chroma keying is unnecessary - semantic masking achieves superior results
- **Dimensions:** Minimum 6m diameter circle (encloses the 4m capture volume with margin)

---

## 6. Networking

### 6.1 Studio Side

| Component | Qty | Specification |
|-----------|-----|--------------|
| QSFP28 fiber optic cables | 120 | OM3/OM4 multimode, camera to switch |
| 100GigE aggregation switches | 3 | NVIDIA/Mellanox Spectrum-4 SN5600 (64x 100GigE ports) |
| 400GigE uplink modules | 2-4 | QSFP-DD, switch to dark fiber interface |

**Switch allocation:**
- Switch 1: 48 cameras (Rings 1-2)
- Switch 2: 48 cameras (Rings 3-4)
- Switch 3: 24 cameras (Ring 5 + Dome) + uplink ports to data center

**Aggregate bandwidth:** 120 cameras × 100 Gbps = 12 Tbps total. Each switch handles 4.8 Tbps, well within the Spectrum-4's 51.2 Tbps switching capacity.

### 6.2 Dark Fiber Link

- Existing dedicated fiber infrastructure to data center (user-confirmed)
- DWDM or 400GigE point-to-point connection
- Used for batch transfer of captured data, not real-time streaming
- Round-trip latency: <1ms at typical metro distances

---

## 7. Compute & Storage (Data Center)

### 7.1 GPU Server

**Model:** NVIDIA DGX B300 (Blackwell Ultra) - 1 unit

| Parameter | Value | Source |
|-----------|-------|--------|
| GPUs | 8x NVIDIA B300 SXM | Confirmed (NVIDIA) |
| GPU memory | 288 GB HBM3e per GPU | Confirmed (NVIDIA) |
| Total GPU memory | 2.1 TB (NVIDIA official) | NVIDIA DGX B300 datasheet |
| Memory bandwidth | 8 TB/s per GPU | Confirmed |
| FP4 dense compute | 108 PFLOPS (system) | NVIDIA official |
| FP4 sparse compute | 144 PFLOPS (system) | NVIDIA official |
| CPU | Intel Xeon 6776P | Confirmed |
| Interconnect | NVLink 5, 1.8 TB/s per GPU | Confirmed |
| System power | ~14 kW | NVIDIA official |
| Cooling | Liquid cooling required | Confirmed |
| Form factor | 10U | Confirmed |
| Availability | Shipping since January 2026 | Confirmed |

### 7.2 Storage

**Type:** NVMe-over-Fabrics (NVMe-oF) storage array

| Parameter | Requirement |
|-----------|-------------|
| Sustained sequential write | ≥ 90 GB/s |
| Capacity | 500 TB+ |
| Interface | 200/400GigE RDMA |
| GPUDirect Storage | Required (cuFile API) |

**Options:** Pure Storage FlashBlade//S, VAST Data, WekaFS, or DDN AI400X2

**Data volume per session:**
- Per frame: 26MP × 12-bit × 120 cameras ≈ 5.6 GB raw
- Per second (60fps): ~336 GB/s
- Per minute: ~20 TB
- 10-minute performance: ~200 TB raw (before compression and multi-resolution storage)

### 7.3 Network Interface

- 2-4x NVIDIA ConnectX-7 or ConnectX-8 NICs (200/400GigE)
- RDMA enabled for GPUDirect data path
- Connected to studio aggregation switches via dark fiber

### 7.4 Ingest Pipeline

Real-time on the DGX B300 during capture:

1. Camera data arrives via RDMA → GPU VRAM (bypasses CPU and system RAM)
2. GPU downscales each frame:
   - Full resolution: 5136x5136 (26MP) → write to NVMe
   - 4K equivalent: ~2568x2568 (~6.6MP) → write to NVMe
   - 2K equivalent: ~1284x1284 (~1.6MP) → write to NVMe
3. All writes via GPUDirect Storage (cuFile API) - GPU VRAM → NVMe, no CPU copy
4. Downscaling uses NVIDIA NPP (Performance Primitives) or custom CUDA kernels

---

## 8. Budget Summary

| Category | Low Estimate | High Estimate | Notes |
|----------|-------------|--------------|-------|
| **Cameras** | | | |
| 120x HZ-25000-SBS bodies | $1,800,000 | $2,400,000 | Volume discount from Emergent |
| 120x Sigma 35mm f/1.4 Art | $96,000 | $96,000 | $800 each |
| 120x EF electronic adapters | $12,000 | $24,000 | Emergent accessory |
| **Lighting** | | | |
| 36x Gardasoft RT820F-20 | $72,000 | $108,000 | 18 per bank, $2,000-3,000 each |
| 300-400x LED fixtures (dual bank) | $120,000 | $240,000 | 150-200 per bank, machine vision grade, high CRI |
| Polarization kit | $13,000 | $20,000 | Film for Bank A + 120x CPL filters (permanent) |
| Diffusion materials | $4,000 | $10,000 | Diffusion fabric/tissue, both banks |
| **Structure** | | | |
| Hemisphere truss rig | $50,000 | $100,000 | Custom aluminum, modular |
| Cyclorama | $10,000 | $30,000 | Matte gray painted cyc |
| **Networking** | | | |
| 3x Spectrum-4 switches | $90,000 | $150,000 | 64-port 100GigE |
| 120x QSFP28 fiber cables | $12,000 | $24,000 | $100-200 each |
| Uplink optics | $5,000 | $10,000 | 400GigE QSFP-DD modules |
| **Compute** | | | |
| 1x DGX B300 | $700,000 | $800,000 | Contact NVIDIA/Lambda |
| NVMe-oF storage (500TB+) | $100,000 | $200,000 | High-throughput array |
| ConnectX NICs + cabling | $10,000 | $20,000 | RDMA enabled |
| **Misc** | | | |
| Power distribution | $10,000 | $20,000 | PDUs, circuits |
| Calibration tools | $2,000 | $5,000 | ChArUco boards, ColorChecker |
| Contingency (5%) | $165,000 | $225,000 | |
| | | | |
| **Total** | **$3,280,000** | **$4,502,000** | |

---

## 9. Key Vendor Contacts

| Vendor | Product | Contact |
|--------|---------|---------|
| Emergent Vision Technologies | HZ-25000-SBS cameras | emergentvisiontec.com/contact |
| NVIDIA Enterprise | DGX B300 | nvidia.com/en-us/data-center/dgx-b300 |
| Gardasoft Vision | RT820F-20 controllers | gardasoft.com |
| Smart Vision Lights | OverDrive LED fixtures | smartvisionlights.com |
| Sigma Corporation | 35mm f/1.4 Art (volume) | sigma-global.com |
| Pure Storage / VAST Data | NVMe-oF arrays | purestorage.com / vastdata.com |

---

## 10. References

### Verified Hardware
- Emergent HZ-25000-SBS: emergentvisiontec.com/products/zenith-hz-100gige-cameras-rdma-area-scan/
- Sony IMX947 sensor: sony-semicon.com/en/products/is/industry/gs/imx927-937.html
- Gardasoft RT8XX: gardasoft.com/LED-Controllers/RT-PP/RT8XX.aspx
- DGX B300: nvidia.com/en-us/data-center/dgx-b300

### Reference Volumetric Capture Stages
- Volucap (Babelsberg): 235 ARRI SkyPanels, 32 cameras, arri.com/news-en/arri-takes-part-in-volucap-volumetric-studio
- Google Relightables: 331 LEDs at 180Hz, 90 cameras, augmentedperception.github.io/therelightables
- Microsoft MRCS: 106 cameras, variety.com/2018/digital/features/microsoft-mixed-reality-capture-behind-the-scenes
- Infinite Realities (Superman 2025): 192 cameras, 4DGS at 48fps, beforesandafters.com
- ESPER LightCage: esperhq.com/product/lightcage-scanning-rig
