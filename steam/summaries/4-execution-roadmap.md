# vZero: 30 Months to Launch

## Execution Roadmap

**Classification:** Operational Plan — Confidential
**Version:** 1.0

---

## Overview

This roadmap covers the 30-month path from project kickoff (Month 0) to the opening of the first vZero Flagship (Month 30). All timelines are expressed as months from the project start date, making this plan independent of calendar dates.

The critical path runs through three parallel workstreams that must converge at launch:
1. **Content** — artist contract, volumetric capture, 4DGS processing, interactive authoring, spatial audio
2. **Technology** — rendering engine, multi-viewer streaming, WFS integration, content tools
3. **Venue** — site selection, lease, design, construction, hardware installation, testing

These workstreams are largely independent until Phase 3 (Integration), where the content must run on the technology in the venue. Delays in any one workstream delay launch.

---

## Phase 0: Foundation (M0 – M4)

**Duration:** 4 months
**Capital deployed:** $2M (of $40M Series A)
**Team size:** 5–8

### Deliverables

| # | Deliverable | Owner | Target |
|---|------------|-------|--------|
| 0.1 | **Secure Steam Frame dev kits** — minimum 6 units for engineering team | Founding Partner | M1 |
| 0.2 | **Execute artist LOI** — Letter of Intent with Tier-1 artist and their label, covering volumetric performance rights, advance terms, and capture scheduling | CEO / Music Counsel | M3 |
| 0.3 | **Hire Technical Director** — single senior hire to own the full technology architecture | CEO | M2 |
| 0.4 | **Volumetric capture venue scouting** — evaluate 4DViews (Grenoble), Metastage (LA), Dimension Studio (London) for Show 1 capture facility | Technical Director | M4 |
| 0.5 | **Legal entity formation** — incorporate vZero Holdings, Publisher, and Retail entities | Legal Counsel | M2 |
| 0.6 | **Begin real estate search** — engage commercial real estate brokers in target cities for 900–1,100 m² (~10,000–12,000 sq ft) entertainment-zoned spaces | COO | M3 |

### Decision Gate (M4)

**Gate 0: Go/No-Go on Series A close**
- Artist LOI signed?
- Steam Frame dev kits in hand?
- Technical Director hired?
- At least 2 viable venue sites identified?

If all four: close Series A and advance to Phase 1.
If artist LOI not signed: pause — no point building a venue without confirmed content.

---

## Phase 1: Proof of Concept (M5 – M11)

**Duration:** 7 months
**Capital deployed:** $8M cumulative
**Team size:** 15–20

### Workstream 1: Technology

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 1.1 | **Hire core engineering team** (12–15 engineers) | 4DGS rendering, systems/infra, spatial audio, VR/SteamVR, tools | M7 |
| 1.2 | **4DGS single-viewer prototype** | Render a DBS/UBS-7D scene on a Steam Frame via Wi-Fi 7 from a local GPU server. Measure motion-to-photon latency. | M9 |
| 1.3 | **Multi-viewer prototype** | Extend to 4 simultaneous Steam Frames rendering independent viewpoints from a shared 4DGS scene on a 2-GPU server. | M11 |
| 1.4 | **WFS audio prototype** | Secure a small-scale Holoplot evaluation kit (16–32 drivers). Demonstrate synchronized audio-visual playback with a 4DGS scene on a Steam Frame. | M10 |

### Workstream 2: Content

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 1.5 | **Execute full artist contract** | Move from LOI to binding agreement. Finalize advance, royalty terms, capture schedule, likeness rights. | M7 |
| 1.6 | **Tech demo capture** | 3-minute volumetric capture of the artist at a partner studio (4DViews or Metastage). No interactive elements — pure 4DGS visual quality proof. | M9 |
| 1.7 | **Tech demo processing** | Process the 3-minute capture through the DBS/UBS-7D pipeline. Produce a viewable 4DGS asset on Steam Frame. | M11 |

### Workstream 3: Venue

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 1.8 | **Site selection — Flagship 1** | Select and sign LOI for the first venue location. Shortlist to final 2 candidates by M8; sign by M10. | M10 |
| 1.9 | **Architectural design — begin** | Engage architecture firm specializing in entertainment/acoustic spaces. Begin schematic design for the performance room layout. | M11 |

### Decision Gate (M11)

**Gate 1: Technology viability**
- Can we render 4DGS on a Steam Frame at acceptable visual quality (>28 dB PSNR)?
- Is motion-to-photon latency <25ms over Wi-Fi 7?
- Does multi-viewer rendering scale (4 headsets from 2 GPUs)?
- Does the tech demo capture look lifelike on the Steam Frame?

If yes on all: proceed to full production.
If latency exceeds 25ms: evaluate wired fallback (USB-C tether as interim solution).
If visual quality is insufficient: evaluate whether the capture rig or the rendering pipeline is the bottleneck.

**This gate is the highest-risk moment in the project.** If the core technical claim — lifelike 4DGS streamed wirelessly to a Steam Frame — does not hold, the entire venture model must be reconsidered. The $8M invested to this point is the cost of proving or disproving the thesis.

---

## Phase 2: Production (M12 – M20)

**Duration:** 9 months
**Capital deployed:** $25M cumulative
**Team size:** 25–35 (including content production crew)

### Workstream 1: Technology

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 2.1 | **12-viewer rendering engine** | Scale from 4-viewer prototype to full 12-viewer server, targeting Rubin-class GPUs | M15 |
| 2.2 | **Interactive overlay SDK** | Build the SDK that allows interactive elements (Unreal Engine) to composite with the server-streamed 4DGS feed, including depth-correct occlusion | M17 |
| 2.3 | **Content authoring toolchain v1** | Editorial tools for sequencing 4DGS takes, spatial audio mixing for WFS, interactive scripting, preview-on-headset workflow | M18 |
| 2.4 | **WFS integration — full scale** | Order and receive full Holoplot array (1,200+ drivers per room). Begin integration with the rendering pipeline on-site or in a test facility. | M20 |

### Workstream 2: Content

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 2.5 | **Full capture sessions** | 3–4 months of studio time with the artist. Multiple performances, wardrobe changes, interactive sequences. Target: 60+ minutes of raw 4DGS material for a 20-minute show. | M12–M15 |
| 2.6 | **4DGS processing — full show** | Process all captured material through the pipeline. Quality iteration, selective re-capture if needed. | M15–M18 |
| 2.7 | **Interactive layer design** | Game designers and the artist's creative team define the interactive mechanics (audience participation, environmental reactions, collective moments). | M14–M17 |
| 2.8 | **Spatial audio production** | Decompose ambisonic captures into WFS sound objects. Author the haptic score (floor transducer mapping). Mix and master. | M17–M20 |

### Workstream 3: Venue

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 2.9 | **Lease execution — Flagship 1** | Sign binding lease. Begin tenant improvement negotiations. | M13 |
| 2.10 | **Construction design finalization** | Complete architectural and MEP (mechanical/electrical/plumbing) design. Acoustic treatment specs. HVAC sizing for GPU heat load. Electrical capacity for WFS array + server rack. | M15 |
| 2.11 | **Construction begins** | Demolition/shell prep, acoustic treatment, electrical rough-in, server room build-out. | M17 |
| 2.12 | **Flagship 2 — site selection begins** | Start scouting for the second location (the city not chosen for Flagship 1). | M20 |

### Decision Gate (M20)

**Gate 2: Production readiness**
- Is the 12-viewer rendering engine stable?
- Is the full show's 4DGS material processed and viewable?
- Is the WFS array ordered and delivery scheduled?
- Is construction on schedule (structural complete, MEP in progress)?

If yes: proceed to integration.
If content is delayed: push launch by equivalent duration (venue can wait; content cannot be rushed without quality loss).
If construction is delayed: evaluate temporary/pop-up venue for soft launch while permanent space completes.

---

## Phase 3: Integration (M21 – M26)

**Duration:** 6 months
**Capital deployed:** $36M cumulative
**Team size:** 30–40 (including construction and installation crews)

This is where the three workstreams converge. The content must run on the technology in the venue.

### Deliverables

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 3.1 | **Server rack installation** | Install 8–12 GPU nodes, NVLink fabric, cooling loop, 100GbE networking in the venue's server room. | M21 |
| 3.2 | **WFS array installation** | Install 1,200+ Holoplot drivers in the performance room walls and ceiling. Wire to spatial audio processors. Acoustic calibration. | M22 |
| 3.3 | **Haptic floor installation** | Install 96–128 transducers in 300 m² modular floor grid. Amplification and control wiring. | M23 |
| 3.4 | **Wi-Fi 7 AP deployment** | Install and optimize 4x enterprise Wi-Fi 7 access points. RF survey and interference testing with WFS array active. | M23 |
| 3.5 | **Full system integration — Show 1** | Load the complete show (4DGS + interactive + spatial audio + haptics) onto the venue hardware. End-to-end playback testing. | M24 |
| 3.6 | **12-viewer stress testing** | Run the full show with 12 simultaneous headsets for 8+ hours continuously. Measure latency, dropped frames, audio sync, thermal stability. | M25 |
| 3.7 | **Content polish** | Iterate on show content based on in-venue testing. Timing adjustments, interactive tuning, audio level balancing, haptic intensity calibration. | M25–M26 |
| 3.8 | **Steam Frame fleet procurement** | Order 40 Steam Frames (12 per room x 2 rooms + spares). Configure with vZero software lock, enterprise face gaskets, belt battery packs. | M24 |
| 3.9 | **Show 2 — begin production** | If Show 1 is on track, begin artist contracting and capture planning for the second show. | M24 |
| 3.10 | **Flagship 2 — lease execution** | Sign lease for second venue. Begin design. | M24 |

### Decision Gate (M26)

**Gate 3: Launch readiness**
- Can the system run 12 headsets for 8 hours without failure?
- Is motion-to-photon latency consistently <25ms?
- Is audio-visual sync within 5ms?
- Is the show content final-locked (no further creative changes)?
- Is the venue construction complete and passed safety inspection?

If yes: proceed to pre-launch.
If system stability issues: extend testing by 1–2 months.
If content not locked: hard-lock content at 90% quality and patch post-launch (the venue can open with v0.9 and update overnight).

---

## Phase 4: Pre-Launch (M27 – M29)

**Duration:** 3 months
**Capital deployed:** $39M cumulative
**Team size:** 45–55 (adding operations and front-of-house staff)

### Deliverables

| # | Deliverable | Detail | Target |
|---|------------|--------|--------|
| 4.1 | **Hire operations team** | Venue manager, 2x technicians, 4x front-of-house/onboarding staff, security, cleaning. | M27 |
| 4.2 | **Staff training** | 4 weeks of training on headset fitting, hygiene protocols, emergency procedures, show operations. | M28 |
| 4.3 | **Soft launch — invite only** | 4 weeks of invite-only sessions (press, industry, VIPs, friends-and-family). 2–4 sessions per day. Collect feedback, identify issues. | M28–M29 |
| 4.4 | **Press and media campaign** | Coordinated reveal with exclusive previews for key outlets (Wired, The Verge, Billboard, Variety). Artist-driven social media campaign. | M29 |
| 4.5 | **Ticketing infrastructure** | Launch vZero website with online booking. Integration with venue ticketing systems. | M28 |
| 4.6 | **B2B licensing framework** | Finalize the "vZero Play" software package and licensing terms for third-party LBE operators. Begin outreach to Sandbox VR, Zero Latency, Hologate, and independent operators. | M29 |
| 4.7 | **Home version — Steam Store submission** | Submit Show 1 home version for Steam Store approval and listing. Target launch date aligned with Flagship opening for cross-promotion. | M29 |

---

## Phase 5: Launch & Steady State (M30+)

**Capital deployed:** $40M (Series A fully deployed)

### Launch Sequence

| Milestone | Timing |
|-----------|--------|
| **Flagship 1 opens to public** | M30. 8 sessions/day initially, ramping to 16/day over 4 weeks. |
| **Show 1 home version on Steam Store** | M30. $29.99 individual purchase. |
| **First operational assessment** | M32. Occupancy rates, customer satisfaction, technical reliability. |
| **Flagship 2 opens** | M36–M38. Second city, 4–6 months after Flagship 1. |
| **Show 2 delivered** | M36. Second AAA spectacle enters rotation at Flagship 1. |
| **Series B raise** | M30–M36. Fund Flagships 3–4, Shows 3–4, Neural Pass pilot. |

### Ongoing Cadence (Post-Launch)

From launch onward, the business operates on a steady rhythm:

| Cadence | Activity |
|---------|----------|
| **Every 6 months** | New AAA show delivered (2 per year) |
| **Every 6 months** | New Flagship opens (2 per year, reaching 12 by Year 6) |
| **Continuous** | B2B licensing expansion |
| **From ~M42** | Neural Pass subscription launches (after 12 months of Flagship operational data) |

---

## Capital Deployment Timeline

| Phase | Period | Cumulative Spend | Primary Uses |
|-------|--------|-----------------|-------------|
| 0 | M0–M4 | $2M | Legal, hiring, scouting, dev kits |
| 1 | M5–M11 | $8M | Engineering team, capture studio, prototyping |
| 2 | M12–M20 | $25M | Full capture, 4DGS processing, construction start, WFS order |
| 3 | M21–M26 | $36M | Hardware installation, integration, system testing |
| 4 | M27–M29 | $39M | Operations hiring, soft launch, marketing |
| 5 | M30+ | $40M | Launch operations, working capital |

---

## Team Growth

| Phase | Period | Engineering | Content/Production | Operations | Corporate | Total |
|-------|--------|------------|-------------------|------------|-----------|-------|
| 0 | M0–M4 | 3 | 0 | 0 | 5 | 8 |
| 1 | M5–M11 | 15 | 3 | 0 | 3 | 21 |
| 2 | M12–M20 | 18 | 10 | 0 | 4 | 32 |
| 3 | M21–M26 | 18 | 8 | 5 | 4 | 35 |
| 4 | M27–M29 | 15 | 5 | 15 | 5 | 40 |
| 5 | M30+ | 15 | 5 | 20 | 5 | 45 |

---

## Key Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Wi-Fi 7 latency exceeds 25ms in venue** | Medium | High | Fallback to USB-C tethered mode (cables on ceiling-mounted retractors). Reduces mobility but preserves experience. |
| **Artist schedule conflict delays capture** | Medium | High | Book capture studio for 6-month window, not just the 3–4 months needed. Buffer absorbs artist rescheduling. |
| **WFS array delivery delay** | Low | Critical | Order 6 months ahead of installation date. Holoplot lead times are 4–6 months for custom installations. |
| **First show exceeds $10M budget** | High | Medium | 10% contingency built in. If exceeded, reduce interactive complexity (still delivers the core 4DGS + WFS experience). |
| **Venue construction delay** | Medium | High | Sign lease with aggressive early-access TI schedule. Engage construction manager in Phase 1, not Phase 2. |
| **Steam Frame supply constraints** | Low | Medium | Founding partner relationship provides priority allocation. Order enterprise quantities 6+ months ahead. |
| **4DGS quality below expectations** | Low | High | Gate 1 (M11) is specifically designed to catch this. $8M sunk cost is the price of knowing. |

---

## First Actions After Green Light

The immediate actions at M0:

1. **Engage legal counsel** — begin entity formation (Holdings, Publisher, Retail)
2. **Draft artist LOI** — define terms for volumetric performance rights
3. **Request Steam Frame dev kits** — through the founding partner's channel
4. **Post Technical Director role** — target hires from the founding partner's platform team, NVIDIA, or major VFX studios (ILM, Weta, Framestore)
5. **Engage commercial real estate broker** — begin site search in target cities

Total cost of these actions: <$200K. No Series A required — funded from pre-seed or the founding partner's commitment.

---

*This document is part of the vZero Strategic Document Suite. See also: Strategic Vision & Market Opportunity, Technical Architecture, and Business Model & Financial Architecture.*
