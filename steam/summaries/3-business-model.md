# vZero: The Dual-Entity Framework

## Business Model & Financial Architecture

**Classification:** Financial Strategy — Confidential
**Date:** February 2026
**Version:** 1.0

---

## I. Corporate Structure

vZero operates as three entities under a single Holding Company. This structure separates high-value intellectual property from operational risk and enables distinct financing strategies for each business.

```
vZero Holdings
├── vZero Publisher (IP Co)
│   - Owns volumetric masters and distribution rights
│   - Negotiates artist contracts and music licensing
│   - Develops capture technology and content tooling
│   - Revenue: content licensing, home sales, B2B royalties
│
├── vZero Retail (Op Co)
│   - Operates flagship venues
│   - Owns/leases real estate and hardware infrastructure
│   - Revenue: ticket sales, F&B, merchandise, corporate rentals
│
└── vZero Finance (SPV — Year 2+)
    - Off-balance-sheet vehicle for home hardware deployment
    - Holds Neural Pass subscriber contracts and leased devices
    - Funded by structured debt facility
```

**Why separate?** The Publisher is valued on IP and recurring revenue — commanding ARR-based multiples characteristic of SaaS and media companies. The Retailer is valued on EBITDA and asset value — commanding lower, asset-based multiples. Combining them would suppress the Publisher's valuation multiple. Separating them also insulates the high-value music rights from the operational liabilities (lease obligations, employment, hardware depreciation) of physical venues.

---

## II. Revenue Architecture

vZero generates revenue through four channels, each utilizing the same 4DGS content asset.

### Channel 1: Flagship Venues (vZero Retail)

**Per-Ticket Economics (2-Room Flagship):**

| Item | Amount | Notes |
|------|--------|-------|
| **Ticket Price** | $100 | Positioned between Sandbox VR ($55) and Sphere average ($217) |
| vZero Retail retention (50%) | $50 | Covers rent, staff, power, hardware amortization |
| vZero Publisher retention (50%) | $50 | Content license fee |
| Publisher → Artist/Label royalty (30% of Publisher gross) | ($15) | Standard LBE sync/performance royalty |
| **Publisher net per ticket** | **$35** | Used to recoup production capex |
| **Retail net per ticket** | **$15–$20** | After operating expenses |

**Venue Throughput (Per 2-Room Flagship):**

| Metric | Conservative | Base | Optimistic |
|--------|-------------|------|-----------|
| Capacity per room | 12 | 12 | 12 |
| Sessions per day (per room) | 14 | 16 | 18 |
| Daily capacity (2 rooms) | 336 | 384 | 432 |
| Occupancy rate | 70% | 80% | 90% |
| **Daily visitors** | **235** | **307** | **389** |
| **Annual visitors** | **86,000** | **112,000** | **142,000** |
| **Annual gross ticket revenue** | **$8.6M** | **$11.2M** | **$14.2M** |

**Operating assumptions:**
- 12 operating hours/day (11 AM – 11 PM)
- 30-minute sessions (20 min experience + 10 min turnover)
- 2 rooms on staggered 15-minute starts (shared onboarding staff)
- 365 days/year (with 10 maintenance days)

**Note on occupancy:** With confirmed A-list artists and only 24 slots per session, demand will significantly exceed supply. ABBA Voyage sustains near-capacity after 4 years with motion-captured avatars of a legacy act. vZero launches with living global superstars whose fanbases number in the tens of millions — and we're offering only ~112,000 tickets per venue per year. 80% occupancy is the conservative baseline; the constraint is supply, not demand. We model 80% to account for operational ramp-up and off-peak hours, but sustained 90%+ is realistic once word-of-mouth builds.

### Channel 2: B2B Licensing (vZero Publisher)

The Publisher licenses vZero content to third-party LBE operators who have Steam Frame/PCVR infrastructure but not the WFS audio or haptic floor.

**Model:** Platform license fee + per-ticket royalty

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Licensed sites | 0 | 30 | 80 |
| Avg. tickets/site/year | — | 8,000 | 10,000 |
| Per-ticket royalty | — | $5.00 | $5.00 |
| Annual platform fee/site | — | $5,000 | $5,000 |
| **Annual B2B revenue** | **$0** | **$1.35M** | **$4.4M** |

**Why conservative?** Third-party LBE operators need to (a) own Steam Frame headsets, (b) have PCVR server infrastructure, and (c) commit to a new content format. Adoption will be gradual. 30 sites by Year 2 is ambitious but achievable if vZero provides a turnkey software solution and the Flagship generates significant press.

### Channel 3: Home B2C (vZero Publisher via Steam Store)

Individual spectacles sold as digital purchases on the Steam Store.

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Steam Frame installed base (est.) | 2M | 4M | 7M |
| vZero conversion rate | 1% | 2% | 3% |
| Units sold | 20,000 | 80,000 | 210,000 |
| Price per show | $29.99 | $29.99 | $29.99 |
| **Annual home revenue** | **$600K** | **$2.4M** | **$6.3M** |

**Why conservative?** For reference, Half-Life: Alyx sold ~2M units on a 2M+ SteamVR installed base, but it was a first-party Valve title with 15 years of franchise equity. A third-party music experience will convert at a lower rate until brand awareness builds. We model conservatively and treat upside as optionality.

### Channel 4: Corporate & Premium Events (vZero Retail)

During off-peak hours (weekday mornings, corporate events), the performance rooms are available for high-margin rentals.

| Use Case | Price | Frequency | Annual Revenue |
|----------|-------|-----------|---------------|
| Corporate keynote / product launch | $25,000–$50,000/day | 2x/month | $600K–$1.2M |
| VIP private sessions (4-person, $250/ticket) | $1,000/session | 4x/week | $200K |
| Wellness/meditation (off-peak, $40/ticket) | $480/session | 5x/week | $125K |
| **Total ancillary** | | | **$925K–$1.5M** |

---

## III. The Content Strategy: The Pyramid

vZero's content operates on three tiers, each serving a distinct strategic function. Together, they create both the tentpole events that drive sign-ups and the persistent content flow that prevents churn.

### Tier 1: AAA Spectacles (2 per year, $10M each)

Full interactive volumetric experiences with game mechanics, WFS spatial audio, and 20+ minutes of runtime. These are the flagship productions — the reason someone subscribes.

| Category | Cost | Detail |
|----------|------|--------|
| **Artist advance** | $4M–$6M | Recoupable against royalties. For a Tier-1 artist, this is below a standard Vegas residency advance ($10M+) because vZero offers global distribution — the artist earns on every venue, every licensed site, and every home sale. |
| **Volumetric capture** | $1.5M–$2M | Studio lease (4DViews or Metastage), camera crew, lighting design, 3–4 months of capture sessions |
| **4DGS processing** | $500K–$800K | Compute costs for multi-month processing pipeline, quality iteration, engineering time |
| **Interactive layer** | $800K–$1.2M | Game design, Unreal/Unity development, playtesting |
| **Spatial audio mix** | $400K–$600K | WFS authoring, ambisonic decomposition, haptic score |
| **Legal & licensing** | $300K–$500K | Grand Rights clearance, mechanical/sync licensing, contract negotiation |
| **Contingency (10%)** | $1M | First show will exceed budget. This is expected. |
| **Total** | **$9M–$12M** | Rounded to $10M for modeling |

### Tier 2: Weekly Sessions (~50 per year, $50K–$100K each)

Intimate 10-minute performances captured at the permanent vZero studio at Miraval. No interactive layer, no game mechanics — pure Neural Presence. An artist performs, and the subscriber stands in the room with them.

**The Studio Residency model:** vZero establishes a permanent 4DGS capture rig at a prestige location — a place with inherent artistic gravity, world-class acoustics, and visual beauty. Artists come to the studio, not the other way around. The setting itself is part of the content.

| Item | Cost | Notes |
|------|------|-------|
| **Studio build-out (one-time)** | $2M–$3M | Permanent 64-camera rig, lighting, ambisonic array, facility lease |
| **Per-session production** | $20K–$40K | Small crew (5–8 people), single-day capture, streamlined processing |
| **Artist fee per session** | $10K–$50K | Wide range: emerging artists for exposure, mid-tier for fee, occasional A-list as prestige drops |
| **Annual operating cost** | $2.5M–$5M | 50 sessions/year at $50K–$100K average all-in |

**Why artists participate:** The Weekly Sessions become a cultural brand — the volumetric equivalent of "MTV Unplugged" or "Tiny Desk Concerts." Mid-tier and emerging artists gain exposure to the entire Neural Pass subscriber base. A-list artists participate occasionally as prestige events (analogous to a Zane Lowe interview or a COLORS session).

### Tier 3: Sports (Weekly+, from Year 3)

Volumetric capture of live sporting events, delivering the experience of standing courtside, pitch-side, or trackside.

**Why sports is the mass-market unlock:** Music alone can drive 1–2 million subscribers. Sports transforms the addressable market entirely. There are 3.5 billion football fans globally, 2 billion+ NBA fans, 1 billion tennis fans. Weekly sports content provides the cadence that keeps subscribers engaged between AAA music drops.

**Entry point: individual sports.** Volumetric capture of a full football pitch (22 players, 100m field) requires camera arrays at a scale beyond Year 3 capability. Individual sports — tennis, boxing, MMA, gymnastics — require capturing 1–2 athletes in a contained volume. This is achievable with portable 4DGS rigs by Year 3.

| Sport | Capture Complexity | Audience | Entry Feasibility |
|-------|--------------------|----------|-------------------|
| **Tennis** | Low (1–2 players, contained court) | 1B+ global fans, premium demographics | Year 3 |
| **Boxing / MMA** | Low (2 athletes, small ring) | 500M+ fans, high engagement per event | Year 3 |
| **Basketball (NBA)** | Medium (10 players, half-court focus) | 2B+ fans, massive US market | Year 4 |
| **Football (Premier League)** | High (22 players, large pitch) | 3.5B fans, largest global sport | Year 5+ |

**Sports content model:** vZero negotiates "Volumetric Digital Rights" — a new rights category distinct from broadcast, streaming, or in-venue. One match per week in partnership with a league or federation. The subscriber stands courtside or pitch-side and watches the match unfold around them in 6DOF.

**Sports rights cost:** Highly variable. A pilot partnership with a tennis federation or boxing promotion for a single event per week could start at $2M–$5M/year. Premium league deals (NBA, Premier League) at scale would be $10M–$30M/year — significant, but a fraction of traditional broadcast rights because volumetric is an additive channel, not a replacement.

### Content Library Growth

| Year | AAA Spectacles | Weekly Sessions | Sports | Total Library |
|------|---------------|----------------|--------|--------------|
| Y1 | 2 | ~25 (half-year ramp) | — | 27 experiences |
| Y2 | 4 cumulative | ~75 cumulative | — | ~79 experiences |
| Y3 | 6 | ~125 | Tennis/boxing pilot (~40 matches) | ~171 experiences |
| Y4 | 8 | ~175 | + NBA (~80 matches) | ~263 experiences |
| Y5 | 10 | ~225 | + additional leagues (~120 matches) | ~355 experiences |
| Y6 | 12 | ~275 | + football (~160 matches) | ~447 experiences |

By Year 6, the subscriber has access to 12 AAA interactive spectacles, 275 intimate studio sessions, and hundreds of live sports captures. This is a library that justifies $50/month — not because any single piece of content is worth it, but because the breadth and freshness make cancellation feel like a loss.

### Annual Content Investment

| Category | Y1 | Y2 | Y3 | Y4 | Y5 | Y6 |
|----------|-----|-----|-----|-----|-----|-----|
| AAA Spectacles (2/year) | $20M | $20M | $20M | $20M | $20M | $20M |
| Weekly Sessions | $1.5M | $2.5M | $3M | $4M | $4.5M | $5M |
| Permanent studio (one-time + ops) | $3M | $1M | $1M | $1M | $1M | $1M |
| Sports rights + capture | — | — | $5M | $10M | $18M | $25M |
| **Total content spend** | **$24.5M** | **$23.5M** | **$29M** | **$35M** | **$43.5M** | **$51M** |

**Recoupment:** Content costs are fixed regardless of subscriber count. At 4 million subscribers paying $720/year, the content investment ($51M in Year 6) represents just 1.8% of subscription revenue. Content is the highest-leverage investment in the business.

---

## IV. Artist Deal Structure

The artist contract is a **"Volumetric Performance Rights"** agreement — a new asset class in the music industry, analogous to "Grand Rights" for theatrical performance.

| Term | Detail |
|------|--------|
| **Advance** | $4M–$6M, recoupable against the 30% royalty |
| **Royalty** | 30% of Publisher gross receipts from all channels |
| **Royalty split** | 50% to Label (master), 25% to Artist, 25% to Songwriter/Publishing |
| **Rights granted** | Exclusive LBE Volumetric Performance (3-year term), Non-exclusive Home VR (perpetual), Non-exclusive B2B licensing (perpetual) |
| **Grand Rights** | Theatrical/dramatic performance rights for the show (avoids per-play PRO licensing) |
| **Image/likeness** | Limited to 4DGS reproduction within vZero experiences; no derivative works, no AI training |
| **Term** | 3-year exclusive for LBE; perpetual non-exclusive for digital distribution |

The 30% royalty and recoupable advance structure mirrors standard music licensing for theatrical/immersive productions (cf. Cirque du Soleil artist deals, ABBA Voyage terms).

---

## V. 6-Year Consolidated Financial Projection

### Flagship Network Growth

The venue network scales to 12 Flagships over 6 years — 2 new venues per year after the initial pair. Each new venue benefits from shared content, proven construction playbooks, and declining per-site build costs.

| Year | New Flagships | Cumulative | Cities (illustrative) |
|------|--------------|------------|----------------------|
| Year 1 (2029) | 2 | 2 | Las Vegas, London |
| Year 2 (2030) | 2 | 4 | New York, Tokyo |
| Year 3 (2031) | 2 | 6 | Dubai, Paris |
| Year 4 (2032) | 2 | 8 | Seoul, Singapore |
| Year 5 (2033) | 2 | 10 | Los Angeles, Berlin |
| Year 6 (2034) | 2 | 12 | Macau, Sydney |

### Revenue Projection

| Stream | Y1 | Y2 | Y3 | Y4 | Y5 | Y6 |
|--------|-----|-----|-----|-----|-----|-----|
| Flagship tickets | $11.5M | $32.0M | $57.0M | $78.0M | $97.0M | $115.0M |
| B2B Licensing | $0 | $1.4M | $4.4M | $8.0M | $12.0M | $16.0M |
| Home B2C (individual sales) | $0.6M | $2.4M | $3.0M | $2.0M | $1.0M | $0.5M |
| Neural Pass (subscription) | — | $31M | $135M | $403M | $1,001M | $1,872M |
| Corporate/Events | $0.9M | $2.4M | $4.2M | $6.0M | $7.5M | $9.0M |
| **Total Gross Revenue** | **$13.0M** | **$69.2M** | **$203.6M** | **$497.0M** | **$1,118.5M** | **$2,012.5M** |

**Flagship assumptions:** Per-venue revenue at base occupancy (80%) is ~$11.2M/year. Year 1 reflects one full-year venue and one partial. New venues in subsequent years contribute partial-year revenue in their opening year and full revenue thereafter. Occupancy ramps from 70% in opening months to 80%+ by month 6 as word-of-mouth builds.

**Home B2C individual sales decline** as the Neural Pass subscription absorbs the addressable market. Individual purchases remain available for consumers who prefer one-time ownership, but the subscription is the dominant home channel from Year 3 onward.

**Neural Pass assumptions** are detailed in Section VI. The subscription is the primary scale engine of the business. Revenue includes base subscriptions ($49.99/mo hardware bundle + $14.99/mo digital-only), add-on packs (sports, gaming, wellness at $9.99/mo), and the Digital-Only tier for existing Steam Frame owners.

### Cost Structure

| Category | Y1 | Y2 | Y3 | Y4 | Y5 | Y6 |
|----------|-----|-----|-----|-----|-----|-----|
| Content (AAA + Sessions + Sports) | $24.5M | $23.5M | $29M | $35M | $43.5M | $51M |
| Venue OpEx (all sites) | $6M | $14M | $22M | $30M | $38M | $44M |
| Engineering & platform | $4M | $6M | $8M | $12M | $16M | $20M |
| Neural Pass OpEx (cloud, support, licensing) | — | $6M | $22M | $72M | $168M | $320M |
| New venue build-out | $0* | $26M | $26M | $26M | $26M | $26M |
| Corporate overhead | $1.5M | $3M | $5M | $8M | $12M | $16M |
| **Total Costs** | **$36M** | **$78.5M** | **$112M** | **$183M** | **$303.5M** | **$477M** |

*Year 1 venue build-out is pre-funded by the Series A (deployed during the 30-month pre-launch phase).*

**Neural Pass OpEx** includes cloud streaming infrastructure, content delivery, customer support, licensing royalties, and platform operations. At scale, this averages ~$8/month per active subscriber in direct operating costs — declining over time as infrastructure amortizes.

### EBITDA (Excluding SPV Debt Service)

| | Y1 | Y2 | Y3 | Y4 | Y5 | Y6 |
|--|-----|-----|-----|-----|-----|-----|
| **EBITDA** | **($23.0M)** | **($9.3M)** | **$91.6M** | **$314.0M** | **$815.0M** | **$1,535.5M** |
| **EBITDA Margin** | (177%) | (13%) | 45% | 63% | 73% | 76% |

**Year 1 is a planned loss.** The first full year of operation carries content production costs while only two Flagships are operational and the subscription has not yet launched. Year 2 approaches break-even as the Neural Pass pilot generates initial revenue. From Year 3 onward, subscription economics drive rapid margin expansion — content and platform costs are largely fixed while subscription revenue scales with subscriber count.

**Note on methodology:** Flagship and B2B projections use base-case assumptions (80% occupancy, gradual licensing adoption). The Neural Pass projections assume the scaling trajectory detailed in Section VI, driven by the introduction of sports content in Year 3 and mass-market adoption from Year 4. Even if subscription growth is 50% slower than modeled, the business is profitable from Year 3 on venue and early subscription revenue alone.

---

## VI. The Neural Pass: The Scale Engine

The Neural Pass is a Hardware-as-a-Service (HaaS) subscription that bundles a Steam Frame headset, haptic vest, and unlimited access to the vZero content library for $49.99/month on a 24-month contract. It is the primary long-term value driver of the business.

The Flagships prove the technology and build the brand. The Neural Pass scales it to millions.

### Unit Economics

| Item | Amount | Notes |
|------|--------|-------|
| Steam Frame (wholesale) | ($400) | ~$500–$600 retail, 40–50% wholesale discount at volume |
| Haptic vest (wholesale) | ($175) | Woojer Vest 4 ~$350 retail, 50% wholesale at volume |
| Shipping & fulfillment | ($75) | |
| **Hardware COGS** | **($650)** | |
| Customer acquisition cost | ($150) | |
| **Total upfront cost per subscriber** | **($800)** | |
| Monthly subscription revenue | $49.99 | |
| Monthly add-on revenue (avg.) | $10.00 | |
| Monthly operating cost (cloud, licensing, support) | ($8.00) | |
| **Monthly net margin (before debt service)** | **$51.99** | |
| 24-month gross revenue | **$1,439.76** | |
| 24-month LTV (net of operating costs) | **$1,247.76** | |
| **Hardware breakeven** | **Month 15** | Was Month 24 at $1,100 COGS |

**The critical inflection:** At Month 16, the hardware is fully amortized. The subscriber's $60/month becomes ~$52/month net margin (86%). Every subscriber who remains past Month 15 generates near-pure profit — and the 24-month contract means 9 months of high-margin revenue before they can churn. Retention is driven by constant content freshness — weekly Sessions, sports matches, and 2 AAA spectacles per year mean there is always something new. Canceling means losing access to a library that grows every week.

### Subscriber Growth Trajectory

| Year | New Subs | Cumulative Active (net of churn) | Annual Revenue | Key Growth Driver |
|------|----------|--------------------------------|---------------|-------------------|
| Y1 | — | — | — | Flagship launch. Subscription not yet active. |
| Y2 | 75,000 | 75,000 | $31M | Pilot launch. Flagship visitors convert. Library: 4 AAA + ~75 Sessions. |
| Y3 | 225,000 | 275,000 | $135M | SPV active. Sports launches (tennis/boxing). Library: 6 AAA + 125 Sessions + weekly sport. |
| Y4 | 725,000 | 900,000 | $403M | Sports as core content. 8 Flagships as conversion funnels. Mass marketing begins. |
| Y5 | 1,500,000 | 2,100,000 | $1,001M | Mass market. Hardware cost declining. Multiple sports leagues. 10 AAA + 225 Sessions. |
| Y6 | 2,500,000 | 4,000,000 | $1,872M | Platform scale. 12 Flagships. 12 AAA + 275 Sessions + daily sports. |

**Churn assumption:** 20% annual churn. This is deliberately higher than comparable services (Peloton ~30% annual at peak, Xbox Game Pass ~15%) to account for mass-market scaling introducing less committed subscribers. Hardware lock-in (24-month contract, device return on cancellation) provides structural churn resistance, but we model conservatively. Cumulative active figures are net of churn.

**Revenue composition per subscriber:** The subscriber mix evolves over time:
- Hardware bundle subscribers ($49.99/mo + add-ons) represent the majority in early years, declining as the Steam Frame installed base grows
- Digital-Only subscribers ($14.99/mo + add-ons) grow from ~30% in Year 2 to ~40% by Year 6 as existing Steam Frame owners convert
- Add-on packs ($9.99/mo sports, gaming, wellness) attach to ~30% of all subscribers
- Blended ARPU: ~$34–$41/month depending on year and subscriber mix, settling at ~$39/month by Year 6

**Why 4 million is realistic:**
- **The music market alone is vast.** Taylor Swift's Eras Tour sold 10.1M tickets. Beyonce's Renaissance Tour sold 2.8M. These fans paid $150–$1,500 for a single evening. A $50/month subscription for unlimited access to lifelike volumetric performances — with hardware included — is a fundamentally different value proposition.
- **Sports multiplies the TAM.** There are 1 billion tennis fans globally, 2 billion+ NBA fans, 3.5 billion football fans. Even fractional conversion rates produce millions of subscribers.
- **The hardware subsidy removes the barrier.** The Steam Frame retails at ~$500–$600, plus a $350 haptic vest — nearly $1,000 in hardware. The Neural Pass eliminates this cost entirely, converting a $1,000 purchase decision into a $50/month commitment — comparable to a gym membership or a premium streaming bundle.
- **Content freshness drives retention.** Weekly Sessions + weekly sports + 2 AAA drops per year means subscribers never run out of reasons to keep the headset on.

### Subscription Tiers

| Tier | Price | Hardware | Content |
|------|-------|---------|---------|
| **Founders Pass** | $49.99/mo (24-month contract) | Steam Frame + Haptic Vest included | Full vZero library + 2 monthly guest passes |
| **Digital-Only** | $14.99/mo (no contract) | None (for existing Steam Frame owners) | Full vZero library |
| **Expansion Packs** | +$9.99/mo each | — | Sports Pack, Gaming Pack, Wellness Pack |

The Digital-Only tier is strategically critical. It captures existing Steam Frame owners at pure software margin — no hardware subsidy, no SPV financing, near-100% margin. As the Steam Frame installed base grows (projected 7–10M+ by Year 6), this tier scales with zero capital intensity. At 1M Digital-Only subscribers, this tier alone generates ~$180M/year at ~95% margin.

### SPV Financing

The upfront hardware cost for Founders Pass subscribers is financed through a Special Purpose Vehicle (SPV):

1. **vZero Finance Co** (bankruptcy-remote subsidiary) holds subscriber contracts and leased hardware
2. A banking partner provides a warehouse credit facility to the SPV
3. Subscriber payments flow through the SPV, servicing the debt first
4. Remaining margin flows to vZero Holdings

**Financing timeline:**

| Phase | Period | Mechanism | Scale |
|-------|--------|-----------|-------|
| **Pilot** | Year 2 | Equity-funded ($10M–$15M from Series B) | 75,000 units |
| **Warehouse facility** | Year 3 | $100M–$300M credit line from banking partner | 275,000 cumulative units |
| **Expansion** | Year 4 | $800M–$1B facility, backed by 12+ months of payment history | 900,000 cumulative units |
| **Securitization** | Year 5+ | Asset-backed securities ("Neural Bond") sold to institutional investors | 2M+ cumulative units |
| **Mature program** | Year 6 | Multi-billion dollar ABS program | 4M cumulative units |

**Total hardware financed over 6 years:** ~$2.6B (4M units at declining COGS — hardware costs reduce from $650 in Year 2 to ~$500 in Year 6 as the Steam Frame matures and production scales).

This is large but not unprecedented. Peloton financed $1.5B+ in hardware through similar structures. Apple's iPhone Upgrade Program finances tens of billions annually. The founding partner's financial backing significantly de-risks the credit facility — his involvement provides the guarantee quality that accelerates the warehouse timeline and enables larger initial facilities than a standalone startup could secure.

### Manufacturing Partnership

At 4 million Neural Pass units, vZero is ordering more Steam Frames than any other single customer. This requires a formal manufacturing agreement with the founding partner's company — not just "priority allocation" but a committed production commitment.

The value exchange is clear: vZero provides a guaranteed hardware distribution channel that drives millions of units into homes that would never have purchased a VR headset for gaming alone. The founding partner's company gains a massive installed base for their platform ecosystem. This is the Xbox-Halo dynamic: the content sells the hardware.

### Why the Subscription Dominates

By Year 4, the Neural Pass generates 6x more revenue than all 8 Flagships combined. By Year 6, it generates 19x the venue revenue. This is by design:

- **Flagships are the funnel.** Every visitor is a potential subscriber. At 12 venues serving 1M+ visitors/year, even a 5% conversion rate produces 50,000+ new subscribers annually — and that's before marketing, word of mouth, and sports-driven acquisition.
- **Content freshness is the retention engine.** Weekly Sessions, weekly sports, and biannual AAA drops mean there is never a reason to cancel. The library grows every single week.
- **Hardware creates switching costs.** A subscriber using the vZero-subsidized Steam Frame and haptic vest is fully integrated into the ecosystem. The cost of leaving (returning hardware or paying the buyout) exceeds the cost of staying.
- **Sports is the mass-market accelerant.** Music drives the early adopters. Sports drives the mass market. The transition from music-only to music+sports in Year 3 is the inflection point that takes the subscriber base from hundreds of thousands to millions.
- **The margin improves with scale.** Content costs are fixed (the same shows serve 75,000 or 4,000,000 subscribers). Per-subscriber cloud costs decline as infrastructure amortizes. At 4 million subscribers, the Neural Pass is a software-margin business generating $1.87B at 76% EBITDA margin.

---

## VII. Capital Requirements & Structure

### Funding Roadmap

The capital strategy matches the business phases: equity for content and venues, structured debt for the subscription hardware rollout.

| Round | Timing | Amount | Primary Uses |
|-------|--------|--------|-------------|
| **Series A** | Pre-launch | $40M | Flagships 1–2, Show 1, engineering team, 30-month runway to launch |
| **Series B** | Year 1 | $80M–$100M | Shows 2–4, Flagships 3–4, Neural Pass pilot (75K units), permanent capture studio, sports rights scouting |
| **Series C** | Year 3 | $120M–$180M | Flagships 5–8, continued content production, sports rights acquisition, Neural Pass scale-out, international expansion |
| **SPV Facility** | Year 3+ | $300M–$5B+ | Neural Pass hardware financing (off balance sheet, scaling with subscriber count) |

**Total equity raised through Year 3:** $240M–$320M
**Total debt (SPV, off balance sheet):** $300M–$5B+ (scales with subscriber count over Years 3–6)

### Series A: $40M (Detail)

| Allocation | Amount | Entity |
|-----------|--------|--------|
| Flagship 1 build-out | $15M | Retail |
| Flagship 2 build-out | $11M | Retail |
| Show 1 production | $10M | Publisher |
| Engineering team (18 months) | $3M | Publisher |
| Operating reserve | $1M | Holdings |
| **Total** | **$40M** | |

### Founding Partner Role

The founding partner and his structures provide:

1. **Anchor equity** across funding rounds (lead or co-lead position in Series A, with pro-rata rights in subsequent rounds)
2. **Strategic hardware access** — priority allocation of Steam Frame units, dev kit access, engineering collaboration with the hardware team
3. **Platform distribution** — vZero content featured on the Steam Store and SteamVR home environment
4. **Neural Pass acceleration** — the founding partner's financial credibility de-risks the SPV facility, enabling faster and larger credit lines than a standalone startup could secure
5. **Credibility signal** — the founding partner's involvement de-risks the venture for co-investors, artist partners, and venue landlords at every stage

In exchange, the founding partner and affiliated structures receive preferred equity with standard protective provisions, a board seat, and first-refusal rights on future rounds.

### Content Investment Over 6 Years

At 2 shows per year and $10M per show, the Publisher invests $120M in content over 6 years. This builds a library of 12 AAA volumetric spectacles — the definitive catalog in a new medium. This library is the core asset that drives flagship attendance, B2B licensing value, and Neural Pass subscriber retention. It is also the primary basis for the Publisher's valuation at exit.

---

## VIII. Exit & Valuation Logic

### Minimum Horizon: 6 Years

vZero is not designed for a quick flip. The business requires 6+ years to realize its full value because:

1. **The content library needs depth.** 12 AAA shows, 275+ Sessions, and years of sports archives create a catalog with lasting value. 2–4 shows is a startup; this library is an institution.
2. **The venue network needs scale.** 12 Flagships across global entertainment capitals creates a defensible physical moat that takes years to build and is impossible to replicate quickly.
3. **The subscription base needs critical mass.** 4 million subscribers generating $1.87B in annual recurring revenue is a fundamentally different asset than a 75,000-subscriber pilot. The flywheel — content drives subscribers, subscribers fund content — needs time to reach full velocity.

Exiting before Year 6 would mean selling the business before the sports-driven mass-market phase delivers its full subscriber cohort — leaving the majority of the value on the table.

### Year 6 Valuation

| Method | Basis | Multiple | Valuation |
|--------|-------|----------|-----------|
| Publisher (subscription ARR) | $1,872M Neural Pass + $16M B2B licensing | 6–8x | $11.3B–$15.1B |
| Retail (venues EBITDA) | ~$64M EBITDA from 12 Flagships | 8–10x | $512M–$640M |
| **Blended enterprise value** | | | **$11.8B–$15.7B** |

The Publisher dominates the valuation because it owns the recurring revenue, the IP library, and the sports rights portfolio. The Retail business is valuable but secondary — it is the physical infrastructure that feeds the subscription funnel.

**Note on multiples:** At $1.87B ARR, we apply 6–8x rather than the 10–12x typical of smaller SaaS companies, reflecting the hardware-subsidized nature of a portion of the revenue and the maturity of the business at Year 6. Pure Digital-Only subscribers (~1M at $14.99/mo) command higher multiples; the blended figure reflects the full subscriber mix.

**Sensitivity:** If Neural Pass reaches 2 million subscribers (roughly the Year 5 target, one year behind plan), the Publisher ARR is ~$1.0B and the blended valuation is $6.2B–$8.3B. If sports content fails to achieve mass-market traction and the subscriber base plateaus at 1 million (music-only), the Publisher ARR is ~$468M and the blended valuation is $3.3B–$4.3B. Even in this downside scenario, the venue network with 12 Flagships, a 12-show library, and B2B licensing supports a $1B+ standalone valuation.

### Strategic Acquirers

1. **The founding partner's platform company** — acquiring vZero secures the "anchor tenant" of their VR ecosystem. At Year 6, vZero is driving 4 million+ headset deployments through the Neural Pass — potentially the single largest distribution channel for the platform's hardware. The acquisition thesis is identical to a console manufacturer acquiring a first-party studio — but with a $1.87B ARR subscription business attached that sells millions of units of the acquirer's own hardware. This is the most natural exit path and the one most aligned with both parties' interests.

2. **Universal Music Group / Warner Music** — acquiring vZero provides the major labels with a proprietary distribution channel for volumetric performance rights, a 4-million-subscriber direct-to-consumer base, and a new revenue line that bypasses streaming economics entirely.

3. **Live Nation / AEG** — vZero's venue network and content library extend their live entertainment portfolios into the volumetric domain, with the Neural Pass providing the recurring revenue that traditional venue operators lack.

4. **IPO** — at $2.0B revenue and 76% EBITDA margins, vZero is a compelling public market candidate. The dual-entity structure allows for a Publisher-only IPO (pure SaaS/media multiples) while retaining the Retail business as a private cash-flow asset. The combination of recurring subscription revenue, a growing content library, and sports rights creates a public market narrative that spans technology, media, and entertainment.

---

*This document is part of the vZero Strategic Document Suite. See also: Strategic Vision & Market Opportunity, Technical Architecture, and Execution Roadmap.*
