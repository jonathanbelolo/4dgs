# 4D Gaussian Splatting: Multi-GPU Scaling Analysis

## Current System

Single-GPU trainer (`train_4d.py`) using batch-swap frame cache. Validated on Neu3D coffee_martini (300 frames, 17 cameras, 2704x2028) on H100 SXM: 40K steps, ~6.7 it/s, PSNR ~30 dB.

## Production Target

- 5 minutes of video (9,000 frames) or 30 minutes (54,000 frames)
- 120 cameras at full resolution (2704x2028)
- Multi-GPU data-parallel training with resolution curriculum

---

## GPU Specifications

| GPU | VRAM | FP32 TF | Mem BW | Frames/GPU (full res, 120 cams) | $/hr (cloud) |
|-----|------|---------|--------|--------------------------------|-------------|
| RTX 4090 | 24 GB | 83 | 1.0 TB/s | 2 | $0.34 |
| RTX 5090 | 32 GB | ~105 | 1.8 TB/s | 3 | $0.69 |
| RTX PRO 6000 | 96 GB | 125 | 1.8 TB/s | 11 | $1.69 |
| H100 SXM | 80 GB | 67 | 3.4 TB/s | 9 | $1.99 |
| H200 SXM | 141 GB | 67 | 4.8 TB/s | 17 | $3.59 |
| B200 | 180 GB | ~70 | 8.0 TB/s | 22 | $5.98 |
| B300 | 288 GB | ~80 | 8.0 TB/s | 35 | ~$8 (est) |

Per frame at full resolution: 120 cameras x 2704 x 2028 x 3 x float32 = **7.9 GB**

---

## Interconnect Requirements

With ~256 MB of gradients at 1M Gaussians (~1.25 GB at 5M):

| Interconnect | Bandwidth | All-reduce (1M G) | All-reduce (5M G) | Viable? |
|---|---|---|---|---|
| 10 Gbps Ethernet | 1.25 GB/s | 400ms | 2000ms | No |
| 25 Gbps Ethernet | 3.1 GB/s | 160ms | 800ms | No |
| 100 Gbps RoCE | 12.5 GB/s | 40ms | 200ms | Marginal |
| 400 Gbps InfiniBand | 50 GB/s | 10ms | 25ms | Yes |
| NVLink (intra-node) | 900 GB/s | 0.3ms | 1.4ms | Ideal |

**Minimum: 400 Gbps InfiniBand for multi-node.** Consumer GPUs (4090, 5090) lack NVLink and are not offered in InfiniBand clusters — they are limited to single-node (max 4-8 GPUs).

## Cloud Providers with InfiniBand Multi-Node Clusters

| Provider | GPUs Available | Interconnect | H100 $/hr |
|----------|---------------|-------------|-----------|
| RunPod | H100, H200, B200 | 1600-3200 Gbps IB/RoCE | ~$2.00 |
| Lambda | H100, B200 | 3200 Gbps Quantum-2 IB | $2.49 |
| CoreWeave | H100, H200, B200, GB200 | InfiniBand | ~$6.15 |
| FluidStack | H100, H200, B200 | InfiniBand | Contact sales |
| Vast.ai | H100 | InfiniBand (clusters) | ~$1.74 |
| Together.ai | H100, H200, B200, GB200 | InfiniBand | Contact sales |

RTX 4090 multi-node clusters with InfiniBand are **not available** from any major provider. The 4090 is best for single-node prototyping only.

---

## 5-Minute Video (9,000 frames, 120 cameras)

**Assumptions:**
- ~300K optimizer steps (single-GPU equivalent)
- ~2-3M Gaussians at convergence
- Resolution curriculum: 1/4 res (40% of steps) -> 1/2 res (30%) -> full res (30%)
- Linear step reduction with GPU count (effective batch size = num_GPUs)
- Communication overhead: 5-15% depending on GPU count

### Training Time & Cost

| GPU | 4 GPUs | | 8 GPUs | | 16 GPUs | | 32 GPUs | |
|-----|--------|-------|--------|-------|---------|-------|---------|-------|
| | **Time** | **Cost** | **Time** | **Cost** | **Time** | **Cost** | **Time** | **Cost** |
| **4090** | 10 hrs | $14 | 5.3 hrs | $14 | — | — | — | — |
| **5090** | 8 hrs | $22 | 4.3 hrs | $24 | — | — | — | — |
| **H100** | 6.7 hrs | $53 | 3.5 hrs | $56 | 1.9 hrs | $61 | — | — |
| **H200** | 5.8 hrs | $83 | 3.1 hrs | $89 | 1.7 hrs | $98 | — | — |
| **B200** | 4.2 hrs | $100 | 2.2 hrs | $105 | 45 min | $72 | 22 min | $71 |
| **B300** | 3.7 hrs | ~$118 | 2.0 hrs | ~$128 | 40 min | ~$85 | 20 min | ~$85 |

"—" = not available as multi-node with InfiniBand, or not cost-effective.

4090/5090 limited to single-node (max 4-8 GPUs). H100+ scales to multi-node via InfiniBand.

### Key Takeaways (5-Minute Video)

- **Cheapest**: 4-8x RTX 4090, single node, ~$14 but 5-10 hours
- **Fastest**: 32x B200, 4 nodes with IB, **~22 minutes**, ~$71
- **Best value at scale**: 16x B200, 2 nodes, **~45 minutes**, ~$72
- Cost is roughly constant across GPU counts (ideal linear scaling) — you trade time for parallelism

---

## 30-Minute Video (54,000 frames, 120 cameras)

### Setup: 64x B200, 8 Nodes, InfiniBand

| Parameter | Value |
|-----------|-------|
| Total training pairs | 54,000 x 120 = 6.5M |
| Model at convergence | ~5M Gaussians |
| Base optimizer steps | ~750K (each pair seen ~2-3x) |
| Steps per GPU (64 GPUs) | ~11,700 |
| Gradient size (5M Gaussians) | ~1.25 GB |
| All-reduce over 400 Gbps IB | ~25ms (<5% of step time) |

### Frame Cache

- 180 GB VRAM per B200 -> 22 frames/GPU at full resolution
- 64 GPUs x 22 = **1,408 frames cached** across cluster (2.6% of 54,000)
- Swap every ~308 steps (~154s of training)
- Double-buffered prefetch: 25s load time fits within 154s window

### Time Breakdown with Resolution Curriculum

| Phase | Steps | Resolution | Gaussians | it/s per GPU | Wall time |
|-------|-------|-----------|-----------|-------------|-----------|
| Phase 1 (40%) | 4,688 | 1/4 res | ~1M | ~18 | 4.3 min |
| Phase 2 (30%) | 3,516 | 1/2 res | ~3M | ~4.7 | 12.5 min |
| Phase 3 (30%) | 3,516 | Full res | ~5M | ~1.9 | 30.8 min |
| **Total** | **11,719** | | | | **~50 min** |

### Summary

| | Value |
|---|---|
| Hardware | 64x B200, 8 nodes, 400 Gbps InfiniBand |
| Training time | **~50 minutes** |
| Cost | 64 x $6/hr x 0.83 hrs = **~$320** |
| Without curriculum | ~1.7 hours, ~$650 |

---

## Scaling Limits

| Factor | Ceiling | Why |
|--------|---------|-----|
| GPU count | ~64 | Beyond 64, batch size scaling saturates and all-reduce overhead grows |
| Inter-node bandwidth | 400 Gbps IB minimum | Anything less than 100 Gbps makes multi-node impractical |
| Frame cache hit rate | ~2-3% at 54K frames | Double-buffering masks I/O latency; NVMe SSD required |
| Gaussian count | ~5M | Memory for model + optimizer grows; rasterization slows linearly |
| Step reduction | ~linear up to 16 GPUs | Diminishing returns beyond 16-32 GPUs |

## Architecture Notes

- **Manual data-parallel** (not DDP) since GaussianModel4D uses raw tensors, not nn.Module
- **Deterministic densification**: all-reduce grad accumulators before densification; seed RNG identically on all ranks so clone/split/prune decisions are bit-identical
- **Resolution curriculum**: 2x overall speedup by training at lower resolution in early phases
- **Double-buffered frame cache**: background thread prefetches next batch from disk to CPU pinned memory while training on current GPU batch
- See `PLAN_MULTI_GPU.md` for implementation details
