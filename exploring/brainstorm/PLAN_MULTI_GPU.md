# Plan: Multi-GPU Data-Parallel 4D Gaussian Splatting Training

## Context

Our single-GPU 4DGS trainer (`train_4d.py`) works well but doesn't scale to production data. The current Neu3D coffee_martini run (300 frames, 17 cameras, H100) takes ~1.7 hours. At production scale (5 min video = 9,000 frames, 120 cameras, full resolution), a single GPU can only fit ~2 frames in memory and training would take days.

We need a multi-GPU data-parallel training system that:
- Scales to 4-16 GPUs (target: RTX 4090 at $0.44/hr each)
- Handles 9,000+ frames with efficient disk I/O
- Keeps all GPU replicas in perfect sync through densification

## Architecture

**Manual data-parallel** (not DDP — our model uses raw tensors, not nn.Module):
- Every GPU holds a full model replica (identical parameters)
- Each GPU renders a different (timestep, camera) pair per step
- All-reduce gradients after backward → identical optimizer step on all ranks
- Effective batch size = num_GPUs

**Key insight**: Densification is deterministic given identical gradient statistics. All-reduce the grad accumulators before densification → all GPUs make identical clone/split/prune decisions → models stay in sync without broadcasting parameters.

## New Files

### 1. `src/dist_utils.py` (~100 lines)

Distributed communication primitives:

- `setup_distributed()` → (rank, world_size, local_rank) from torchrun env vars, NCCL backend
- `all_reduce_gradients(model, world_size)` — fused: flatten all 9 grad tensors into one buffer, single NCCL all-reduce, unflatten back. ~256 MB at 1M Gaussians, ~8ms over PCIe Gen4
- `all_reduce_accumulators(grad_accum, grad_count, grad_accum_t, grad_count_t)` — SUM reduction (no division). `avg = sum_accum / sum_count` is mathematically correct
- `sync_seed(step)` — deterministic seed for `torch.manual_seed` before densification so `torch.randn_like` in spatial split produces identical results on all ranks

### 2. `src/frame_cache_distributed.py` (~200 lines)

Double-buffered distributed frame cache:

```
DistributedFrameCache
├── front_buffer: dict[frame_idx → Tensor(C,H,W,3)] on GPU  (active batch)
├── back_buffer: dict[frame_idx → Tensor(C,H,W,3)] on CPU pinned  (prefetch)
└── executor: ThreadPoolExecutor  (background disk I/O)
```

**Partition strategy**: Each rank deterministically generates the same random permutation (seeded by swap epoch), takes `perm[rank::world_size]` — non-overlapping, no broadcast needed.

**Double-buffer flow**:
1. While training on front_buffer, background thread loads next batch into CPU pinned memory (back_buffer)
2. On swap: move back_buffer → GPU (front_buffer), free old front_buffer, start next prefetch
3. If prefetch isn't ready when swap is due, block and wait

**Resolution-aware loading**: Loads images at the current curriculum resolution (downsample at read time in the prefetch thread to save GPU memory).

### 3. `src/train_4d_distributed.py` (~350 lines)

Imports from `train_4d.py`: GaussianModel4D, render_4d, make_optimizer_4d, densify_and_prune_4d, render_velocity_map, flow_gradient_loss, save_ply_4d, load_scene_4d. Imports ssim_loss from `train.py`.

**Training loop changes vs single-GPU**:

```python
for step in range(num_steps):
    # Each rank: sample from LOCAL frame partition
    frame_idx = cache.sample()
    cam_idx = random camera

    # Forward + backward (independent per GPU)
    loss.backward()

    # Accumulate grad stats from LOCAL gradients (pre-all-reduce)
    grad_accum[mask] += means.grad.norm(...)
    grad_count[mask] += 1

    # ALL-REDUCE gradients (the sync point)
    all_reduce_gradients(model, world_size)

    # Optimizer step (identical on all ranks)
    optimizer.step()

    # DENSIFICATION (every 100 steps):
    if should_densify:
        all_reduce_accumulators(...)       # sync grad stats
        torch.manual_seed(sync_seed(step)) # deterministic RNG
        densify_and_prune_4d(...)          # identical on all ranks
        optimizer = make_optimizer_4d(...) # identical rebuild
```

**Resolution curriculum** (3 phases):
- Phase 1 (steps 0–10K): 1/4 resolution → ~38 frames/GPU on 4090 → good coverage
- Phase 2 (steps 10K–20K): 1/2 resolution → ~8 frames/GPU
- Phase 3 (steps 20K+): full resolution → ~2 frames/GPU (rely on double-buffer)

**LR scaling**: `lr *= sqrt(world_size)` with 1000-step warmup.

**Rank 0 only**: logging, checkpointing, evaluation.

**Launch**: `torchrun --nproc_per_node=4 src/train_4d_distributed.py --scene_dir ... --num_steps 30000`

## No Changes to Existing Files

- `train_4d.py` — all classes/functions imported, not modified
- `train.py`, `data.py`, `data_pku.py`, `viewer_4d.py` — unchanged

## Memory Budget (per RTX 4090, 24 GB)

| Component | Size |
|-----------|------|
| Model (1M Gaussians, 9 tensors) | ~250 MB |
| Adam state (2 moments) | ~500 MB |
| Rasterization buffers (full res) | ~3 GB |
| Gradients + accumulators | ~260 MB |
| **Available for frames** | **~19 GB** |

Frames per GPU by resolution (120 cameras, 2704×2028):
- Full res: ~2 frames (7.9 GB each)
- Half res: ~8 frames (2.0 GB each)
- Quarter res: ~38 frames (0.5 GB each)

## Communication Cost

| Operation | Data | Frequency | Time (PCIe Gen4) |
|-----------|------|-----------|-------------------|
| Gradient all-reduce | ~256 MB | every step | ~8 ms |
| Accumulator all-reduce | ~16 MB | every 100 steps | ~0.5 ms |
| Frame swap (disk→CPU→GPU) | 2 frames = ~16 GB | every 500 steps | masked by double-buffer |

Overhead: ~8ms per 200ms step = ~4%. Negligible.

## Verification

1. **Sync check**: In debug mode, broadcast rank 0 params every 1000 steps, assert max diff < 1e-5 against other ranks
2. **Quality check**: Run on Neu3D coffee_martini with 4 GPUs, compare final PSNR against single-GPU baseline (should match within ~0.5 dB)
3. **Scaling check**: Measure it/s at 1, 2, 4, 8 GPUs. Expect near-linear scaling of images/sec
4. **Memory check**: Monitor `nvidia-smi` during training, verify no OOM at any phase

## Implementation Order

1. `dist_utils.py` — distributed primitives
2. `frame_cache_distributed.py` — double-buffered cache with partitioning
3. `train_4d_distributed.py` — main training loop
4. Test with 2 GPUs on existing Neu3D data
5. Resolution curriculum
6. Scale testing (4-8 GPUs)
