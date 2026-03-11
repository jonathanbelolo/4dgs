# Post-Mortem: 3DGS Quality Experiments (v3–v7)

## Experiment Results Summary

| Version | Test PSNR | Train PSNR (late) | Gaussians | Changes from v5 baseline | Verdict |
|---------|-----------|-------------------|-----------|--------------------------|---------|
| v3 | 13.38 dB | ~30 dB | 564K | Random init (no SfM) | Floaters from bad geometry |
| v4 | 21.73 dB | ~25 dB | 157K | Sparse SfM init (3.4K pts) | +8 dB from SfM init |
| **v5** | **25.78 dB** | **~34 dB** | **379K** | **Dense SfM (287K), full res** | **Best so far** |
| v6 | 24.43 dB | ~34 dB | 374K | + COLMAP stereo depth (L1) | -1.35 dB — depth noise hurt |
| v7 | 15.35 dB | ~33 dB | 1,185K | + Mono depth, SH=2, no reset | -10.4 dB — catastrophic floaters |

## What Worked

1. **Dense SfM initialization (v4→v5: +4 dB)**: Going from 3.4K sparse to 287K dense points was the single biggest quality jump. Provides correct initial geometry that training refines rather than discovers.

2. **Full resolution (v4→v5)**: Training at 2704x2028 instead of 1352x1014 preserves detail that downsampled training smears away.

3. **Opacity reset (v5)**: The periodic opacity reset every 5K steps during densification is essential. v5 had 5 resets (at 5K, 10K, 15K, 20K, 25K). These prune accumulated floaters and force the model to re-justify every Gaussian's existence.

4. **SH degree 3 (v5)**: With 17 training views, SH degree 3 gives the model enough capacity to match training views well (34 dB). The overfitting concern is real but moderate — v5 still got 25.78 dB test.

## What Failed and Why

### COLMAP Stereo Depth (v6): -1.35 dB
- **What we did**: Added L1 depth loss using COLMAP `patch_match_stereo` depth maps, normalized by median depth.
- **Why it failed**: With only 17 cameras in a wide-baseline setup, stereo depth is noisy — only 49% of pixels had valid depth, and the valid pixels had errors. The depth maps at max ranged to 4855 (should be ~100). Forcing Gaussians to match incorrect depth moved them away from photometrically correct positions.
- **Lesson**: Stereo depth from sparse viewpoints is unreliable as supervision. It's useful for initialization (which worked great) but not as a loss signal.

### Monocular Depth + No Opacity Reset + SH=2 (v7): -10.4 dB
Multiple compounding failures:

**1. No opacity reset was catastrophic.**
The DepthRegGS paper recommends removing opacity reset for few-view scenarios, but their setup is very different (6K iterations, 3-8 views, strong depth regularization from the start). In our 100K iteration, 17-view setup:
- v5 had 5 opacity resets that pruned floaters during the 40K-step densification window
- v7 had ZERO resets while growing from 287K to 1.185M Gaussians
- Floaters accumulated unchecked for 40K steps, then persisted for 60K more steps of pure refinement
- The 0.001 opacity regularization was far too weak to substitute for periodic hard resets
- Result: the test render shows massive streak/floater artifacts everywhere

**2. Pearson depth correlation didn't constrain geometry enough.**
- Pearson correlation is scale-invariant and shift-invariant — it only enforces relative ordering
- This means rendered depth can be at completely wrong absolute positions and still have low Pearson loss
- It doesn't prevent Gaussians from "floating" at incorrect depths as long as the depth ordering is preserved
- With initial weight 0.5 decaying by `exp(-3t)`, the strong early weight (~0.5) pushed initial geometry toward monocular depth ordering, potentially conflicting with photometric gradients and distorting early learning

**3. SH degree cap at 2 was probably fine but untestable.**
- Because the other two changes were so destructive, we can't isolate the SH=2 effect
- In principle, reducing SH from 16 to 9 coefficients should reduce overfitting
- But the test image is so full of floaters that any SH benefit is invisible

**4. The changes interacted badly.**
- More densification (from depth loss gradients) → 1.185M Gaussians (3x v5)
- No opacity reset → floaters never pruned
- More Gaussians + more floaters = worse novel views
- The depth loss was "working" in the sense that training loss was low and Gaussians were being created in depth-consistent locations, but the lack of pruning let garbage accumulate

## Key Insights

### The Overfitting vs. Underfitting Tradeoff
- **v5** slightly overfits (34 dB train, 25.78 dB test = 8.2 dB gap)
- **v7** also has a gap (33 dB train, 15.35 dB test = 17.7 dB gap) but the gap is from floaters, not overfitting
- The real enemy isn't overfitting in the parameter sense — it's geometric artifacts (floaters, elongated Gaussians) that happen to look correct from training angles

### Opacity Reset is Non-Negotiable (for now)
- Without it, we need a much stronger alternative: either very aggressive opacity regularization (maybe 10-100x what we used), or a completely different pruning strategy
- The periodic reset is a blunt instrument but it works: it forces all Gaussians to re-earn their opacity from near-zero

### Depth Supervision Needs Careful Integration
- Both COLMAP stereo (absolute metric depth, noisy) and monocular (relative depth, smooth but no metric) degraded quality
- The depth information might be valuable if applied as a very soft guide (weight 0.01-0.05) rather than a strong constraint (0.1-0.5)
- Or applied only during the first few thousand steps to guide initial geometry, then fully disabled

### Gaussian Count is Not the Bottleneck
- v5 (379K Gaussians) outperformed v7 (1.185M Gaussians) by 10 dB
- More Gaussians only help if they're in the right places
- The quality ceiling is about geometric accuracy, not representation capacity

## Recommended Next Experiment (v8)

**Strategy: Minimal delta from v5 (our best), one change at a time.**

### v8a: v5 + antialiased rendering only
- Keep everything from v5 unchanged (SH=3, opacity reset, no depth)
- Only add `rasterize_mode="antialiased"` and `absgrad=True`
- This isolates the anti-aliasing effect, which should be a pure improvement with no downside
- Expected: small improvement (~0.3-0.5 dB) from reduced aliasing artifacts

### v8b: v5 + SH degree cap at 2
- Keep everything from v5 unchanged
- Only change `sh_degree_max=2`
- This isolates the overfitting reduction effect
- Expected: could improve test PSNR by 0.5-1 dB, OR could lose ~1 dB if the scene needs view-dependent effects

### v8c: v5 + very mild monocular depth (weight 0.02, no decay)
- Keep everything from v5 unchanged (including opacity reset!)
- Add monocular depth Pearson loss at very low weight (0.02, constant — not decaying)
- This tests whether a gentle depth nudge helps without disrupting training
- Expected: uncertain, but with opacity reset still active, it should be safe

### v8d: v5 + antialiased + SH=2 + mild depth
- If v8a, v8b, v8c individually show gains, combine the winners
- Only combine techniques that independently improved quality

**The principle: never change more than one thing at a time. v7 changed 4 things at once and we can't tell which one (or which combination) caused the failure.**
