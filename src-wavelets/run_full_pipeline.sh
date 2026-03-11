#!/bin/bash
# Full 5-stage frequency-matched pipeline with svox2-4d improvements.
# Run: nohup bash run_full_pipeline.sh &> output/fm_poc6/run.log &
set -e

OUT="output/fm_poc6"
MODE="fm"
COMMON="--data_dir data/nerf_synthetic --scene lego --device cuda --val_every 5000 --log_every 1000 --batch_rays 4096"

mkdir -p "$OUT/lego/fm"

echo "=============================================="
echo "  svox2-4d full pipeline: fm_poc6"
echo "  Mods: RMS preservation, sparse upsample, adaptive LR"
echo "  Start: $(date)"
echo "=============================================="

# Stage 0: 136px / 136³, from scratch
echo ""
echo ">>> Stage 0: 136px / 136³ (from scratch)"
python -u train_frequency_matched.py --mode $MODE --stage 0 \
    --output_dir $OUT $COMMON

# Stage 1: 264px / 264³, resume from stage 0
echo ""
echo ">>> Stage 1: 264px / 264³"
python -u train_frequency_matched.py --mode $MODE --stage 1 \
    --resume $OUT/lego/fm/stage0.npz \
    --output_dir $OUT $COMMON

# Stage 2: 520px / 520³, resume from stage 1
echo ""
echo ">>> Stage 2: 520px / 520³"
python -u train_frequency_matched.py --mode $MODE --stage 2 \
    --resume $OUT/lego/fm/stage1.npz \
    --output_dir $OUT $COMMON

# Stage 3: 800px / 520³, resume from stage 2
echo ""
echo ">>> Stage 3: 800px / 520³"
python -u train_frequency_matched.py --mode $MODE --stage 3 \
    --resume $OUT/lego/fm/stage2.npz \
    --output_dir $OUT $COMMON

# Stage 4: 800px / 1032³, resume from stage 3 (sparse_upsample + adaptive LR)
echo ""
echo ">>> Stage 4: 800px / 1032³ (svox2-4d: sparse_upsample + adaptive LR)"
python -u train_frequency_matched.py --mode $MODE --stage 4 \
    --resume $OUT/lego/fm/stage3.npz \
    --output_dir $OUT $COMMON

echo ""
echo "=============================================="
echo "  Pipeline complete: $(date)"
echo "=============================================="
