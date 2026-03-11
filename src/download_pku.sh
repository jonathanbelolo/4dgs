#!/bin/bash
# Download PKU-DyMVHumans 4K scene to RunPod
# Run this on RunPod: bash src/download_pku.sh
set -e

SCENE="4K_Studios_Show_Pair_f16f17"
DATA_DIR="/workspace/Data/PKU-DyMVHumans"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading $SCENE (26.6 GB) ==="
pip install -q huggingface_hub

# Download the tar file
huggingface-cli download zxyun/PKU-DyMVHumans \
    "part1/${SCENE}.tar" \
    --repo-type dataset \
    --local-dir .

echo "=== Extracting ==="
tar xf "part1/${SCENE}.tar"

echo "=== Verifying structure ==="
echo "Cameras:"
ls "${SCENE}/cams/" | head -5
echo "..."
ls "${SCENE}/cams/" | wc -l

echo ""
echo "Frames:"
ls "${SCENE}/per_frame/" | head -5
echo "..."
ls "${SCENE}/per_frame/" | wc -l

echo ""
echo "Images in first frame:"
FIRST_FRAME=$(ls "${SCENE}/per_frame/" | head -1)
ls "${SCENE}/per_frame/${FIRST_FRAME}/images/" | head -5
echo "..."
ls "${SCENE}/per_frame/${FIRST_FRAME}/images/" | wc -l

echo ""
echo "COLMAP data:"
ls "${SCENE}/data_COLMAP/" 2>/dev/null | head -5 || echo "(not found)"

echo ""
echo "=== Done. Scene at: ${DATA_DIR}/${SCENE} ==="
