#!/bin/bash
# Download NeRF Synthetic dataset
set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/nerf_synthetic/lego" ]; then
    echo "NeRF Synthetic dataset already exists at $DATA_DIR/nerf_synthetic/"
    exit 0
fi

echo "Downloading NeRF Synthetic dataset..."
wget -q --show-progress -O "$DATA_DIR/nerf_synthetic.zip" \
    "https://huggingface.co/datasets/ipmlab/NeRF-Synthetic/resolve/main/nerf_synthetic.zip"

echo "Extracting..."
unzip -q "$DATA_DIR/nerf_synthetic.zip" -d "$DATA_DIR"
rm "$DATA_DIR/nerf_synthetic.zip"

echo "Done. Dataset at $DATA_DIR/nerf_synthetic/"
ls "$DATA_DIR/nerf_synthetic/"
