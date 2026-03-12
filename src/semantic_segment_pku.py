"""Generate 5-class semantic segmentation for PKU-DyMVHumans using SegFormer.

Uses a SegFormer-B5 model fine-tuned on ATR/LIP human parsing dataset
(via HuggingFace transformers). Maps 18-class human parsing to 5 classes:
  0 = background
  1 = skin/face
  2 = hair
  3 = clothing
  4 = shoes

Usage:
    python semantic_segment_pku.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# SegFormer ATR/LIP 18-class labels → 5-class mapping
# ATR classes: 0=bg, 1=hat, 2=hair, 3=sunglasses, 4=upper-clothes, 5=skirt,
# 6=pants, 7=dress, 8=belt, 9=left-shoe, 10=right-shoe, 11=face,
# 12=left-leg, 13=right-leg, 14=left-arm, 15=right-arm, 16=bag, 17=scarf
ATR_TO_5CLASS = np.array([
    0,  # 0: background
    3,  # 1: hat → clothing
    2,  # 2: hair
    3,  # 3: sunglasses → clothing
    3,  # 4: upper-clothes → clothing
    3,  # 5: skirt → clothing
    3,  # 6: pants → clothing
    3,  # 7: dress → clothing
    3,  # 8: belt → clothing
    4,  # 9: left-shoe → shoes
    4,  # 10: right-shoe → shoes
    1,  # 11: face → skin
    1,  # 12: left-leg → skin
    1,  # 13: right-leg → skin
    1,  # 14: left-arm → skin
    1,  # 15: right-arm → skin
    3,  # 16: bag → clothing
    3,  # 17: scarf → clothing
], dtype=np.uint8)


def load_model(device="cuda"):
    """Load SegFormer human parsing model from HuggingFace."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_name = "matei-dorian/segformer-b5-finetuned-human-parsing"
    print(f"Loading {model_name}...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def run_inference(model, processor, image_pil, device="cuda"):
    """Run SegFormer inference on a single image.

    Args:
        model: SegFormer model
        processor: SegFormer image processor
        image_pil: PIL Image (RGB)
        device: torch device

    Returns:
        (H, W) numpy array with class labels (0-17)
    """
    W_orig, H_orig = image_pil.size
    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # (1, num_classes, H/4, W/4)

    # Upsample to original resolution
    logits = torch.nn.functional.interpolate(
        logits, size=(H_orig, W_orig), mode="bilinear", align_corners=False
    )
    labels = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return labels


def map_to_5class(labels, fg_mask=None):
    """Map 18-class ATR/LIP labels to 5-class labels.

    Args:
        labels: (H, W) uint8, values 0-17
        fg_mask: (H, W) bool, foreground mask (optional)

    Returns:
        (H, W) uint8, values 0-4
    """
    labels_5 = ATR_TO_5CLASS[np.clip(labels, 0, len(ATR_TO_5CLASS) - 1)]

    if fg_mask is not None:
        labels_5[~fg_mask] = 0

    return labels_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    per_view = scene_dir / "per_view"
    device = args.device

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(device)
    print("Model loaded.")

    # ── Process all cameras × all frames ──────────────────────────────────────
    cam_dirs = sorted(
        [d for d in per_view.iterdir() if d.is_dir() and d.name.startswith("cam_")],
        key=lambda d: int(d.name.split("_")[1])
    )

    total_processed = 0

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        images_dir = cam_dir / "images"
        pha_dir = cam_dir / "pha"
        semantic_dir = cam_dir / "semantic"
        semantic_dir.mkdir(exist_ok=True)

        frame_files = sorted(images_dir.glob("*.png"))
        if not frame_files:
            continue

        print(f"\n{cam_name}: {len(frame_files)} frames")

        for f_idx, img_path in enumerate(frame_files):
            out_path = semantic_dir / img_path.name
            if out_path.exists():
                continue  # skip already processed

            img = Image.open(img_path).convert("RGB")

            # Load foreground mask
            mask_path = pha_dir / img_path.name
            fg_mask = None
            if mask_path.exists():
                mask_img = np.array(Image.open(mask_path).convert("L"))
                fg_mask = mask_img > 127

            # Run inference
            labels = run_inference(model, processor, img, device)

            # Map to 5 classes
            labels_5 = map_to_5class(labels, fg_mask)

            # Save as uint8 PNG
            Image.fromarray(labels_5).save(str(out_path))
            total_processed += 1

            if f_idx % 50 == 0:
                unique, counts = np.unique(labels_5[labels_5 > 0], return_counts=True)
                dist = {int(u): int(c) for u, c in zip(unique, counts)}
                print(f"  Frame {f_idx}: classes {dist}")

    print(f"\nDone. Processed {total_processed} images.")
    print(f"Output: per_view/cam_N/semantic/*.png (uint8, 0-4)")


if __name__ == "__main__":
    main()
