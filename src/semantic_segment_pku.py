"""Generate 5-class semantic segmentation for PKU-DyMVHumans using SCHP.

Maps SCHP's 20-class human parsing to 5 classes:
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
from torchvision import transforms

# SCHP 20-class labels → 5-class mapping
# SCHP classes: 0=bg, 1=hat, 2=hair, 3=sunglass, 4=upper-clothes, 5=skirt,
# 6=pants, 7=dress, 8=belt, 9=left-shoe, 10=right-shoe, 11=face,
# 12=left-leg, 13=right-leg, 14=left-arm, 15=right-arm, 16=bag,
# 17=scarf, 18=torso-skin, 19=socks
SCHP_TO_5CLASS = {
    0: 0,   # background
    1: 3,   # hat → clothing
    2: 2,   # hair
    3: 3,   # sunglasses → clothing
    4: 3,   # upper-clothes → clothing
    5: 3,   # skirt → clothing
    6: 3,   # pants → clothing
    7: 3,   # dress → clothing
    8: 3,   # belt → clothing
    9: 4,   # left-shoe → shoes
    10: 4,  # right-shoe → shoes
    11: 1,  # face → skin
    12: 1,  # left-leg → skin
    13: 1,  # right-leg → skin
    14: 1,  # left-arm → skin
    15: 1,  # right-arm → skin
    16: 3,  # bag → clothing
    17: 3,  # scarf → clothing
    18: 1,  # torso-skin → skin
    19: 4,  # socks → shoes
}


def load_schp_model(device="cuda"):
    """Load SCHP (Self-Correction for Human Parsing) model."""
    try:
        from schp import SCHP
        model = SCHP(num_classes=20)
        model.to(device)
        model.eval()
        return model
    except ImportError:
        # Try alternative: use the schp package from pip
        pass

    # Fallback: try loading from a known checkpoint path
    try:
        import torchvision.models as models
        # SCHP is typically a DeepLabV3+ with ResNet backbone
        # Load from checkpoint if available
        ckpt_paths = [
            Path("/workspace/schp/exp-schp-201908261155-lip.pth"),
            Path("schp/exp-schp-201908261155-lip.pth"),
        ]
        for ckpt in ckpt_paths:
            if ckpt.exists():
                print(f"Loading SCHP from {ckpt}")
                from schp.networks import init_model
                model = init_model('resnet101', num_classes=20, pretrained=None)
                state = torch.load(str(ckpt), map_location=device)
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                return model
    except Exception as e:
        print(f"SCHP load error: {e}")

    raise RuntimeError(
        "Could not load SCHP model. Install via:\n"
        "  pip install schp\n"
        "Or download checkpoint to /workspace/schp/"
    )


def get_schp_transform():
    """Get SCHP input transform."""
    return transforms.Compose([
        transforms.Resize((473, 473)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485],
                             std=[0.225, 0.224, 0.229]),
    ])


def run_schp_inference(model, image_pil, transform, device="cuda"):
    """Run SCHP inference on a single image.

    Args:
        model: SCHP model
        image_pil: PIL Image (RGB)
        transform: torchvision transform
        device: torch device

    Returns:
        (H, W) numpy array with 20-class labels
    """
    W_orig, H_orig = image_pil.size
    inp = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)
        if isinstance(output, (list, tuple)):
            output = output[-1]  # last output is finest
        # Upsample to original size
        output = torch.nn.functional.interpolate(
            output, size=(H_orig, W_orig), mode="bilinear", align_corners=True
        )
        labels = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    return labels


def map_to_5class(labels_20, fg_mask=None):
    """Map 20-class SCHP labels to 5-class labels.

    Args:
        labels_20: (H, W) uint8, values 0-19
        fg_mask: (H, W) bool, foreground mask (optional)

    Returns:
        (H, W) uint8, values 0-4
    """
    labels_5 = np.zeros_like(labels_20)
    for schp_class, our_class in SCHP_TO_5CLASS.items():
        labels_5[labels_20 == schp_class] = our_class

    # Enforce background from fg_mask
    if fg_mask is not None:
        labels_5[~fg_mask] = 0

    return labels_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference (not yet implemented)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    per_view = scene_dir / "per_view"
    device = args.device

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading SCHP model...")
    model = load_schp_model(device)
    transform = get_schp_transform()
    print("SCHP model loaded.")

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

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Load foreground mask
            mask_path = pha_dir / img_path.name
            fg_mask = None
            if mask_path.exists():
                mask_img = np.array(Image.open(mask_path).convert("L"))
                fg_mask = mask_img > 127

            # Run SCHP
            labels_20 = run_schp_inference(model, img, transform, device)

            # Map to 5 classes
            labels_5 = map_to_5class(labels_20, fg_mask)

            # Save as uint8 PNG
            Image.fromarray(labels_5).save(str(out_path))
            total_processed += 1

            if f_idx % 50 == 0:
                # Print class distribution for this frame
                unique, counts = np.unique(labels_5[labels_5 > 0], return_counts=True)
                dist = {int(u): int(c) for u, c in zip(unique, counts)}
                print(f"  Frame {f_idx}: classes {dist}")

    print(f"\nDone. Processed {total_processed} images.")
    print(f"Output: per_view/cam_N/semantic/*.png (uint8, 0-4)")


if __name__ == "__main__":
    main()
