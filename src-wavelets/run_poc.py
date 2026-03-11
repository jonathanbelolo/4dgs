"""Run all frequency-matched PoC configurations and analyze results.

Convenience script that trains all four methods (FM, SP, SS, Hybrid),
then runs wavelet analysis to compare them.

Usage:
    # Run all training + analysis
    python run_poc.py --scene lego

    # Run only analysis (if training is already done)
    python run_poc.py --scene lego --analysis_only

    # Run a single mode
    python run_poc.py --scene lego --modes fm
"""

import argparse
import json
import os
import time

from config import Config
from train_frequency_matched import train_frequency_matched
from wavelet_analysis import compare_methods


MODES = ["fm", "sp", "ss", "hybrid"]


def run_poc(
    scene: str = "lego",
    data_dir: str = "data/nerf_synthetic",
    output_dir: str = "output/fm_poc",
    modes: list[str] | None = None,
    analysis_only: bool = False,
    num_channels: int = 28,
    batch_rays: int = 4096,
    device: str = "cuda",
):
    """Run complete frequency-matched PoC experiment."""
    if modes is None:
        modes = MODES

    base_config = Config(
        scene=scene,
        data_dir=data_dir,
        output_dir=output_dir,
        num_channels=num_channels,
        batch_rays=batch_rays,
        device=device,
        val_every=2500,
        log_every=100,
    )

    t0 = time.time()
    reports = {}

    # --- Training ---
    if not analysis_only:
        for mode in modes:
            print(f"\n{'#'*70}")
            print(f"#  Training: {mode.upper()}")
            print(f"{'#'*70}\n")

            mode_t0 = time.time()
            _, report = train_frequency_matched(base_config, mode=mode)
            reports[mode] = report
            reports[mode]["training_time"] = time.time() - mode_t0

            print(f"\n{mode.upper()} completed in "
                  f"{reports[mode]['training_time']/60:.1f} min")

        # Save combined training report
        combined_report_path = os.path.join(output_dir, scene, "training_report.json")
        with open(combined_report_path, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\nTraining report saved to: {combined_report_path}")

    # --- Analysis ---
    print(f"\n{'#'*70}")
    print(f"#  Wavelet Analysis")
    print(f"{'#'*70}\n")

    # Collect checkpoints that exist
    checkpoints = []
    labels = []
    for mode in modes:
        ckpt_path = os.path.join(output_dir, scene, mode, "final.pt")
        if os.path.exists(ckpt_path):
            checkpoints.append(ckpt_path)
            labels.append(mode.upper())
        else:
            print(f"  Warning: {ckpt_path} not found, skipping {mode}")

    if len(checkpoints) >= 2:
        analysis_dir = os.path.join(output_dir, scene, "analysis")
        compare_methods(
            checkpoints, labels, base_config,
            analysis_dir, wavelet="bior4.4", levels=3,
        )
    elif len(checkpoints) == 1:
        print("  Only one checkpoint found — skipping comparative analysis.")
        print("  Run with multiple modes for comparison.")
    else:
        print("  No checkpoints found. Run training first.")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run frequency-matched PoC experiment"
    )
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--data_dir", default="data/nerf_synthetic")
    parser.add_argument("--output_dir", default="output/fm_poc")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=MODES,
                        help="Which modes to run (default: all)")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip training, only run analysis")
    parser.add_argument("--num_channels", type=int, default=28,
                        help="Feature channels (4=density+RGB, 28=density+SH)")
    parser.add_argument("--batch_rays", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_poc(
        scene=args.scene,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        modes=args.modes,
        analysis_only=args.analysis_only,
        num_channels=args.num_channels,
        batch_rays=args.batch_rays,
        device=args.device,
    )
