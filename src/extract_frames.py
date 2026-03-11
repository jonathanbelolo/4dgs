"""Extract frames from Neu3D multi-view video dataset."""
import argparse
import subprocess
from pathlib import Path


def extract_frame(video_path: Path, output_path: Path, frame_idx: int = 0):
    """Extract a single frame from a video file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1",
        "-q:v", "1",  # highest quality JPEG
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene directory (e.g. /workspace/Data/Neu3D/coffee_martini)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for frames. Defaults to scene_dir/frames")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Which frame to extract (default: 0)")
    parser.add_argument("--all_frames", action="store_true",
                        help="Extract all frames (for 4DGS later)")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Number of frames to extract when --all_frames is set")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor (2 = half resolution)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scene_dir / "frames"

    videos = sorted(scene_dir.glob("cam*.mp4"))
    # cam00 is the held-out test camera
    train_videos = [v for v in videos if v.stem != "cam00"]
    test_videos = [v for v in videos if v.stem == "cam00"]

    print(f"Found {len(videos)} cameras ({len(train_videos)} train, {len(test_videos)} test)")

    if args.all_frames:
        frame_indices = range(args.num_frames)
    else:
        frame_indices = [args.frame_idx]

    for frame_idx in frame_indices:
        for split, cam_list in [("train", train_videos), ("test", test_videos)]:
            for video in cam_list:
                cam_name = video.stem
                if len(frame_indices) == 1:
                    out_path = output_dir / split / f"{cam_name}.jpg"
                else:
                    out_path = output_dir / split / f"frame_{frame_idx:04d}" / f"{cam_name}.jpg"

                if out_path.exists():
                    continue

                print(f"  Extracting {cam_name} frame {frame_idx} -> {out_path}")

                # Build ffmpeg command with optional downsampling
                out_path.parent.mkdir(parents=True, exist_ok=True)
                vf_filters = [f"select=eq(n\\,{frame_idx})"]
                if args.downsample > 1:
                    vf_filters.append(f"scale=iw/{args.downsample}:ih/{args.downsample}")

                cmd = [
                    "ffmpeg", "-y", "-i", str(video),
                    "-vf", ",".join(vf_filters),
                    "-vframes", "1",
                    "-q:v", "1",
                    str(out_path),
                ]
                subprocess.run(cmd, capture_output=True, check=True)

    print(f"Done. Frames saved to {output_dir}")


if __name__ == "__main__":
    main()
