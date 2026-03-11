"""Convert N3V dataset (LLFF format) to svox2-compatible format.

Reads poses_bounds.npy + extracted frame images, outputs:
  - transforms_train.json (NeRF synthetic format)
  - images/ directory with renamed PNG files

The N3V dataset uses LLFF camera convention:
  Rotation columns = [down, right, backwards]
  
We convert to OpenGL convention (matching NeRF synthetic):
  Rotation columns = [right, up, -forward]

Usage:
    python prepare_n3v.py /workspace/data/n3v/coffee_martini
"""

import argparse
import json
import os
import shutil

import numpy as np


def llff_to_opengl_c2w(poses_raw, bounds):
    """Convert LLFF poses (N, 3, 5) to OpenGL c2w matrices (N, 4, 4).
    
    LLFF columns: [down, right, backwards, translation, hwf]
    OpenGL columns: [right, up, -forward, translation]
    """
    N = poses_raw.shape[0]
    c2ws = np.zeros((N, 4, 4), dtype=np.float64)
    c2ws[:, 3, 3] = 1.0
    
    # Rotation mapping
    c2ws[:, :3, 0] = poses_raw[:, :3, 1]   # right = LLFF right
    c2ws[:, :3, 1] = -poses_raw[:, :3, 0]  # up = -LLFF down
    c2ws[:, :3, 2] = poses_raw[:, :3, 2]   # -forward = LLFF backwards
    
    # Translation
    c2ws[:, :3, 3] = poses_raw[:, :3, 3]
    
    return c2ws


def center_and_scale(c2ws, target_cam_dist=4.0):
    """Center scene at camera centroid, scale to target camera distance."""
    positions = c2ws[:, :3, 3]
    centroid = positions.mean(axis=0)
    
    # Center
    c2ws[:, :3, 3] -= centroid
    
    # Scale
    dists = np.linalg.norm(c2ws[:, :3, 3], axis=1)
    scale = target_cam_dist / dists.mean()
    c2ws[:, :3, 3] *= scale
    
    return c2ws, centroid, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to N3V scene (e.g. coffee_martini)")
    parser.add_argument("--frame_dir", default="images_frame0",
                        help="Subdirectory with extracted frame images")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: scene_dir/nerf_format)")
    parser.add_argument("--target_cam_dist", type=float, default=4.0,
                        help="Normalize average camera distance to this value")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Downsample images to this resolution (longest edge)")
    args = parser.parse_args()
    
    scene_dir = args.scene_dir
    output_dir = args.output_dir or os.path.join(scene_dir, "nerf_format")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load poses
    pb = np.load(os.path.join(scene_dir, "poses_bounds.npy"))
    poses_raw = pb[:, :15].reshape(-1, 3, 5)
    bounds = pb[:, 15:]
    N = poses_raw.shape[0]
    
    # Extract image dimensions and focal length
    h, w, focal = poses_raw[0, :, 4]
    h, w = int(h), int(w)
    print(f"Original: {w}x{h}, focal={focal:.1f}, {N} cameras")
    
    # Convert to OpenGL c2w
    c2ws = llff_to_opengl_c2w(poses_raw, bounds)
    
    # Center and scale
    c2ws, centroid, scale = center_and_scale(c2ws, args.target_cam_dist)
    scaled_focal = focal * scale if args.resolution is None else focal
    near = bounds[:, 0].min() * scale
    far = bounds[:, 1].max() * scale
    
    print(f"Scene centered at {centroid}")
    print(f"Scale factor: {scale:.4f}")
    print(f"Scaled near/far: {near:.2f} / {far:.2f}")
    
    # Determine output resolution
    if args.resolution:
        out_res = args.resolution
        res_scale = out_res / max(w, h)
        out_w = int(w * res_scale)
        out_h = int(h * res_scale)
        out_focal = focal * res_scale
    else:
        out_w, out_h = w, h
        out_focal = focal
    print(f"Output: {out_w}x{out_h}, focal={out_focal:.1f}")
    
    # Build transforms.json (NeRF synthetic format)
    # camera_angle_x = 2 * atan(w / (2 * focal))
    camera_angle_x = 2 * np.arctan(out_w / (2 * out_focal))
    
    # List frame images
    frame_dir = os.path.join(scene_dir, args.frame_dir)
    cam_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    
    if len(cam_files) != N:
        print(f"WARNING: {len(cam_files)} images but {N} cameras in poses!")
    
    frames = []
    img_out_dir = os.path.join(output_dir, "train")
    os.makedirs(img_out_dir, exist_ok=True)
    
    for i, cam_file in enumerate(cam_files):
        src = os.path.join(frame_dir, cam_file)
        
        if args.resolution:
            # Resize with ffmpeg
            dst = os.path.join(img_out_dir, f"r_{i:03d}.png")
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", src,
                "-vf", f"scale={out_w}:{out_h}",
                "-q:v", "1", dst,
            ], capture_output=True)
        else:
            dst = os.path.join(img_out_dir, f"r_{i:03d}.png")
            shutil.copy2(src, dst)
        
        frame = {
            "file_path": f"train/r_{i:03d}",
            "transform_matrix": c2ws[i].tolist(),
        }
        frames.append(frame)
    
    transforms = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames,
    }
    
    # Save as train and val (use same images for now)
    for split in ["train", "val", "test"]:
        out_path = os.path.join(output_dir, f"transforms_{split}.json")
        with open(out_path, "w") as f:
            json.dump(transforms, f, indent=2)
    
    # Save scene info
    info = {
        "original_resolution": [w, h],
        "output_resolution": [out_w, out_h],
        "focal": float(out_focal),
        "centroid": centroid.tolist(),
        "scale": float(scale),
        "near": float(near),
        "far": float(far),
        "scene_bound": float(far * 0.5),  # conservative
        "n_cameras": N,
    }
    with open(os.path.join(output_dir, "scene_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nSaved to {output_dir}")
    print(f"  transforms_{{train,val,test}}.json")
    print(f"  train/r_000.png .. r_{N-1:03d}.png")
    print(f"  scene_info.json")
    print(f"\nScene bound: {info[scene_bound]:.1f}")
    print(f"Camera angle X: {np.degrees(camera_angle_x):.1f}°")


if __name__ == "__main__":
    main()
