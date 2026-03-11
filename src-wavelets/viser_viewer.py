"""Interactive Viser viewer for svox2 grids.

Renders the trained voxel grid in real-time using svox2's CUDA kernel,
streaming frames to a browser viewer via Viser.

Controls:
    Left-click drag:  Orbit around subject
    Right-click drag: Pan
    Scroll wheel:     Dolly in/out
    GUI sliders:      FOV (15-90°), Resolution (128-800px)

Usage:
    python viser_viewer.py output/fm_poc6/lego/fm/stage3.npz
    python viser_viewer.py output/fm_poc6/lego/fm/stage4.npz --port 8080
"""

import argparse
import math
import time
import threading

import numpy as np
import torch
import svox2
import viser


def quaternion_to_rotation_matrix(wxyz):
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


# World frame rotation: Viser (+Z up) -> NeRF synthetic (+Y up)
VISER_TO_NERF = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float32)


def render_frame(grid, wxyz, position, fov_rad, render_res):
    """Render one frame from Viser camera parameters."""
    R_viser = quaternion_to_rotation_matrix(wxyz)
    pos_viser = np.array(position, dtype=np.float32)

    R_nerf = VISER_TO_NERF @ R_viser
    t_nerf = VISER_TO_NERF @ pos_viser

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_nerf
    c2w[:3, 3] = t_nerf

    focal = 0.5 * render_res / np.tan(0.5 * fov_rad)

    c2w_torch = torch.from_numpy(c2w).cuda()
    cam = svox2.Camera(
        c2w_torch,
        focal, focal,
        render_res * 0.5, render_res * 0.5,
        render_res, render_res,
        ndc_coeffs=(-1.0, -1.0),
    )

    with torch.no_grad():
        img = grid.volume_render_image(cam, use_kernel=True)
        img.clamp_(0.0, 1.0)

    return (img.cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Interactive svox2 viewer")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render_res", type=int, default=512)
    args = parser.parse_args()

    # Load grid
    print(f"Loading {args.checkpoint}...")
    grid = svox2.SparseGrid.load(args.checkpoint, device="cuda")
    grid.opt.near_clip = 0.0
    grid.opt.background_brightness = 1.0

    reso = grid.links.shape[0]
    n_voxels = grid.density_data.shape[0]
    print(f"Grid: {reso}^3, {n_voxels:,} voxels, basis_dim={grid.basis_dim}")

    # Warm up CUDA
    _ = render_frame(grid, (1, 0, 0, 0), (0, -4, 0), 0.6, 256)
    print("CUDA warm-up done")

    # Start Viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # World up = +Z in Viser frame (maps to +Y in NeRF frame)
    server.scene.set_up_direction("+z")

    # GUI controls
    with server.gui.add_folder("Rendering"):
        res_slider = server.gui.add_slider(
            "Resolution", min=128, max=800, step=64,
            initial_value=args.render_res,
        )
        fov_slider = server.gui.add_slider(
            "FOV (deg)", min=15, max=90, step=1, initial_value=35,
        )
        fps_display = server.gui.add_text(
            "FPS", initial_value="--", disabled=True,
        )

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        # 35° FOV ~ 50mm equivalent: neutral perspective, minimal distortion.
        # Camera at ~5.0 distance to compensate for narrower FOV.
        initial_fov = math.radians(fov_slider.value)
        client.camera.fov = initial_fov
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (-3.5, -3.5, 2.0)

        render_lock = threading.Lock()
        last_render_time = [0.0]

        def do_render():
            now = time.time()
            if now - last_render_time[0] < 0.016:
                return
            if not render_lock.acquire(blocking=False):
                return
            try:
                t0 = time.time()
                res = int(res_slider.value)
                fov = math.radians(fov_slider.value)
                client.camera.fov = fov

                img = render_frame(
                    grid,
                    client.camera.wxyz,
                    client.camera.position,
                    fov,
                    res,
                )
                client.scene.set_background_image(img, format="jpeg")

                dt = time.time() - t0
                last_render_time[0] = time.time()
                fps_display.value = f"{1.0/max(dt, 1e-6):.0f} ({dt*1000:.0f}ms)"
            finally:
                render_lock.release()

        @client.camera.on_update
        def on_camera_update(_):
            do_render()

        @fov_slider.on_update
        def on_fov_update(_):
            do_render()

        @res_slider.on_update
        def on_res_update(_):
            do_render()

        # Initial render after client settles
        time.sleep(0.3)
        do_render()
        print("Client connected")

    print(f"\nViewer ready at http://localhost:{args.port}")
    print("Controls: left-drag=orbit, right-drag=pan, scroll=dolly")
    print("GUI: FOV slider (default 35° ~ 50mm), Resolution slider\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
