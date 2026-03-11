"""Interactive 4D Gaussian Splat viewer using viser.

Loads a trained 4D PLY, renders with gsplat on the GPU, and serves
an interactive viewer with a time slider in the browser.

Usage:
    python viewer_4d.py --ply_path outputs/4d_v1b/point_cloud_4d_final.ply
    Then open http://localhost:7007 (or SSH tunnel: ssh -L 7007:localhost:7007 runpod)
"""
import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser

import gsplat
from train_4d import GaussianModel4D


def load_4d_ply(ply_path, device="cuda"):
    """Load a 4D Gaussian model from PLY."""
    model = GaussianModel4D(device=device)
    model.init_from_static_ply(ply_path)

    # The static PLY loader doesn't load temporal params — read them manually
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    if "mu_t" in v.data.dtype.names:
        model.mu_t = torch.tensor(
            np.array(v["mu_t"]), dtype=torch.float32, device=device
        ).requires_grad_(False)
        model.s_t = torch.tensor(
            np.array(v["s_t"]), dtype=torch.float32, device=device
        ).requires_grad_(False)
        model.velocity = torch.tensor(
            np.stack([v["velocity_0"], v["velocity_1"], v["velocity_2"]], axis=1),
            dtype=torch.float32, device=device,
        ).requires_grad_(False)
        print(f"Loaded temporal params: mu_t, s_t, velocity")

        # Load temporal derivatives if present
        if "d_scale_0" in v.data.dtype.names:
            n = len(v)
            model.d_scales = torch.tensor(
                np.stack([v["d_scale_0"], v["d_scale_1"], v["d_scale_2"]], axis=1),
                dtype=torch.float32, device=device,
            ).requires_grad_(False)
            model.d_quats = torch.tensor(
                np.stack([v["d_rot_0"], v["d_rot_1"], v["d_rot_2"], v["d_rot_3"]], axis=1),
                dtype=torch.float32, device=device,
            ).requires_grad_(False)
            d_sh0 = np.stack([v["d_f_dc_0"], v["d_f_dc_1"], v["d_f_dc_2"]], axis=1)
            model.d_sh0 = torch.tensor(d_sh0, dtype=torch.float32,
                                        device=device).reshape(n, 1, 3).requires_grad_(False)
            d_shN = np.stack([v[f"d_f_rest_{i}"] for i in range(45)], axis=1)
            model.d_shN = torch.tensor(d_shN, dtype=torch.float32,
                                        device=device).reshape(n, 15, 3).requires_grad_(False)
            print(f"  Loaded temporal derivatives: d_scales, d_quats, d_sh0, d_shN")
        else:
            n = len(v)
            model.d_scales = torch.zeros(n, 3, device=device).requires_grad_(False)
            model.d_quats = torch.zeros(n, 4, device=device).requires_grad_(False)
            model.d_sh0 = torch.zeros(n, 1, 3, device=device).requires_grad_(False)
            model.d_shN = torch.zeros(n, 15, 3, device=device).requires_grad_(False)
    else:
        print("WARNING: No temporal params in PLY, using defaults")

    return model


def wxyz_to_rotation_matrix(wxyz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


@torch.no_grad()
def render_at_time(model, c2w, K, W, H, t0, near=0.01, far=1000.0, device="cuda"):
    """Render the 4D model at a given camera pose and timestamp."""
    viewmat = torch.linalg.inv(c2w)

    dt = model.mu_t - t0
    means_t0 = model.means + model.velocity * dt.unsqueeze(-1)

    s_t = torch.exp(model.s_t).clamp(min=1e-6)
    temporal_weight = torch.exp(-0.5 * (dt / s_t) ** 2)
    opacities_t0 = torch.sigmoid(model.opacities) * temporal_weight

    # Time-varying scales, quats, SH
    scales_t0 = model.scales + model.d_scales * dt.unsqueeze(-1)
    scales = torch.exp(scales_t0)
    quats_t0 = model.quats + model.d_quats * dt.unsqueeze(-1)
    quats_t0 = F.normalize(quats_t0, dim=-1)
    dt_sh = dt.unsqueeze(-1).unsqueeze(-1)
    sh0_t0 = model.sh0 + model.d_sh0 * dt_sh
    shN_t0 = model.shN + model.d_shN * dt_sh
    sh_coeffs = torch.cat([sh0_t0, shN_t0], dim=1)

    renders, _, _ = gsplat.rasterization(
        means=means_t0,
        quats=quats_t0,
        scales=scales,
        opacities=opacities_t0,
        colors=sh_coeffs,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        near_plane=near,
        far_plane=far,
        sh_degree=3,
        render_mode="RGB",
        rasterize_mode="classic",
        absgrad=False,
    )

    return renders[0].clamp(0, 1)  # (H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="4D Gaussian Splat Viewer")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--render_width", type=int, default=960)
    parser.add_argument("--render_height", type=int, default=540)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Scene dir to load initial camera pose from poses_bounds.npy")
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading model from {args.ply_path}...")
    model = load_4d_ply(args.ply_path, device=device)
    print(f"Loaded {model.num_gaussians:,} Gaussians")

    # Load initial camera pose if scene_dir provided
    init_c2w = None
    if args.scene_dir:
        from data import load_llff_poses
        c2w_all, hwf, bounds = load_llff_poses(
            str(Path(args.scene_dir) / "poses_bounds.npy"))
        init_c2w = c2w_all[1]  # cam01 (first training camera)
        print(f"Loaded initial camera from poses_bounds.npy")

    W, H = args.render_width, args.render_height

    server = viser.ViserServer(port=args.port)

    # Neu3D / LLFF convention: Y is up
    server.scene.set_up_direction("+y")

    # GUI controls
    time_slider = server.gui.add_slider(
        "Time", min=0.0, max=1.0, step=0.01, initial_value=0.5)
    fps_text = server.gui.add_text("FPS", initial_value="--", disabled=True)
    gaussian_text = server.gui.add_text(
        "Gaussians", initial_value=f"{model.num_gaussians:,}", disabled=True)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    play_speed = server.gui.add_slider(
        "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)

    print(f"\nViewer ready at http://localhost:{args.port}")
    print(f"Render resolution: {W}x{H}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} runpod")

    def do_render_for_client(client):
        """Render current view for a specific client."""
        camera = client.camera
        t_start = time.time()

        pos = np.array(camera.position)
        R = wxyz_to_rotation_matrix(np.array(camera.wxyz))

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        # viser uses OpenGL (Y-up, -Z forward), gsplat uses OpenCV (Y-down, Z forward)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)

        fov_y = camera.fov
        fy = H / (2.0 * math.tan(fov_y / 2.0))
        fx = fy
        K = torch.tensor([
            [fx, 0, W / 2.0],
            [0, fy, H / 2.0],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        t0 = float(time_slider.value)

        rendered = render_at_time(
            model, c2w, K, W, H, t0,
            near=args.near, far=args.far, device=device)

        img = (rendered.cpu().numpy() * 255).astype(np.uint8)
        client.scene.set_background_image(img, format="jpeg")

        dt = time.time() - t_start
        fps_text.value = f"{1.0/dt:.1f}"

    @server.on_client_connect
    def on_connect(client):
        print(f"Client connected: {client.client_id}")

        # Set initial camera to a training viewpoint
        if init_c2w is not None:
            # Convert OpenCV c2w to viser OpenGL convention
            c2w_gl = init_c2w.copy().astype(np.float64)
            c2w_gl[:3, 1] *= -1  # flip Y back
            c2w_gl[:3, 2] *= -1  # flip Z back
            pos = c2w_gl[:3, 3]
            R = c2w_gl[:3, :3]
            # Convert rotation to quaternion (w, x, y, z)
            from scipy.spatial.transform import Rotation
            quat_xyzw = Rotation.from_matrix(R).as_quat()  # scipy: (x,y,z,w)
            wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            client.camera.position = pos
            client.camera.wxyz = wxyz

        render_lock = threading.Lock()

        def safe_render(camera):
            if not render_lock.acquire(blocking=False):
                return
            try:
                do_render_for_client(client)
            finally:
                render_lock.release()

        # Re-render when camera moves
        client.camera.on_update(lambda cam: safe_render(cam))

        # Re-render when time slider changes
        @time_slider.on_update
        def _(_):
            safe_render(client.camera)

    # Playback loop — renders directly for each connected client
    def playback_loop():
        while True:
            time.sleep(0.033)  # ~30 fps target
            if not playing.value:
                continue
            t = time_slider.value + 0.01 * play_speed.value
            if t > 1.0:
                t = 0.0
            time_slider.value = t
            # Trigger render for all connected clients
            for client_id, client in server.get_clients().items():
                try:
                    do_render_for_client(client)
                except Exception:
                    pass

    # Store per-client render functions
    client_renderers = {}

    playback_thread = threading.Thread(target=playback_loop, daemon=True)
    playback_thread.start()

    # Keep alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer")


if __name__ == "__main__":
    main()
