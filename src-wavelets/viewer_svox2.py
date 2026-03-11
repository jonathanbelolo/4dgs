"""Interactive 3D viewer for svox2 (Plenoxels) checkpoints using viser.

Usage:
    python viewer_svox2.py --ckpt /path/to/ckpt.npz --port 7080

    # Remote access:
    ssh -L 7080:localhost:7080 B200
    # Then open http://localhost:7080
"""

import argparse
import threading
import time
import traceback

import numpy as np
import torch
import svox2
import viser
import viser.transforms as vtf


# svox2's nerf dataset loader applies:
#   c2w = c2w @ diag(1,-1,-1,1)   (OpenGL -> OpenCV)
#   c2w[:3, 3] *= 2/3             (scene scale)
SCENE_SCALE = 2.0 / 3.0
CAM_TRANS = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def launch_viewer(
    ckpt_path: str,
    port: int = 7080,
    host: str = "0.0.0.0",
    default_resolution: int = 400,
):
    device = "cuda:0"

    print(f"Loading svox2 checkpoint from {ckpt_path}...")
    grid = svox2.SparseGrid.load(ckpt_path, device=device)
    grid.opt.near_clip = 0.0
    grid.opt.background_brightness = 1.0
    R = grid.links.shape[0]
    N = grid.density_data.shape[0]
    print(f"Grid: {R}³, {N:,} sparse voxels")

    render_lock = threading.Lock()
    server = viser.ViserServer(host=host, port=port)

    with server.gui.add_folder("Rendering"):
        res_slider = server.gui.add_slider(
            "Resolution", min=100, max=800, initial_value=default_resolution, step=50,
        )
    with server.gui.add_folder("Info"):
        server.gui.add_text("Grid", initial_value=f"{R}³", disabled=True)
        status_text = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    def render_for_client(client: viser.ClientHandle) -> None:
        try:
            cam = client.camera
            res = int(res_slider.value)

            # viser camera -> c2w in OpenGL convention
            R_mat = vtf.SO3(wxyz=cam.wxyz).as_matrix()
            c2w_gl = np.eye(4, dtype=np.float32)
            c2w_gl[:3, :3] = R_mat
            c2w_gl[:3, 3] = np.array(cam.position, dtype=np.float32)

            # OpenGL -> OpenCV + scene scale (matches svox2 training)
            c2w = c2w_gl @ CAM_TRANS
            c2w[:3, 3] *= SCENE_SCALE

            # Focal from viser FOV
            fov = cam.fov
            focal = 0.5 * res / np.tan(0.5 * fov)

            c2w_t = torch.from_numpy(c2w).to(device)
            svox2_cam = svox2.Camera(
                c2w_t, focal, focal,
                res * 0.5, res * 0.5,
                res, res,
                ndc_coeffs=(-1.0, -1.0),
            )

            with render_lock:
                with torch.no_grad():
                    im = grid.volume_render_image(svox2_cam, use_kernel=True)
                    im.clamp_(0.0, 1.0)

            # Flip vertically (svox2 renders Y-down, viser expects Y-up)
            img = np.ascontiguousarray(np.flipud(
                (im.cpu().numpy() * 255).astype(np.uint8),
            ))
            client.scene.set_background_image(img, format="jpeg", jpeg_quality=80)
            status_text.value = f"Done ({res}x{res})"
        except Exception:
            traceback.print_exc()

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        print(f"Client {client.client_id} connected")

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            render_for_client(client)

    print(f"\nViewer at http://localhost:{port}")
    print("Orbit with left-drag, pan with right-drag, zoom with scroll.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=7080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--resolution", type=int, default=400)
    a = p.parse_args()
    launch_viewer(a.ckpt, a.port, a.host, a.resolution)
