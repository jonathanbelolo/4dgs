"""Interactive 3D viewer using viser.

Renders wavelet volume models in real-time from arbitrary camera positions.
Accessible via browser — works remotely over SSH tunnels.

Usage:
    python viewer.py --checkpoint output/lego/best.pt --port 8080

    # Remote access (from local machine):
    ssh -L 8080:localhost:8080 B200
    # Then open http://localhost:8080 in browser
"""

import argparse
import time

import numpy as np
import torch

try:
    import viser
    import viser.transforms as vtf
except ImportError:
    raise ImportError(
        "viser is required for the interactive viewer. "
        "Install with: pip install viser"
    )

from eval import load_model
from renderer import render_image
from data import NerfSyntheticDataset


def launch_viewer(
    checkpoint_path: str,
    port: int = 8080,
    host: str = "0.0.0.0",
    default_resolution: int = 400,
):
    """Launch interactive viser viewer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}...")
    model, config, ckpt = load_model(checkpoint_path, device)
    print(f"Model loaded. PSNR={ckpt.get('psnr', 'N/A')}")

    # Get focal length from dataset (load just metadata)
    try:
        data = NerfSyntheticDataset(
            config.data_dir, config.scene, "test",
            resolution=config.train_resolution,
            white_bg=config.white_background,
            device="cpu",
        )
        default_focal = data.focal
    except Exception:
        # Fallback: standard NeRF Synthetic FOV
        default_focal = 0.5 * config.train_resolution / np.tan(0.5 * 0.6911112)

    server = viser.ViserServer(host=host, port=port)

    # GUI controls
    with server.gui.add_folder("Rendering"):
        resolution_slider = server.gui.add_slider(
            "Resolution", min=100, max=800, initial_value=default_resolution, step=50,
        )
        render_button = server.gui.add_button("Render")

    with server.gui.add_folder("Info"):
        psnr_text = server.gui.add_text("PSNR", initial_value=str(ckpt.get("psnr", "N/A")), disabled=True)
        status_text = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    @render_button.on_click
    def on_render(_) -> None:
        """Render from the current camera position for all connected clients."""
        for client in server.get_clients().values():
            render_for_client(client)

    def render_for_client(client):
        """Render the current view for a specific client."""
        status_text.value = "Rendering..."

        # Get camera parameters from viser
        cam = client.camera
        res = int(resolution_slider.value)

        # Build pose matrix from viser camera
        # viser gives us wxyz quaternion and position
        wxyz = cam.wxyz
        position = cam.position

        # Convert quaternion to rotation matrix
        R = vtf.SO3(wxyz=wxyz).as_matrix()

        # viser camera convention → OpenGL convention
        # viser looks along +Z, OpenGL looks along -Z
        # Flip Z and Y to match our renderer
        R_opengl = R.copy()
        R_opengl[:, 1] *= -1  # flip Y
        R_opengl[:, 2] *= -1  # flip Z

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_opengl
        pose[:3, 3] = np.array(position, dtype=np.float32)
        pose_tensor = torch.from_numpy(pose).to(device)

        # Scale focal for current resolution
        focal = default_focal * res / config.train_resolution

        with torch.no_grad():
            result = render_image(
                model, pose_tensor, res, res, focal, config,
            )

        img = (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        client.scene.set_background_image(img)
        status_text.value = f"Done ({res}×{res})"

    print(f"\nViewer running at http://localhost:{port}")
    print("Use 'Render' button to capture the current camera view.")
    print("Controls: Left-click drag to orbit, right-click to pan, scroll to zoom.")

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive wavelet volume viewer")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--resolution", type=int, default=400,
                        help="Default rendering resolution")
    args = parser.parse_args()

    launch_viewer(
        args.checkpoint,
        port=args.port,
        host=args.host,
        default_resolution=args.resolution,
    )
