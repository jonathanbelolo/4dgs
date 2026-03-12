"""Fit SMPL-X to PKU-DyMVHumans using canonical reference frame.

Stage A: Per-camera HMR2.0 initialization (body pose + shape)
Stage B: Multi-view SMPL-X joint optimization (all 60 cameras)

Uses the canonical bounding box from canonical_bbox.py to ensure consistent
crops across all cameras, eliminating depth scale ambiguity and coordinate
frame drift.

Usage:
    python fit_smplx_canonical.py --scene_dir /path/to/1080_Kungfu_Basic_Single_c24
"""
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ─── Constants ───────────────────────────────────────────────────────────────

SMPLX_NUM_VERTICES = 10475
SMPLX_NUM_JOINTS = 55
SMPLX_BODY_JOINTS = 21
SMPLX_HAND_JOINTS = 15  # per hand

# Number of cameras to use for HMR2.0 initialization (evenly spaced)
NUM_INIT_CAMERAS = 16

# Optimization hyperparameters
OPT_ITERS_STAGE_B = 200
OPT_LR_BODY_POSE = 1e-2
OPT_LR_GLOBAL_ORIENT = 5e-3
OPT_LR_TRANSL = 1e-2
OPT_LR_BETAS = 1e-3
OPT_LR_HAND = 5e-3
OPT_LAMBDA_REPROJ = 1.0
OPT_LAMBDA_MASK = 0.5
OPT_LAMBDA_PRIOR = 0.01
OPT_LAMBDA_SHAPE = 0.005
OPT_LAMBDA_SMOOTH = 0.1
OPT_LAMBDA_ACCEL = 0.05


# ─── Helpers ─────────────────────────────────────────────────────────────────

def project_points(points_3d, K, w2c):
    """Project 3D points to 2D using camera intrinsics and extrinsics.

    Args:
        points_3d: (N, 3) world-space points
        K: (3, 3) intrinsics
        w2c: (4, 4) world-to-camera

    Returns:
        (N, 2) pixel coordinates
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ points_3d.T + t[:, None]).T  # (N, 3)
    pts_2d = (K @ pts_cam.T).T  # (N, 3)
    return pts_2d[:, :2] / np.clip(pts_2d[:, 2:3], 1e-6, None)


def project_bbox_to_camera(origin, axes, half_extents, K, w2c, H, W):
    """Project 3D canonical bounding box to 2D crop rectangle.

    Args:
        origin: (3,) canonical origin
        axes: (3, 3) canonical axes [right, up, forward]
        half_extents: (3,) half-widths
        K: (3, 3) intrinsics
        w2c: (4, 4) world-to-camera
        H, W: image dimensions

    Returns:
        (x1, y1, x2, y2) crop rectangle in pixel coordinates, or None if behind camera
    """
    # Generate 8 corners of the bounding box
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                      [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    corners_local = signs * half_extents  # (8, 3)
    corners_world = origin + corners_local @ axes  # (8, 3)

    # Check if any corners are behind the camera
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    corners_cam = (R @ corners_world.T + t[:, None]).T
    if (corners_cam[:, 2] <= 0).all():
        return None

    # Project visible corners
    valid = corners_cam[:, 2] > 0.01
    corners_2d = project_points(corners_world[valid], K, w2c)

    x1 = max(0, int(corners_2d[:, 0].min()) - 20)
    y1 = max(0, int(corners_2d[:, 1].min()) - 20)
    x2 = min(W, int(corners_2d[:, 0].max()) + 20)
    y2 = min(H, int(corners_2d[:, 1].max()) + 20)

    if x2 - x1 < 50 or y2 - y1 < 50:
        return None

    return (x1, y1, x2, y2)


def select_evenly_spaced_cameras(reg_indices, n_select):
    """Select n_select evenly spaced cameras from registered indices."""
    if len(reg_indices) <= n_select:
        return reg_indices
    step = len(reg_indices) / n_select
    return [reg_indices[int(i * step)] for i in range(n_select)]


def load_foreground_mask(mask_path):
    """Load binary foreground mask from alpha image."""
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 127


def load_semantic_mask(semantic_path):
    """Load 5-class semantic mask."""
    if not semantic_path.exists():
        return None
    return np.array(Image.open(semantic_path))


# ─── Stage A: HMR2.0 Per-Camera Initialization ──────────────────────────────

def init_hmr2():
    """Initialize HMR2.0 (4DHumans) model.

    Returns:
        model: HMR2.0 predictor
        cfg: model config
    """
    try:
        from hmr2.models import load_hmr2
        model, cfg = load_hmr2()
        model.eval()
        return model, cfg
    except ImportError:
        pass

    # Alternative import path
    try:
        from fourDHumans import load_model
        model = load_model()
        model.eval()
        return model, None
    except ImportError:
        pass

    raise RuntimeError(
        "Could not load HMR2.0. Install via:\n"
        "  pip install 4dhumans\n"
        "Or clone https://github.com/shubham-goel/4D-Humans"
    )


def run_hmr2_on_crop(hmr2_model, image_pil, crop_rect, device="cuda"):
    """Run HMR2.0 on a cropped region of an image.

    Args:
        hmr2_model: HMR2.0 model
        image_pil: PIL Image (RGB, full resolution)
        crop_rect: (x1, y1, x2, y2) crop rectangle
        device: torch device

    Returns:
        dict with keys: body_pose (63,), global_orient (3,), betas (10,),
                        cam_trans (3,), keypoints_2d (44, 3), vertices (6890, 3)
        or None if detection fails
    """
    x1, y1, x2, y2 = crop_rect
    crop = image_pil.crop((x1, y1, x2, y2))

    # HMR2.0 expects 256x256 input
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    inp = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            # HMR2.0 expects a dict with 'img' key
            batch = {"img": inp}
            output = hmr2_model(batch)
        except Exception as e:
            print(f"  HMR2.0 inference failed: {e}")
            return None

    # Extract predictions
    if isinstance(output, dict):
        pred = output
    elif isinstance(output, (list, tuple)):
        pred = output[-1] if isinstance(output[-1], dict) else {"pred_smpl_params": output}
    else:
        return None

    result = {}

    # Extract SMPL parameters (HMR2.0 outputs SMPL, not SMPL-X)
    if "pred_smpl_params" in pred:
        smpl_params = pred["pred_smpl_params"]
        result["body_pose"] = smpl_params.get("body_pose", torch.zeros(1, 63))[0].cpu().numpy()
        result["global_orient"] = smpl_params.get("global_orient", torch.zeros(1, 3))[0].cpu().numpy()
        result["betas"] = smpl_params.get("betas", torch.zeros(1, 10))[0].cpu().numpy()
    else:
        # Fallback: try direct attribute access
        for key in ["body_pose", "global_orient", "betas"]:
            if key in pred:
                val = pred[key]
                if isinstance(val, torch.Tensor):
                    val = val[0].cpu().numpy() if val.dim() > 1 else val.cpu().numpy()
                result[key] = val

    # Camera translation (weak perspective)
    if "pred_cam" in pred:
        cam = pred["pred_cam"]
        if isinstance(cam, torch.Tensor):
            result["cam_trans"] = cam[0].cpu().numpy() if cam.dim() > 1 else cam.cpu().numpy()
    elif "cam_trans" in pred:
        result["cam_trans"] = pred["cam_trans"][0].cpu().numpy()

    # 2D keypoints
    if "pred_keypoints_2d" in pred:
        kp = pred["pred_keypoints_2d"]
        if isinstance(kp, torch.Tensor):
            result["keypoints_2d"] = kp[0].cpu().numpy() if kp.dim() > 2 else kp.cpu().numpy()

    # Vertices
    if "pred_vertices" in pred:
        verts = pred["pred_vertices"]
        if isinstance(verts, torch.Tensor):
            result["vertices"] = verts[0].cpu().numpy() if verts.dim() > 2 else verts.cpu().numpy()

    return result if "body_pose" in result else None


def weak_persp_to_full_persp(cam_trans, K, crop_rect, img_H, img_W):
    """Convert HMR2.0 weak perspective camera to full perspective translation.

    HMR2.0 outputs [s, tx, ty] in normalized crop coordinates.
    We convert to full 3D translation using known intrinsics.

    Args:
        cam_trans: (3,) weak perspective params [s, tx, ty]
        K: (3, 3) intrinsics
        crop_rect: (x1, y1, x2, y2)
        img_H, img_W: full image dimensions

    Returns:
        (3,) translation in camera space
    """
    s, tx, ty = cam_trans
    x1, y1, x2, y2 = crop_rect
    crop_cx = (x1 + x2) / 2.0
    crop_cy = (y1 + y2) / 2.0
    crop_size = max(x2 - x1, y2 - y1)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Depth from weak perspective scale
    # s ≈ 2 * focal / (body_height_px * depth)
    # Approximate body height as ~crop_size * 0.8
    tz = 2.0 * fx / (s * crop_size + 1e-6)

    # XY from weak perspective offsets
    # tx, ty are in [-1, 1] normalized crop coords
    px = crop_cx + tx * crop_size / 2.0
    py = crop_cy + ty * crop_size / 2.0

    t_x = (px - cx) * tz / fx
    t_y = (py - cy) * tz / fy

    return np.array([t_x, t_y, tz])


def run_stage_a(scene_dir, canonical, calib, device="cuda"):
    """Stage A: Per-camera HMR2.0 initialization.

    Args:
        scene_dir: Path to scene directory
        canonical: dict from canonical_frame.npz
        calib: dict from calibration.npz
        device: torch device

    Returns:
        init_params: dict with median body_pose, betas, global_orient
    """
    w2c = calib["w2c"]
    c2w = calib["c2w"]
    K = calib["K"]
    registered = calib["registered"]
    cam_names = calib["cam_names"]
    reg_indices = np.where(registered)[0]

    origin = canonical["origin"]
    axes = canonical["axes"]
    half_extents = canonical["bbox_half_extents"]

    per_view = scene_dir / "per_view"

    # Select evenly spaced cameras for initialization
    init_cams = select_evenly_spaced_cameras(list(reg_indices), NUM_INIT_CAMERAS)
    print(f"Stage A: Using {len(init_cams)} cameras for HMR2.0 initialization")

    # Load HMR2.0
    print("  Loading HMR2.0 model...")
    hmr2_model, _ = init_hmr2()
    hmr2_model = hmr2_model.to(device)
    print("  HMR2.0 loaded.")

    # Get image dimensions from first camera
    sample_img = Image.open(per_view / cam_names[reg_indices[0]] / "images" / "000000.png")
    img_W, img_H = sample_img.size

    # Run HMR2.0 on each selected camera
    all_body_pose = []
    all_betas = []
    all_global_orient = []
    all_transl_world = []

    for cam_idx in init_cams:
        cam_name = cam_names[cam_idx]

        # Project canonical bbox to this camera
        crop_rect = project_bbox_to_camera(
            origin, axes, half_extents, K[cam_idx], w2c[cam_idx], img_H, img_W
        )
        if crop_rect is None:
            print(f"  {cam_name}: bbox behind camera, skipping")
            continue

        # Load image
        img_path = per_view / cam_name / "images" / "000000.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")

        # Run HMR2.0
        result = run_hmr2_on_crop(hmr2_model, img, crop_rect, device)
        if result is None:
            print(f"  {cam_name}: HMR2.0 detection failed")
            continue

        all_body_pose.append(result["body_pose"])
        all_betas.append(result["betas"])
        if "global_orient" in result:
            all_global_orient.append(result["global_orient"])

        # Convert weak perspective to full perspective, then to world space
        if "cam_trans" in result:
            t_cam = weak_persp_to_full_persp(
                result["cam_trans"], K[cam_idx], crop_rect, img_H, img_W
            )
            # Camera space to world space
            t_world = c2w[cam_idx][:3, :3] @ t_cam + c2w[cam_idx][:3, 3]
            all_transl_world.append(t_world)

        print(f"  {cam_name}: OK (crop {crop_rect[0]},{crop_rect[1]} → {crop_rect[2]},{crop_rect[3]})")

    if not all_body_pose:
        raise RuntimeError("HMR2.0 failed on all cameras!")

    # Aggregate: median across cameras
    init_params = {
        "body_pose": np.median(all_body_pose, axis=0),
        "betas": np.median(all_betas, axis=0),
    }
    if all_global_orient:
        init_params["global_orient"] = np.median(all_global_orient, axis=0)
    else:
        init_params["global_orient"] = np.zeros(3)

    if all_transl_world:
        init_params["transl"] = np.median(all_transl_world, axis=0)
    else:
        init_params["transl"] = origin.copy()

    print(f"  Stage A complete: {len(all_body_pose)} cameras contributed")
    return init_params


# ─── Stage B: Multi-view SMPL-X Optimization ────────────────────────────────

def load_smplx_model(model_path=None, device="cuda"):
    """Load SMPL-X body model.

    Returns:
        smplx model instance
    """
    try:
        import smplx

        # Try common model paths
        if model_path is None:
            candidates = [
                Path("/workspace/body_models/smplx"),
                Path("/workspace/smplx_models"),
                Path("body_models/smplx"),
                Path.home() / ".smplx" / "models",
            ]
            for p in candidates:
                if p.exists():
                    model_path = str(p.parent)
                    break

        if model_path is None:
            raise FileNotFoundError("SMPL-X model not found")

        model = smplx.create(
            model_path,
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to(device)
        return model
    except ImportError:
        raise RuntimeError("pip install smplx")


def render_silhouette_differentiable(vertices, faces, K, w2c, H, W, device="cuda"):
    """Render differentiable silhouette from mesh vertices.

    Uses PyTorch3D if available, falls back to simple z-buffer rasterization.

    Args:
        vertices: (1, V, 3) world-space vertices (torch tensor)
        faces: (F, 3) face indices
        K: (3, 3) numpy intrinsics
        w2c: (4, 4) numpy world-to-camera
        H, W: image dimensions

    Returns:
        (H, W) differentiable silhouette [0, 1]
    """
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            MeshRasterizer, RasterizationSettings,
            SoftSilhouetteShader,
            PerspectiveCameras,
        )

        # Convert w2c to PyTorch3D convention
        R = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=device).unsqueeze(0)
        T = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device).unsqueeze(0)

        focal = torch.tensor([[K[0, 0], K[1, 1]]], dtype=torch.float32, device=device)
        principal = torch.tensor([[K[0, 2], K[1, 2]]], dtype=torch.float32, device=device)

        cameras = PerspectiveCameras(
            R=R, T=T,
            focal_length=focal,
            principal_point=principal,
            image_size=((H, W),),
            in_ndc=False,
            device=device,
        )

        mesh = Meshes(
            verts=vertices,
            faces=torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0),
        )

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-5,
            faces_per_pixel=10,
        )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = SoftSilhouetteShader()
        fragments = rasterizer(mesh)
        silhouette = shader(fragments, mesh)[..., 3]  # (1, H, W)
        return silhouette[0]

    except ImportError:
        # Fallback: project vertices and create hard mask
        verts_np = vertices[0].detach().cpu().numpy()
        pts_2d = project_points(verts_np, K, w2c)
        mask = np.zeros((H, W), dtype=np.float32)
        pts_int = pts_2d.astype(np.int32)
        valid = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < W) & \
                (pts_int[:, 1] >= 0) & (pts_int[:, 1] < H)
        mask[pts_int[valid, 1], pts_int[valid, 0]] = 1.0
        return torch.tensor(mask, dtype=torch.float32, device=device)


def optimize_smplx_multiview(smplx_model, init_params, calib, scene_dir,
                              frame_idx, canonical, device="cuda",
                              n_iters=OPT_ITERS_STAGE_B):
    """Stage B: Multi-view SMPL-X joint optimization for a single frame.

    Args:
        smplx_model: SMPL-X body model
        init_params: dict with body_pose, betas, global_orient, transl from Stage A
        calib: calibration dict
        scene_dir: Path to scene
        frame_idx: int frame index
        canonical: canonical frame dict
        device: torch device
        n_iters: number of optimization iterations

    Returns:
        dict with optimized SMPL-X params + vertices + joints
    """
    w2c = calib["w2c"]
    K_all = calib["K"]
    registered = calib["registered"]
    cam_names = calib["cam_names"]
    reg_indices = np.where(registered)[0]

    per_view = scene_dir / "per_view"
    frame_str = f"{frame_idx:06d}"

    # Get image dimensions
    sample_path = per_view / cam_names[reg_indices[0]] / "images" / f"{frame_str}.png"
    if not sample_path.exists():
        return None
    sample_img = Image.open(sample_path)
    img_W, img_H = sample_img.size

    # Initialize SMPL-X parameters as torch tensors
    body_pose = torch.tensor(init_params["body_pose"], dtype=torch.float32,
                             device=device).reshape(1, -1)
    global_orient = torch.tensor(init_params["global_orient"], dtype=torch.float32,
                                 device=device).reshape(1, 3)
    betas = torch.tensor(init_params["betas"], dtype=torch.float32,
                         device=device).reshape(1, 10)
    transl = torch.tensor(init_params["transl"], dtype=torch.float32,
                          device=device).reshape(1, 3)
    lhand_pose = torch.zeros(1, 45, dtype=torch.float32, device=device)
    rhand_pose = torch.zeros(1, 45, dtype=torch.float32, device=device)

    # Make parameters optimizable
    body_pose.requires_grad_(True)
    global_orient.requires_grad_(True)
    transl.requires_grad_(True)
    lhand_pose.requires_grad_(True)
    rhand_pose.requires_grad_(True)
    betas.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {"params": [body_pose], "lr": OPT_LR_BODY_POSE},
        {"params": [global_orient], "lr": OPT_LR_GLOBAL_ORIENT},
        {"params": [transl], "lr": OPT_LR_TRANSL},
        {"params": [betas], "lr": OPT_LR_BETAS},
        {"params": [lhand_pose], "lr": OPT_LR_HAND},
        {"params": [rhand_pose], "lr": OPT_LR_HAND},
    ])

    # Load foreground masks for all registered cameras (precompute)
    fg_masks = {}
    for idx in reg_indices:
        cam_name = cam_names[idx]
        mask_path = per_view / cam_name / "pha" / f"{frame_str}.png"
        mask = load_foreground_mask(mask_path)
        if mask is not None:
            fg_masks[idx] = torch.tensor(mask, dtype=torch.float32, device=device)

    # Select subset of cameras for each optimization step (rotate through)
    cams_per_step = min(8, len(reg_indices))
    faces = smplx_model.faces.astype(np.int64)

    # Optimization loop
    for it in range(n_iters):
        optimizer.zero_grad()

        # Forward pass: SMPL-X
        output = smplx_model(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas,
            transl=transl,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            return_verts=True,
        )
        vertices = output.vertices  # (1, 10475, 3)
        joints = output.joints  # (1, J, 3)

        # Rotate camera subset
        start = (it * cams_per_step) % len(reg_indices)
        cam_subset = list(reg_indices[start:start + cams_per_step])
        if len(cam_subset) < cams_per_step:
            cam_subset += list(reg_indices[:cams_per_step - len(cam_subset)])

        loss_mask = torch.tensor(0.0, device=device)
        loss_reproj = torch.tensor(0.0, device=device)
        n_mask = 0

        for cam_idx in cam_subset:
            if cam_idx not in fg_masks:
                continue

            # Silhouette loss
            sil = render_silhouette_differentiable(
                vertices, faces, K_all[cam_idx], w2c[cam_idx], img_H, img_W, device
            )
            gt_mask = fg_masks[cam_idx]

            # BCE silhouette loss
            sil_clamped = sil.clamp(1e-6, 1.0 - 1e-6)
            bce = -(gt_mask * torch.log(sil_clamped) +
                    (1 - gt_mask) * torch.log(1 - sil_clamped))
            loss_mask = loss_mask + bce.mean()
            n_mask += 1

            # Joint-in-mask loss: penalize joints that project outside foreground
            # Project joints differentiably to 2D
            R_t = torch.tensor(w2c[cam_idx][:3, :3], dtype=torch.float32, device=device)
            t_t = torch.tensor(w2c[cam_idx][:3, 3], dtype=torch.float32, device=device)
            K_t = torch.tensor(K_all[cam_idx], dtype=torch.float32, device=device)
            joints_cam = (R_t @ joints[0].T + t_t.unsqueeze(1)).T  # (J, 3)
            proj = (K_t @ joints_cam.T).T
            proj_2d = proj[:, :2] / proj[:, 2:3].clamp(min=0.01)

            # Sample mask value at projected joint locations (differentiable via grid_sample)
            # Normalize to [-1, 1] for grid_sample
            proj_norm_x = proj_2d[:, 0] / img_W * 2 - 1
            proj_norm_y = proj_2d[:, 1] / img_H * 2 - 1
            grid = torch.stack([proj_norm_x, proj_norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, J, 2)
            mask_input = gt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            sampled = F.grid_sample(mask_input, grid, align_corners=True, mode="bilinear")  # (1, 1, 1, J)
            mask_at_joints = sampled.squeeze()  # (J,)
            # Penalize joints outside mask (mask_at_joints should be 1.0 for visible joints)
            in_frame = (joints_cam[:, 2] > 0.01)
            if in_frame.sum() > 5:
                loss_reproj = loss_reproj + (1 - mask_at_joints[in_frame]).mean()

        if n_mask > 0:
            loss_mask = loss_mask / n_mask
            loss_reproj = loss_reproj / n_mask

        # Priors
        loss_prior = OPT_LAMBDA_PRIOR * (body_pose ** 2).mean()
        loss_shape = OPT_LAMBDA_SHAPE * (betas ** 2).mean()
        loss_hand = OPT_LAMBDA_PRIOR * 0.1 * ((lhand_pose ** 2).mean() + (rhand_pose ** 2).mean())

        loss = (OPT_LAMBDA_MASK * loss_mask +
                OPT_LAMBDA_REPROJ * loss_reproj +
                loss_prior + loss_shape + loss_hand)

        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(f"    Iter {it}: loss={loss.item():.4f} "
                  f"mask={loss_mask.item():.4f} reproj={loss_reproj.item():.6f}")

    # Final forward pass
    with torch.no_grad():
        output = smplx_model(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas,
            transl=transl,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            return_verts=True,
        )

    return {
        "body_pose": body_pose.detach().cpu().numpy().flatten(),
        "global_orient": global_orient.detach().cpu().numpy().flatten(),
        "betas": betas.detach().cpu().numpy().flatten(),
        "transl": transl.detach().cpu().numpy().flatten(),
        "lhand_pose": lhand_pose.detach().cpu().numpy().flatten(),
        "rhand_pose": rhand_pose.detach().cpu().numpy().flatten(),
        "vertices": output.vertices[0].detach().cpu().numpy(),
        "joints": output.joints[0].detach().cpu().numpy(),
    }


# ─── Semantic Label Assignment ───────────────────────────────────────────────

def assign_vertex_semantics(vertices, calib, scene_dir, frame_idx):
    """Assign per-vertex semantic labels via multi-view majority vote.

    Args:
        vertices: (V, 3) world-space vertices
        calib: calibration dict
        scene_dir: Path to scene
        frame_idx: frame index

    Returns:
        (V,) uint8 semantic labels (0=bg, 1=skin, 2=hair, 3=clothing, 4=shoes)
    """
    w2c = calib["w2c"]
    K_all = calib["K"]
    registered = calib["registered"]
    cam_names = calib["cam_names"]
    reg_indices = np.where(registered)[0]

    per_view = scene_dir / "per_view"
    frame_str = f"{frame_idx:06d}"

    V = vertices.shape[0]
    # Vote counts per vertex per class (classes 0-4)
    votes = np.zeros((V, 5), dtype=np.int32)

    for cam_idx in reg_indices:
        cam_name = cam_names[cam_idx]
        sem_path = per_view / cam_name / "semantic" / f"{frame_str}.png"
        sem = load_semantic_mask(sem_path)
        if sem is None:
            continue

        H, W = sem.shape

        # Project all vertices to this camera
        pts_2d = project_points(vertices, K_all[cam_idx], w2c[cam_idx])

        # Check visibility (z > 0 in camera space)
        R = w2c[cam_idx][:3, :3]
        t = w2c[cam_idx][:3, 3]
        pts_cam = (R @ vertices.T + t[:, None]).T
        visible = pts_cam[:, 2] > 0.01

        # Check within image bounds
        px = pts_2d[:, 0].astype(np.int32)
        py = pts_2d[:, 1].astype(np.int32)
        in_bounds = visible & (px >= 0) & (px < W) & (py >= 0) & (py < H)

        # Sample semantic labels
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) == 0:
            continue

        labels = sem[py[valid_idx], px[valid_idx]]
        for i, label in zip(valid_idx, labels):
            if 0 <= label <= 4:
                votes[i, label] += 1

    # Majority vote (excluding background votes for foreground vertices)
    semantic_labels = np.zeros(V, dtype=np.uint8)
    for i in range(V):
        fg_votes = votes[i, 1:]  # classes 1-4
        if fg_votes.sum() > 0:
            semantic_labels[i] = np.argmax(fg_votes) + 1
        # else: stays 0 (background)

    return semantic_labels


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--smplx_model_path", type=str, default=None)
    parser.add_argument("--opt_iters", type=int, default=OPT_ITERS_STAGE_B)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    device = args.device

    # ── Load calibration + canonical frame ────────────────────────────────────
    print("Loading calibration and canonical frame...")
    calib = dict(np.load(str(scene_dir / "calibration.npz"), allow_pickle=True))
    canonical = dict(np.load(str(scene_dir / "canonical_frame.npz"), allow_pickle=True))

    cam_names = calib["cam_names"]
    registered = calib["registered"]
    reg_indices = np.where(registered)[0]
    print(f"  {len(reg_indices)} registered cameras")

    # ── Determine frame range ─────────────────────────────────────────────────
    per_view = scene_dir / "per_view"
    sample_cam = cam_names[reg_indices[0]]
    frame_files = sorted((per_view / sample_cam / "images").glob("*.png"))
    n_total_frames = len(frame_files)

    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else n_total_frames
    end = min(end, n_total_frames)
    frame_indices = list(range(start, end, args.frame_stride))
    print(f"  Processing {len(frame_indices)} frames ({start} to {end}, stride {args.frame_stride})")

    # ── Load SMPL-X model ─────────────────────────────────────────────────────
    print("Loading SMPL-X model...")
    smplx_model = load_smplx_model(args.smplx_model_path, device)
    skinning_weights = smplx_model.lbs_weights.detach().cpu().numpy()  # (10475, 55)
    print("  SMPL-X loaded.")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = scene_dir / "smplx_params"
    out_dir.mkdir(exist_ok=True)

    # ── Stage A: HMR2.0 initialization (frame 0 only) ────────────────────────
    print("\n=== Stage A: HMR2.0 Initialization ===")
    init_params = run_stage_a(scene_dir, canonical, calib, device)

    # Shared betas across all frames (initialized from HMR2.0, refined in first few frames)
    shared_betas = init_params["betas"].copy()
    prev_params = None

    # ── Process each frame ────────────────────────────────────────────────────
    for f_count, f_idx in enumerate(frame_indices):
        frame_str = f"{f_idx:06d}"
        out_path = out_dir / f"frame_{frame_str}.npz"

        if args.skip_existing and out_path.exists():
            print(f"\nFrame {f_idx}: skipping (exists)")
            continue

        print(f"\n=== Frame {f_idx} ({f_count + 1}/{len(frame_indices)}) ===")

        # Update translation from canonical per-frame tracking
        frame_transl = canonical["origin"] + canonical["per_frame_translation"][f_idx]

        # Initialize from previous frame (temporal coherence) or Stage A
        if prev_params is not None:
            frame_init = {
                "body_pose": prev_params["body_pose"].copy(),
                "global_orient": prev_params["global_orient"].copy(),
                "betas": shared_betas.copy(),
                "transl": frame_transl,
            }
        else:
            frame_init = {
                "body_pose": init_params["body_pose"].copy(),
                "global_orient": init_params["global_orient"].copy(),
                "betas": shared_betas.copy(),
                "transl": frame_transl,
            }

        # ── Stage B: Multi-view optimization ──────────────────────────────────
        print(f"  Stage B: Multi-view optimization ({args.opt_iters} iters)...")
        result = optimize_smplx_multiview(
            smplx_model, frame_init, calib, scene_dir, f_idx, canonical, device,
            n_iters=args.opt_iters,
        )

        if result is None:
            print(f"  Frame {f_idx}: optimization failed!")
            continue

        # Lock betas after first 10 frames
        if f_count < 10:
            shared_betas = result["betas"].copy()

        # ── Semantic label assignment ─────────────────────────────────────────
        print(f"  Assigning semantic labels...")
        semantic_labels = assign_vertex_semantics(
            result["vertices"], calib, scene_dir, f_idx
        )

        # Label distribution
        unique, counts = np.unique(semantic_labels[semantic_labels > 0], return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  Semantic labels: {dist}")

        # ── Save ──────────────────────────────────────────────────────────────
        np.savez(
            str(out_path),
            body_pose=result["body_pose"],
            global_orient=result["global_orient"],
            betas=result["betas"],
            transl=result["transl"],
            lhand_pose=result["lhand_pose"],
            rhand_pose=result["rhand_pose"],
            vertices=result["vertices"],
            joints=result["joints"],
            skinning_weights=skinning_weights,
            semantic_labels=semantic_labels,
        )

        # Update temporal tracking (previous frame init for next frame)
        prev_params = result

    print(f"\nDone. Saved SMPL-X params to {out_dir}/")


if __name__ == "__main__":
    main()
