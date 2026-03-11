"""Hyperparameters for wavelet volume rendering."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    scene: str = "lego"
    data_dir: str = "data/nerf_synthetic"
    output_dir: str = "output"
    train_resolution: int = 800
    white_background: bool = True  # NeRF Synthetic uses white bg

    # Wavelet volume
    # decomp_levels = number of DWT decomposition levels
    # target_resolution = base_resolution * 2^decomp_levels
    # 3 levels: 32³ → 256³ (~1.8 GB params fp32, fits any GPU)
    # 4 levels: 32³ → 512³ (~14 GB params fp32, needs H100)
    decomp_levels: int = 3
    base_resolution: int = 32  # coarsest level (approximation)
    wavelet: str = "bior4.4"  # CDF 9/7 (JPEG 2000)
    num_channels: int = 28  # 1 density + 27 SH (degree 2, 3 colors)
    sh_degree: int = 2  # 9 SH basis functions

    # Scene bounds
    scene_bound: float = 1.5  # scene fits in [-bound, bound]³
    near: float = 2.0
    far: float = 6.0

    # Volume rendering
    coarse_samples: int = 64
    fine_samples: int = 64

    # Training
    iterations: int = 50000
    batch_rays: int = 8192
    lr: float = 5e-3
    lr_min: float = 1e-5

    # Progressive training: iteration at which each detail level activates.
    # If empty, auto-generated as evenly spaced across iterations.
    # Length must equal decomp_levels.
    progressive_starts: list = field(default_factory=lambda: [])

    # Tiled reconstruction (for decomp_levels > 4)
    tile_size: int = 64  # output tile size at finest level
    base_level: int = 2  # levels 0..base_level are dense, rest are sparse/tiled

    # Per-level channel counts (None = all levels use num_channels).
    # Example for 6 levels: [28, 28, 28, 16, 8, 4]
    # Fine levels only need density + base color, not full SH.
    channels_per_level: list | None = None

    # Occupancy estimation (for sparse coefficient allocation)
    occupancy_threshold: float = 0.1
    occupancy_dilate: int = 3
    occupancy_grid_resolution: int = 128
    coarse_pretrain_iters: int = 20000  # iters to train coarse model for occupancy

    # Direct grid training
    training_mode: str = "wavelet"  # "wavelet" | "direct" | "direct_then_tiled"
    direct_res_schedule: list = field(default_factory=lambda: [
        (128, 20000), (256, 40000), (512, 60000),
    ])
    direct_lr: float = 0.02  # higher LR for direct grid (Plenoxels-style)

    # svox2 conversion
    svox2_ckpt: str = ""  # path to svox2 .npz checkpoint for conversion
    dwt_batch_channels: int = 4  # channels per DWT batch (memory vs speed)

    # Frequency-matched training
    # Stages: list of (image_res, volume_res, iterations) tuples
    fm_stages: list | None = None
    fm_wavelet: str = "bior4.4"  # wavelet for upsampling transitions
    fm_upsample_mode: str = "wavelet"  # "wavelet" or "trilinear"
    fm_prune_every: int = 5000  # voxel pruning interval
    fm_prune_threshold: float = 0.01  # density threshold for pruning

    # Regularization
    lambda_sparse: float = 1e-3  # sparsity loss weight on density
    lambda_tv: float = 1e-3      # TV loss weight (direct grid only)
    tv_decay_factor: float = 0.01  # TV decays to lambda_tv * this over training

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "bfloat16"

    # Logging
    val_every: int = 5000
    save_every: int = 10000
    log_every: int = 100

    # Device
    device: str = "cuda"

    def __post_init__(self):
        if not self.progressive_starts:
            # Auto-generate: first level from start, rest evenly spaced
            self.progressive_starts = [
                int(i * self.iterations / (self.decomp_levels + 1))
                for i in range(self.decomp_levels)
            ]

    @property
    def target_resolution(self) -> int:
        """Finest voxel grid resolution."""
        return self.base_resolution * (2 ** self.decomp_levels)

    @property
    def num_sh_coeffs(self) -> int:
        """Number of SH basis functions."""
        return (self.sh_degree + 1) ** 2
