# RunPod Setup Log

## Pod Info
- **Name**: fine_chocolate_chameleon
- **ID**: 12wq996g8k3an2
- **GPU**: H100 SXM 80GB (1x)
- **SSH**: `ssh runpod` (root@216.243.220.223 -p 17852)
- **Template**: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- **Volume**: 150 GB at /workspace
- **Cost**: $2.72/hr

---

## Setup Commands

### 1. Environment check
```
nvidia-smi → Driver 580.126.09, NVIDIA H100 80GB HBM3, 81559 MiB
python3 → PyTorch 2.4.1+cu124, CUDA available: True
nvcc → Build cuda_12.4.r12.4/compiler.34097967_0
```

### 2. Directory structure
```
mkdir -p /workspace/Data/Neu3D /workspace/4DGS
```

### 3. Install gsplat + tools
```
pip install gsplat → gsplat-1.5.3 installed
apt-get install unzip gh → OK
```

### 4. Download Neu3D dataset (parallel curl)
```
Downloading: coffee_martini.zip, cook_spinach.zip, cut_roasted_beef.zip,
flame_steak.zip, sear_steak.zip, flame_salmon_1_split.{z01,z02,z03,zip}
Status: COMPLETE (~2 min on datacenter bandwidth)
```

### 5. Extract and organize Neu3D
```
unzip → 5 simple scenes OK
7z x → flame_salmon_1 split archive OK
Flattened nested dirs

/workspace/Data/Neu3D/
├── coffee_martini/   (1.2G, 18 cams + poses_bounds.npy)
├── cook_spinach/     (1.2G)
├── cut_roasted_beef/ (1.1G)
├── flame_salmon_1/   (4.7G)
├── flame_steak/      (1.2G)
└── sear_steak/       (1.2G)

Each scene: cam00-cam20 .mp4 files (18 cameras, some skipped) + poses_bounds.npy
```

### 6. Install gsplat + viser + nerfstudio
```
pip install gsplat → gsplat-1.5.3
pip install nerfstudio → 1.1.5 (pulled torch 2.10.0, broke CUDA compat)
pip install torch==2.4.1 torchvision==0.19.1 --index-url cu124 → fixed
Final: PyTorch 2.4.1+cu124, gsplat OK, viser OK
```

### Viewer setup
```
Live viewer: viser serves on port 7007
Access via: ssh -L 7007:localhost:7007 runpod
Then open: http://localhost:7007

Post-training: scp .ply to Mac → open superspl.at → drag & drop
```

