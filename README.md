# 3D LiDAR Semantic Segmentation

A deep learning system for **real-time 3D LiDAR point cloud semantic segmentation** using a novel fusion of **Point Transformer V3 (PTv3)** and **DINOv2** vision features.

## Results

| Model | mIoU | allAcc | mAcc |
|-------|------|--------|------|
| PTv3 Only | 0.6500 | - | - |
| DirectFusion | 0.6768 | - | - |
| ProjectionFusion (10ep) | 0.6896 | - | - |
| **25ep + CosineAnnealingLR** | **0.6913** | **92.06%** | **73.82%** |

## Architecture

- **PTv3 (46.16M params)** — processes raw LiDAR point clouds → 64-dim features per voxel
- **DINOv2 ViT-Small/14 (22.06M params)** — processes 6 nuScenes cameras → 384-dim patch features
- **Point→Image Projection** — projects each LiDAR point onto camera pixels to retrieve visual features
- **DINOPT Fusion MLP** — concatenates PTv3 + DINOv2 features → 128-dim → 16-class segmentation

## Per-Class IoU

| Class | IoU |
|-------|-----|
| bus | 95.69% |
| car | 94.68% |
| truck | 90.24% |
| driveable_surface | 89.40% |
| vegetation | 90.86% |
| terrain | 80.38% |
| pedestrian | 79.74% |
| motorcycle | 66.58% |
| sidewalk | 36.92% |
| bicycle | 20.97% |

## Demo

Interactive 3D visualization with:
- Real-time semantic segmentation
- 4 visualization modes: Segmentation / Height / Intensity / Confidence
- Ground truth vs prediction comparison
- 6 camera views (DINOv2 input)
- Per-class IoU metrics
- Model architecture diagram

## Dataset

nuScenes mini — 404 samples across train/val/test splits

## Setup

\`\`\`bash
git clone https://github.com/Vyomkesh13/3D-Lidar-Segmentation.git
cd 3D-Lidar-Segmentation

pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 mmsegmentation==0.30.0 spconv-cu118
pip install nuscenes-devkit pyquaternion open3d timm
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install "numpy<2" SharedArray fastapi uvicorn
\`\`\`

## Run Web Demo

\`\`\`bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
\`\`\`

## Project Structure

\`\`\`
3D-Lidar-Segmentation/
├── src/
│   ├── DINOPT.py          # Fusion model
│   ├── model.py           # PTv3 backbone
│   ├── dataset_nuscenes.py
│   ├── dataset_joint_features.py
│   ├── fusion_utils.py
│   └── ...
├── app/
│   ├── main.py            # FastAPI backend
│   └── static/            # React frontend
├── weights/               # Trained model weights
└── dinopt.ipynb           # Training notebook
\`\`\`

## Key Innovations

1. **25 epochs + CosineAnnealingLR** — Extended training with cosine LR schedule improved mIoU from 0.6896 → 0.6913
2. **Point→Image Projection Fusion** — Each LiDAR point retrieves visual features from corresponding camera patch
3. **Multi-modal fusion** — Combines geometric (PTv3) + visual (DINOv2) understanding
