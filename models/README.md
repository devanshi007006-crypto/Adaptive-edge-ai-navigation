# Models Directory

This directory stores deep learning weights and model assets used by the perception layer.

---

## 1. Required Model Checkpoints

| Model Type | Canonical Subdirectory | Expected File Name | Status | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection** | `models/detector/` | `yolo11n.pt` (or `yolov8n.pt`) | **MISSING** | 2D bounding-box localization & classification across 80 COCO classes |
| **Monocular Depth** | `models/depth/` | `depth_anything_v2_vits.pth` | **MISSING** | Full-frame monocular relative depth map estimation |

---

## 2. Model Download Instructions

When preparing models for Phase 1:

### Object Detector (Ultralytics YOLO)
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Automatically downloads official weights
```

### Depth Anything V2 (Small - ViT-S)
Official weights available from the Depth Anything V2 GitHub repository:
- URL: `https://github.com/DepthAnything/Depth-Anything-V2`
- Checkpoint: `depth_anything_v2_vits.pth` (24.8M parameters)
- Target Path: `models/depth/depth_anything_v2_vits.pth`

---

## 3. Policy Regarding Large Files
- Model weight files (`*.pt`, `*.pth`, `*.onnx`, `*.engine`) must **never be committed to Git**.
- `.gitignore` must ignore binary checkpoint weights while tracking directory structure via `.gitkeep`.
