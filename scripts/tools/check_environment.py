"""
Environment Verification & Diagnostic Tool.
Checks Python version, installed dependencies, CUDA/GPU availability,
and presence of required neural model weights.
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def check_environment():
    print("=" * 70)
    print("ADAPTIVE EDGE-AI NAVIGATION — ENVIRONMENT DIAGNOSTIC")
    print("=" * 70)
    print(f"Python Version:    {sys.version.split()[0]} ({sys.executable})")

    # 1. PyTorch & Acceleration
    try:
        import torch
        print(f"PyTorch Version:   {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available:    {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Device:       {torch.cuda.get_device_name(0)}")
            print(f"Device Count:      {torch.cuda.device_count()}")
        else:
            print("CUDA Note:         Running on CPU. GPU acceleration is unavailable.")
    except ImportError:
        print("ERROR: PyTorch is NOT installed.")

    # 2. OpenCV
    try:
        import cv2
        print(f"OpenCV Version:    {cv2.__version__}")
    except ImportError:
        print("ERROR: OpenCV is NOT installed.")

    # 3. NumPy & SciPy
    try:
        import numpy as np
        import scipy
        print(f"NumPy / SciPy:     {np.__version__} / {scipy.__version__}")
    except ImportError as e:
        print(f"ERROR: {e}")

    # 4. Ultralytics (YOLO)
    try:
        import ultralytics
        print(f"Ultralytics:       {ultralytics.__version__}")
    except ImportError:
        print("WARNING: ultralytics is NOT installed in this environment.")
        print("         Install via: pip install ultralytics>=8.1.0")

    # 5. Audio / TTS
    try:
        import pyttsx3
        print(f"pyttsx3 (TTS):     Installed ({pyttsx3.__name__})")
    except ImportError:
        print("WARNING: pyttsx3 is NOT installed.")

    # 6. Model Weight Check
    print("-" * 70)
    print("MODEL CHECKPOINTS CHECK:")
    models_dir = _repo_root / "models"
    detector_model = models_dir / "detector" / "yolo11n.pt"
    depth_model = models_dir / "depth" / "depth_anything_v2_vits.pth"

    print(f"  Detector (yolo11n.pt):              {'EXISTS' if detector_model.exists() else 'MISSING'}")
    print(f"  Depth (depth_anything_v2_vits.pth): {'EXISTS' if depth_model.exists() else 'MISSING'}")

    print("=" * 70)


if __name__ == "__main__":
    check_environment()
