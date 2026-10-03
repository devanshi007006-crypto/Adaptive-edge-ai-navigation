# Adaptive Navigation System for Visually Impaired Users

An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction.

## System Architecture

```
CAMERA
  │
  ▼
PERCEPTION (Detector + Tracker + Depth + Preprocessing)
  │
  ▼
TEMPORAL (History + Velocity + Smoothing + Camera Motion)
  │
  ▼
RISK (TTC + Features + Score + State Machine + Uncertainty)
  │
  ▼
NAVIGATION (Zones + Free Space + Decision Engine)
  │
  ▼
FEEDBACK (Warning Manager + Message Generator + Offline Speech)
  │
  ▼
SPEAKER (Audio Announcement)
  │
  ▼
ENVIRONMENT CHANGES & RE-EVALUATION
```

## Directory Structure

```
adaptive_navigation/
├── main.py                    # Pipeline orchestrator
├── config.yaml                # Centralized configuration & thresholds
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
│
├── perception/                # Vision and sensor processing
│   ├── __init__.py
│   ├── detector.py            # YOLO object detector interface
│   ├── tracker.py             # BoT-SORT tracker interface
│   ├── depth.py               # Depth Anything V2 interface
│   └── preprocessing.py       # Frame validation & transforms
│
├── temporal/                  # Multi-frame reasoning
│   ├── __init__.py
│   ├── history.py             # Rolling object track history buffer
│   ├── velocity.py            # Motion & approach state estimation
│   ├── smoothing.py           # Temporal filter & smoothing
│   └── camera_motion.py       # Ego-motion compensation interface
│
├── risk/                      # Safety assessment
│   ├── __init__.py
│   ├── ttc.py                 # Time-to-Collision calculation
│   ├── features.py            # Risk feature extraction
│   ├── score.py               # Analytical risk scoring
│   ├── state_machine.py       # Warning priority state transitions
│   └── uncertainty.py         # Track & depth uncertainty model
│
├── navigation/                # Path planning & zone analysis
│   ├── __init__.py
│   ├── zones.py               # Left / Center / Right zone partitioning
│   ├── free_space.py          # Free space & clearance estimation
│   └── decision.py            # Directional guidance decision engine
│
├── feedback/                  # Human interaction & speech
│   ├── __init__.py
│   ├── speech.py              # Offline pyttsx3 speech interface
│   ├── message_generator.py   # Context-aware warning phrasing
│   └── warning_manager.py     # Cooldown & escalation priority manager
│
├── evaluation/                # Performance tracking & logging
│   ├── __init__.py
│   ├── logger.py              # Structured CSV & frame logger
│   ├── metrics.py             # Latency, FPS, & decision metrics
│   └── report.py              # Report generation
│
├── models/                    # Model checkpoints & assets
│   ├── detector/
│   └── depth/
│
└── data/                      # Datasets, benchmarks, and outputs
    ├── raw/
    ├── processed/
    ├── annotations/
    └── results/
```

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Validation
```bash
python main.py --dry-run
```


## Pipeline Implementation Status
- **Step 1**: Architectural Skeleton & Interfaces (Completed)
- **Step 2**: Camera / Frame Acquisition Layer (Completed)
- **Step 3**: YOLO Object Detection (Completed - YOLO11n)
- **Step 4**: BoT-SORT Object Tracking (Completed - Ultralytics BoT-SORT with persistent Track IDs)
- **Step 5**: Depth Estimation (Completed - Depth Anything V2 ViT-S relative depth & object-level depth estimation)
- **Step 6**: Temporal History Buffer (Completed - Rolling per-track observation buffers with bounded memory, TTL cleanup, and depth preservation)
- **Step 7**: Velocity, Approach Estimation & Temporal Smoothing (Completed - Image-plane velocity, relative depth rate, EMA smoothing, and approach classification)
- **Step 8**: Camera Motion Compensation (Completed - Lucas-Kanade sparse optical flow with RANSAC outlier rejection, foreground obstacle masking, and compensated velocity estimation)
- **Step 9**: Time-to-Collision (TTC) Estimation (Completed - Analytical closing speed calculation, explicit metric vs relative depth distinction, out-of-range clamping, and state reporting)

### Running the Integrated Pipeline (Step 5)
`ash
# Live camera with YOLO + BoT-SORT + Depth Anything V2 visualization
python adaptive_navigation/main.py --cam 0

# Headless mode for benchmark / testing
python adaptive_navigation/main.py --cam 0 --max-frames 10 --headless
`

### Depth Representation Note
- Model: depth_anything_v2_vits (models/depth/depth_anything_v2_vits.pth)
- Output: **Relative Depth** (dimensionless affine-invariant depth, NOT metric meters)
- Object Depth: Robust median calculation over clamped bounding box interior
- Reliability Levels: HIGH, MEDIUM, LOW, INVALID based on pixel validity and bounding box size
