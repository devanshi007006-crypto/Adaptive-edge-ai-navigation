# Adaptive Multimodal Edge-AI Navigation System for Visually Impaired Mobility Assistance

> **An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11-3.14](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu130-red.svg)](https://pytorch.org/)
[![TensorRT 11.3](https://img.shields.io/badge/TensorRT-11.3.0-green.svg)](https://developer.nvidia.com/tensorrt)
[![Live Throughput: 14.9 FPS](https://img.shields.io/badge/Live_Throughput-14.93_FPS-brightgreen.svg)]()
[![p50 Latency: 50.1 ms](https://img.shields.io/badge/p50_Latency-50.13_ms-blue.svg)]()
[![Status: Live Demo Capable](https://img.shields.io/badge/Status-Research_Prototype_Live_Demo_Capable-purple.svg)]()

---

## Quick Start (Run Final Research Prototype)

```powershell
# 1. Activate project environment
.\.venv\Scripts\activate

# 2. Launch Canonical Layer 1 + Layer 2 Research Prototype (YOLO26n Primary)
python scripts/run/run_final_prototype.py

# 3. Launch Live Demo Control Center GUI (S01–S06 Scenarios)
python scripts/run/run_demo_control_center.py
```

---

## Current Status

* **Status**: **Research Prototype — Live Webcam Demonstration & Judge Demo Capable**
* **Target Environment**: Laptop Integrated Webcam ($640 \times 480$ @ 30 FPS) + NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`).
* **Primary Detector**: **Ultralytics YOLO26n** (`models/detector/yolo26n.pt`) with **9.02 ms** detection latency. *(Research baseline YOLO11n remains available).*
* **Live Performance**: **17.68 FPS** mean throughput (**50.13 ms** p50 latency) with native TensorRT FP16 monocular depth inference (`models/deployment/depth_anything_v2_vits_fp16.engine`) and Risk-Aware Adaptive Computation Controller.
* **Evidence Package**: Poster evidence report, CSV tables, visualizations, and judge demo script compiled in [`validation/results/final_research_evidence/`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/final_research_evidence/).
* **Safety Boundary**: Strictly for **controlled, open-eye, supervised technical demonstrations**. Not certified as a medical device or blind mobility aid.

---

## Documentation Map

- [`PROJECT_INDEX.md`](PROJECT_INDEX.md) — Master project database, sitemap, metrics provenance, and complete changelog.
- [`validation/results/final_research_evidence/reports/poster_evidence_report.md`](validation/results/final_research_evidence/reports/poster_evidence_report.md) — Canonical final poster evidence report.
- [`validation/results/final_research_evidence/judge_demo/judge_demo_script.md`](validation/results/final_research_evidence/judge_demo/judge_demo_script.md) — 2–3 minute Research Conclave judge demonstration walkthrough.
- [`validation/results/final_research_evidence/reports/poster_claims.md`](validation/results/final_research_evidence/reports/poster_claims.md) — Safe, qualified, and excluded scientific claims guidance.
- [`docs/FILE_MANIFEST.md`](docs/FILE_MANIFEST.md) — Detailed 2nd-level map defining ownership, purpose, status, and references for all repository files.

---

## 1. Overview

Visually impaired individuals face continuous collision threats during independent mobility. Traditional electronic travel aids (ETAs) rely on basic ultrasonic or infrared range sensors that emit continuous acoustic beeps for every nearby obstacle surface. This causes severe cognitive sensory fatigue, fails to discriminate approaching threats from receding obstacles, and lacks semantic scene understanding.

This project implements a monocular edge-AI navigation architecture that combines deep learning perception models with dynamic physical Time-to-Collision (TTC), camera ego-motion compensation, and spatial walking corridor analysis. Rather than alerting on every static surface, the system tracks object trajectories, absorbs walking gait sway using optical flow, estimates Closing Velocity ($v_{\text{rel}}$) and scale-invariant Time-to-Collision ($\tau = d / \dot{d}$), and dispatches context-aware, non-blocking spoken navigation advisories (e.g., *"Caution, approaching pedestrian, step right"*).

---

## 2. Research Objectives

1. **Ego-Motion Compensated Trajectory Prediction**: Discriminate closing collision threats from receding or stationary objects using background optical flow expansion divergence ($\bar{\gamma}_{\text{bg}}$) and multi-frame least-squares regression.
2. **False Alert Elimination**: Suppress alert flickering and cognitive fatigue through a 2-frame temporal hysteresis state machine and central $\pm 0.6\text{m}$ lateral walking corridor partitioning.
3. **Scale-Invariant Optical Divergence TTC**: Derivation of physical collision timing ($\tau = d / \dot{d}$) directly from monocular relative disparity derivatives without depending on unstable absolute distance calibration.
4. **Real-Time Edge Deployment**: Sustain real-time throughput ($>14\text{ FPS}$, $<55\text{ ms}$ latency) on consumer mobile GPU hardware using native TensorRT FP16 compilation and 2:1 subsampled depth inference.

---

## 3. System Architecture

The unified feed-forward perception, risk, and navigation decision loop executes 15 sequential stages:

```
[Monocular RGB Webcam / Recorded Video Stream]
                   │
                   ▼
 1. Frame Ingestion & Preprocessing (640x480 @ 30 FPS)
                   │
                   ▼
 2. Object Detection (Ultralytics YOLO11n, PyTorch / ONNX)
                   │
                   ▼
 3. Multi-Object Tracking (BoT-SORT Appearance + Kalman Filtering)
                   │
                   ▼
 4. Monocular Depth Estimation (Depth Anything V2 ViT-S, TensorRT FP16)
                   │
                   ▼
 5. Object Depth Association & Rolling History (Deque maxlen=25)
                   │
                   ▼
 6. Motion & Range Rate Estimation (Multi-Frame Slope Regression)
                   │
                   ▼
 7. Camera Ego-Motion Compensation (Sparse Lucas-Kanade Flow + Radial Divergence)
                   │
                   ▼
 8. Scale-Invariant Time-to-Collision (TTC = d / v_closing)
                   │
                   ▼
 9. Multi-Factor Risk Assessment Engine (TTC, Distance, Motion, Path, Class)
                   │
                   ▼
10. Uncertainty & Reliability Calibration (Evidence Coverage Scoring)
                   │
                   ▼
11. Temporal Risk Stabilization & Warning Machine (2-Frame Hysteresis)
                   │
                   ▼
12. Explainable Spoken Warning Generator (Concise Alert Formatting)
                   │
                   ▼
13. Spatial Corridor & Navigation Decision (CONTINUE / CAUTION / AVOID_LEFT / AVOID_RIGHT / STOP)
                   │
                   ▼
14. Non-Blocking Spoken Audio Guidance (pyttsx3 Priority Queue Engine)
                   │
                   ▼
15. Real-Time Visual HUD Overlay & Event Telemetry Logger (JSON / CSV Logging)
```

---

## 4. Key Technical Components

| Component / Subsystem | Primary Architecture / Implementation | Purpose & Operational Role |
| :--- | :--- | :--- |
| **Object Detection** | Ultralytics YOLO11n (`yolo11n.pt`) | 2D obstacle localization and 80-class COCO classification |
| **Multi-Object Tracking** | BoT-SORT Adapter | Multi-frame track continuity and ID association (buffer=25) |
| **Monocular Depth** | Depth Anything V2 ViT-S (`vits`) | Monocular inverse depth map estimation (TensorRT FP16 / PyTorch) |
| **Ego-Motion Compensation** | Sparse Lucas-Kanade Optical Flow + RANSAC | Radial divergence ($\bar{\gamma}_{\text{bg}}$) estimation absorbing gait sway |
| **Collision Timing (TTC)** | Inverse Disparity Divergence ($\tau = d / \dot{d}$) | Scale-invariant kinematic Time-to-Collision estimation |
| **Dynamic Risk Engine** | 5-Feature Fused Score | Fuses TTC, normalized depth, approach state, path overlap, and class |
| **Warning State Machine** | 2-Frame Hysteresis State Machine | Escalation confirmation and de-escalation dampening |
| **Navigation Engine** | Spatial Corridor Partitioning ($\pm 0.6\text{m}$) | Lateral free-space analysis & safe steering guidance |
| **Spoken Audio TTS** | Asynchronous `pyttsx3` Priority Queue | Non-blocking spoken audio advisories with 2.0s repetition suppression |
| **Deployment Acceleration** | Native TensorRT 11.3 FP16 Engine | Replaces PyTorch depth backbone (accelerates latency to 50.1 ms) |

---

## 5. Research & Validation Progression

### Phase 1: Real Pipeline Bring-Up & GPU Acceleration
- **Objective**: Transition from synthetic simulation to real PyTorch inference on NVIDIA RTX 4050 GPU (`cuda:0`).
- **Outcome**: Achieved **10.3× depth speedup** (728 ms $\rightarrow$ 70.5 ms) and **7.6× pipeline speedup** (772 ms $\rightarrow$ 101 ms).

### Phase 2: Controlled Video Behavioral Validation
- **Phase 2A Baseline**: Validated across 5 recorded indoor pedestrian scenarios (5,235 frames).
- **Phase 2B Fixes**: Implemented indoor class filtering policy (suppressing architectural COCO clutter) and multi-frame least-squares regression slope in `motion.py`. Completely eliminated hallway false alarms in `S01` (warnings dropped 793 $\rightarrow$ 0) and reduced receding approach spikes by 64%.
- **Phase 2C Throughput & Ego-Motion**: Implemented 2:1 depth inference cadence (+68.7% throughput speedup to 13.4 FPS) and background radial flow divergence compensation ($\bar{\gamma}_{\text{bg}}$).

### Phase 3: External Egocentric Validation (HEADS-UP)
- **Phase 3A Exploratory**: Evaluated 250 unconstrained egocentric frames across 3 target episodes from EPFL HEADS-UP benchmark.
- **Phase 3B & 3B.1 Timestamp-Aware Revalidation**: Corrected constant-30 FPS video timing artifacts by integrating original source frame timestamps ($dt = (id_k - id_{k-1})/30$). Removed 74 constant-FPS false critical alerts under unchanged thresholds.

### Phase 4: Metric Depth Calibration & TensorRT FP16 Optimization
- **Phase 4A Depth Calibration**: Fitted inverse disparity affine metric scale model ($d_{\text{metric}} = 1 / (a \cdot d_{\text{rel}} + b)$) on calibration dataset.
- **Phase 4B External Evaluation**: Verified metric depth zone agreement (94.5% classification accuracy for NEAR/MID/FAR zones).
- **Phase 4C & 4C.1 Native TensorRT FP16**: Built native TensorRT 11.3 FP16 engine (`depth_anything_v2_vits_fp16.engine`), elevating live depth latency to **33.8 ms** and end-to-end pipeline throughput to **14.93 FPS** (50.13 ms p50 latency).

### Phase 5: Live Webcam Prototype & Bug Audit
- **Bug Audit**: Isolated and resolved persistent global `CAUTION + STOP` false trigger by filtering spatial occupancy accumulation to active warning/critical threats in `navigation_decision.py` and aligning candidate caution threshold to 0.50 in `state_machine.py`.
- **400-Frame Baseline Revalidation**: Executed 400-frame static room baseline: **357/400 frames `CONTINUE` (89.25%)**, **43/400 frames `CAUTION` (10.75%)**, **0 false STOP calls (0.0%)**, **0 false steering calls (0.0%)**, and **0 spoken audio alerts**.

---

## 6. Current Measured Performance Results

### Offline & Live Benchmark Summary

| Evaluation Benchmark | Evaluated Sample Size | Target Metric | Measured Result | Provenance Reference Report |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2A Baseline** | 5 videos (5,235 frames) | PyTorch GPU Throughput | **8.00 FPS** (120.5 ms p50) | [`phase2a_video_validation_report.md`](validation/results/phase2a_video_validation_report.md) |
| **Phase 2B Revalidation** | 5 videos (5,235 frames) | False Warning Drop | **-100% false alarms in S01** | [`phase2b_behavioral_validation_report.md`](validation/results/phase2b_behavioral_validation_report.md) |
| **Phase 2C Cadence** | 5 videos (5,235 frames) | 2:1 Depth Cadence Speedup | **13.38 FPS** (+68.7% speedup) | [`phase2c_validation_report.md`](validation/results/phase2c_validation_report.md) |
| **Phase 3B.1 Timing** | 8 ranges (852 valid frames) | Temporal Timing Correction | **74 false criticals removed** | [`phase3b1_temporal_report.md`](validation/results/heads_up/phase3b1_temporal_report.md) |
| **Phase 4B Metric Depth**| 3,919 depth points | Zone Classification | **94.5% zone agreement** | [`phase4b_external_depth_report.md`](validation/results/phase4/phase4b_external_depth_report.md) |
| **Phase 4C.1 Native TRT**| 150 benchmark frames | Native TensorRT FP16 Speed | **14.90 FPS** (33.8 ms depth) | [`phase4c1_native_tensorrt_report.md`](validation/results/phase4/phase4c1_native_tensorrt_report.md) |
| **Phase 5 Live Baseline**| 400 webcam frames | Static Room Stability | **0 STOP / 0 AVOID / 0 Alerts** | [`live_risk_navigation_audit.md`](validation/results/live/live_risk_navigation_audit.md) |

---

## 7. Current Architecture Decisions

1. **Scale-Invariant Disparity TTC as Authoritative Timing Signal**: Scale-invariant optical divergence ($\tau = d / \dot{d}$) remains the primary temporal collision estimator because unknown spatial metric scale cancels out in relative inverse depth.
2. **Metric Depth Reserved for Range Categorization**: Calibrated affine depth ($d_{\text{metric}}$) is used strictly for static proximity categorization (`NEAR` $<1.5\text{m}$, `MID` $1.5\text{m}$--$3.5\text{m}$, `FAR` $>3.5\text{m}$), avoiding noisy distance derivatives.
3. **Timestamp-Aware Source Frame Timing**: Kinematic rate calculations evaluate actual source frame delta ($dt = t_k - t_{k-1}$) rather than assuming constant 30 FPS spacing.
4. **Native TensorRT FP16 Deployment**: Live camera deployment compiles Depth Anything V2 into a native TensorRT FP16 execution engine, bypassing PyTorch Python overhead.

---

## 8. Live Demo Control Center & Scenarios

Launch the Phase 5.1 **Live Demo Control Center** GUI:

```powershell
python scripts/run/run_demo_control_center.py
```

### The Six Controlled Demonstration Scenarios
- **S01 — Clear Path**: Clear walking corridor; verifies `NO_WARNING` and `CONTINUE` baseline.
- **S02 — Static Obstacle**: Stationary obstacle placed 2–3 m ahead; verifies spatial zone placement and evasive steering (`AVOID_LEFT`/`AVOID_RIGHT`).
- **S03 — Person Approaching**: Subject approaches camera; verifies `APPROACHING` classification, decreasing TTC, risk escalation, and spoken alert.
- **S04 — Person Receding**: Subject walks away from camera; verifies `RECEDING` classification and suppressed closing risk.
- **S05 — Person Crossing**: Subject crosses corridor laterally; verifies lateral motion tracking and path clearance.
- **S06 — Head / Camera Movement**: Deliberate head panning and gait sway (>30°/s); verifies optical-flow ego-motion compensation.

> **Safety Notice**: Demo executions are strictly for **controlled, open-eye, supervised technical demonstrations**.

---

## 9. Repository Structure

```
Adaptive-edge-ai-navigation/
├── PROJECT_INDEX.md                             # Master project index & single source of truth
├── README.md                                    # Repository overview, architecture, & quickstart
├── LICENSE                                      # MIT Open-Source Research License
├── requirements.txt                             # Target Python package dependencies
├── pyproject.toml / pyrefly.toml                # Build system & static type checking configs
├── main.py                                      # Unified CLI application entry point
│
├── adaptive_navigation/                         # Core Python package (15 modular subsystems)
│   ├── perception/                              # Camera source, preprocessor, YOLO detector, BoT-SORT, Depth Anything V2
│   ├── tracking/                                # Object tracking interfaces
│   ├── temporal/                                # Motion estimator, history buffer, sparse LK optical flow ego-motion compensator
│   ├── risk/                                    # Scale-invariant optical divergence TTC, 5-feature RiskEngine
│   ├── navigation/                              # Spatial 2D geometry, walking corridor analysis, NavigationEngine
│   ├── audio/                                   # Offline non-blocking pyttsx3 TTS engine
│   ├── warning/                                 # 2-frame hysteresis WarningStateMachine & message generator
│   └── utils/                                   # General utilities
│
├── models/                                      # Canonical model artifacts
│   ├── detector/                                # YOLO11n weights (yolo11n.pt, yolo11n.onnx)
│   ├── depth/                                   # Depth Anything V2 weights (depth_anything_v2_vits.pth)
│   └── deployment/                              # TensorRT engines (yolo11n_fp16.engine, depth_anything_v2_vits_fp16.engine)
│
├── scripts/                                     # Diagnostic & execution scripts
│   ├── run/                                     # Execution entry points (run_demo_control_center.py, run_live_camera.py, etc.)
│   ├── tools/                                   # Diagnostic & download tools (check_environment.py, etc.)
│   └── benchmark/                               # Latency & throughput profiling scripts
│
├── validation/                                  # Research validation evidence & datasets
│   ├── datasets/                                # Dataset adapters & metadata (HEADS-UP egocentric dataset)
│   ├── calibration/                             # Camera intrinsic matrices & scale parameters
│   ├── videos/                                  # 5 controlled real-world validation videos (5,235 frames)
│   ├── results/                                 # Telemetry logs, phase reports (Phases 1A–5), plots, live audit JSONs
│   └── logs/                                    # Per-run execution telemetry logs
│
├── docs/                                        # Master documentation repository
│   ├── FILE_MANIFEST.md                         # Detailed file ownership & status index
│   ├── PROJECT_CLEANUP_REPORT.md                # Maintenance phase execution report
│   ├── live_demo_control_center.md              # Demo Control Center operator guide
│   ├── dataset_card.md / model_card.md          # Dataset & model specification cards
│   └── research/ validation/ architecture/      # Domain specific documentation
│
├── outputs/                                     # Export assets (demo overlays, figures, exported plots)
├── tests/                                       # Unit & integration test suite (17 tests)
└── archive/                                     # Archived non-active materials
    ├── debug/                                   # 20 temporary diagnostic scripts
    ├── obsolete/                                # Legacy synthetic evaluation scripts & CSVs
    └── legacy/                                  # Superseded early prototype code
```

---

## 10. Installation (Windows Setup)

```powershell
# 1. Clone repository
git clone https://github.com/devanshi007006-crypto/Adaptive-edge-ai-navigation.git
cd Adaptive-edge-ai-navigation

# 2. Activate Python virtual environment
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 11. Running the Project (Canonical Commands)

### A. Live Demo Control Center GUI (S01–S06 Scenarios)
```powershell
python scripts/run/run_demo_control_center.py
```

### B. Raw Live Camera Stream (TensorRT FP16)
```powershell
python scripts/run/run_live_camera.py
```

### C. Main Navigation Pipeline (Pre-Recorded Video)
```powershell
python main.py --config configs/final.yaml --video data/test_clip.mp4 --headless --max-frames 60
```

### D. System & Hardware Diagnostics
```powershell
python scripts/tools/check_environment.py
```

### E. Full Subsystem Unit Test Suite
```powershell
python -m unittest discover -s tests
```

---

## 12. Validation Reproduction

All research reports, telemetry logs, and evaluation metrics are preserved and reproducible:
- **Phase 2 Baseline & Fixes**: `validation/results/video_runs/`, `video_runs_phase2b/`, `video_runs_phase2c_egomotion/`
- **Phase 3 HEADS-UP Egocentric Evaluation**: `validation/results/heads_up/`
- **Phase 4 Depth Calibration & TensorRT Audit**: `validation/results/phase4/`
- **Phase 5 Live Webcam Telemetry & Audit**: `validation/results/live/` and [`live_risk_navigation_audit.md`](validation/results/live/live_risk_navigation_audit.md)

---

## 13. Documented System Limitations

1. **Severe Rapid Head Rotation**: Unconstrained head panning exceeding $>60^\circ/\text{s}$ can cause transient track ID fragmentation before optical flow compensation stabilizes.
2. **Far-Range Monocular Relative Depth Variance**: At distances $>5\text{m}$, monocular depth estimation exhibits higher variance, making relative disparity TTC authoritative over metric ranging.
3. **Exploratory External Benchmark Sample Size**: HEADS-UP benchmark evaluation was conducted on 250 extracted egocentric frames (8.33s duration) as exploratory stress testing rather than full-dataset clinical validation.
4. **No Visually Impaired Human-Subject Testing**: Clinical evaluation with visually impaired participants has **NOT** been performed. All testing is strictly restricted to controlled open-eye laboratory demonstrations.

---

## 14. Safety & Intended Use Notice

> [!WARNING]
> **RESEARCH PROTOTYPE NOTICE**:
> This software is an **experimental academic research prototype**.
> It is **NOT**:
> - A certified electronic travel aid (ETA) or medical device.
> - A replacement for a white cane, guide dog, or human mobility specialist.
> - Approved for unsupervised real-world navigation by blind or visually impaired individuals.

---

## 15. Technical Environment & Reproducibility Specs

- **Operating System**: Windows 11 x64 (Build 26100)
- **Python Runtime**: Python 3.14.4 / 3.11
- **Deep Learning Framework**: PyTorch `2.10.0+cu130` (CUDA 13.0)
- **Deployment Accelerator**: TensorRT `11.3.0.99` / ONNX Runtime `1.30.0`
- **Primary Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) + Intel Core i7
- **Configuration Profiles**: [`configs/final.yaml`](configs/final.yaml), [`configs/deployment.yaml`](configs/deployment.yaml)

---

## 16. License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for full terms. Third-party deep learning foundation models are subject to their respective upstream licenses (Ultralytics AGPL-3.0 / Depth Anything V2 Apache 2.0).
