# HEADS-UP External Validation Run Report — HEADS-UP S03 Strong Head Motion

**Sequence ID**: `HU_unconstrained_s03_headmotion`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Forward Ego-Motion Compensation)  

---

## 1. Scenario Description
- **Description**: Strong head rotation / scanning (>60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m -> 4.4m)
- **Total Valid Frames**: 75
- **Resolution**: 1280 × 720 @ 30 FPS
- **Sequence Duration**: 2.50 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 7.63 | seconds |
| **Real Throughput** | **9.83** | **FPS** |
| **Total Frame Latency (Mean)** | 91.79 | ms |
| **Total Frame Latency (Median / p50)** | **137.18** | **ms** |
| **Total Frame Latency (p95)** | **148.66** | **ms** |
| **Depth Step Latency (Mean)** | 52.52 | ms |
| **Depth Step Latency (Median / p50)** | 99.61 | ms |
| **Detector Latency (Mean)** | 11.61 | ms |
| **Tracker Latency (Mean)** | 7.70 | ms |
| **Peak GPU VRAM Allocated** | 158.61 | MB |

---

## 3. Perception, Tracking & Behavioral Dynamics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Unique Track IDs** | 14 | Track IDs: `[1, 2, 3, 4, 22, 33, 43, 49, 62, 63, 64, 71, 82, 95]` |
| **Longest Track Lifetime** | 75 frames | Max continuous track persistence |
| **Valid TTC Calculations** | 199 | Finite, positive time-to-collision frames |
| **Minimum TTC Observed** | 0.28 s | Closest collision hazard horizon |
| **Mean TTC Observed** | 2.93 s | Average collision horizon |

### Object Kinematic Motion States
- **APPROACHING**: 113 object-frames
- **RECEDING**: 44 object-frames
- **STATIONARY**: 0 object-frames
- **STABLE / UNKNOWN**: 169 object-frames

### Warning State Distribution
- **NO_WARNING (Clear)**: 1 frames (1.3%)
- **CAUTION**: 2 frames (2.7%)
- **WARNING**: 72 frames (96.0%)

### Navigation Guidance & Evasive Maneuvers
- **CONTINUE**: 2 frames (2.7%)
- **CAUTION (Slow / Monitor)**: 0 frames (0.0%)
- **AVOID_LEFT (Steer Left)**: 15 frames (20.0%)
- **AVOID_RIGHT (Steer Right)**: 41 frames (54.7%)
- **STOP (Emergency Stop)**: 17 frames (22.7%)

### Recommended Safe Steering Directions
- **NONE (Path unblocked / Stopped)**: 19 frames (25.3%)
- **RIGHT**: 41 frames (54.7%)
- **LEFT**: 15 frames (20.0%)

---

## 4. Dataset Provenance Limitation Statement
*Note: The reference trajectory annotations in HEADS-UP were machine-generated using YOLOv8, ByteTrack, temporal averaging, and Kalman filtering. They represent an external trajectory reference for qualitative comparison, NOT an independent ground-truth oracle.*
