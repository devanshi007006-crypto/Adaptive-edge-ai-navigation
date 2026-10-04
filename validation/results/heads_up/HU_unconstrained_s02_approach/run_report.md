# HEADS-UP External Validation Run Report — HEADS-UP S02 Steady Approach

**Sequence ID**: `HU_unconstrained_s02_approach`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Forward Ego-Motion Compensation)  

---

## 1. Scenario Description
- **Description**: Ordinary head motion, steady walking closing in on pedestrian (Agent 5, distance 10.5m -> 6.5m)
- **Total Valid Frames**: 102
- **Resolution**: 1280 × 720 @ 30 FPS
- **Sequence Duration**: 3.40 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 9.77 | seconds |
| **Real Throughput** | **10.44** | **FPS** |
| **Total Frame Latency (Mean)** | 89.12 | ms |
| **Total Frame Latency (Median / p50)** | **92.80** | **ms** |
| **Total Frame Latency (p95)** | **147.18** | **ms** |
| **Depth Step Latency (Mean)** | 51.57 | ms |
| **Depth Step Latency (Median / p50)** | 50.06 | ms |
| **Detector Latency (Mean)** | 11.61 | ms |
| **Tracker Latency (Mean)** | 7.09 | ms |
| **Peak GPU VRAM Allocated** | 148.42 | MB |

---

## 3. Perception, Tracking & Behavioral Dynamics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Unique Track IDs** | 4 | Track IDs: `[1, 6, 17, 29]` |
| **Longest Track Lifetime** | 102 frames | Max continuous track persistence |
| **Valid TTC Calculations** | 125 | Finite, positive time-to-collision frames |
| **Minimum TTC Observed** | 2.30 s | Closest collision hazard horizon |
| **Mean TTC Observed** | 6.05 s | Average collision horizon |

### Object Kinematic Motion States
- **APPROACHING**: 98 object-frames
- **RECEDING**: 9 object-frames
- **STATIONARY**: 0 object-frames
- **STABLE / UNKNOWN**: 71 object-frames

### Warning State Distribution
- **NO_WARNING (Clear)**: 1 frames (1.0%)
- **CAUTION**: 101 frames (99.0%)
- **WARNING**: 0 frames (0.0%)

### Navigation Guidance & Evasive Maneuvers
- **CONTINUE**: 3 frames (2.9%)
- **CAUTION (Slow / Monitor)**: 99 frames (97.1%)
- **AVOID_LEFT (Steer Left)**: 0 frames (0.0%)
- **AVOID_RIGHT (Steer Right)**: 0 frames (0.0%)
- **STOP (Emergency Stop)**: 0 frames (0.0%)

### Recommended Safe Steering Directions
- **NONE (Path unblocked / Stopped)**: 102 frames (100.0%)
- **RIGHT**: 0 frames (0.0%)
- **LEFT**: 0 frames (0.0%)

---

## 4. Dataset Provenance Limitation Statement
*Note: The reference trajectory annotations in HEADS-UP were machine-generated using YOLOv8, ByteTrack, temporal averaging, and Kalman filtering. They represent an external trajectory reference for qualitative comparison, NOT an independent ground-truth oracle.*
