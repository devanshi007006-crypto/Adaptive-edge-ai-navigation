# HEADS-UP External Validation Run Report — HEADS-UP S01 Multi-Pedestrian

**Sequence ID**: `HU_unconstrained_s01_multiped`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Forward Ego-Motion Compensation)  

---

## 1. Scenario Description
- **Description**: Multiple pedestrians in field of view (Agents 1, 2, 3 simultaneously active)
- **Total Valid Frames**: 73
- **Resolution**: 1280 × 720 @ 30 FPS
- **Sequence Duration**: 2.43 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 10.31 | seconds |
| **Real Throughput** | **7.08** | **FPS** |
| **Total Frame Latency (Mean)** | 109.55 | ms |
| **Total Frame Latency (Median / p50)** | **135.56** | **ms** |
| **Total Frame Latency (p95)** | **156.84** | **ms** |
| **Depth Step Latency (Mean)** | 55.02 | ms |
| **Depth Step Latency (Median / p50)** | 99.79 | ms |
| **Detector Latency (Mean)** | 24.31 | ms |
| **Tracker Latency (Mean)** | 8.10 | ms |
| **Peak GPU VRAM Allocated** | 138.24 | MB |

---

## 3. Perception, Tracking & Behavioral Dynamics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Unique Track IDs** | 7 | Track IDs: `[1, 2, 3, 4, 5, 6, 11]` |
| **Longest Track Lifetime** | 73 frames | Max continuous track persistence |
| **Valid TTC Calculations** | 170 | Finite, positive time-to-collision frames |
| **Minimum TTC Observed** | 1.24 s | Closest collision hazard horizon |
| **Mean TTC Observed** | 8.49 s | Average collision horizon |

### Object Kinematic Motion States
- **APPROACHING**: 74 object-frames
- **RECEDING**: 78 object-frames
- **STATIONARY**: 0 object-frames
- **STABLE / UNKNOWN**: 152 object-frames

### Warning State Distribution
- **NO_WARNING (Clear)**: 1 frames (1.4%)
- **CAUTION**: 67 frames (91.8%)
- **WARNING**: 5 frames (6.8%)

### Navigation Guidance & Evasive Maneuvers
- **CONTINUE**: 2 frames (2.7%)
- **CAUTION (Slow / Monitor)**: 33 frames (45.2%)
- **AVOID_LEFT (Steer Left)**: 0 frames (0.0%)
- **AVOID_RIGHT (Steer Right)**: 38 frames (52.1%)
- **STOP (Emergency Stop)**: 0 frames (0.0%)

### Recommended Safe Steering Directions
- **NONE (Path unblocked / Stopped)**: 35 frames (47.9%)
- **RIGHT**: 38 frames (52.1%)
- **LEFT**: 0 frames (0.0%)

---

## 4. Dataset Provenance Limitation Statement
*Note: The reference trajectory annotations in HEADS-UP were machine-generated using YOLOv8, ByteTrack, temporal averaging, and Kalman filtering. They represent an external trajectory reference for qualitative comparison, NOT an independent ground-truth oracle.*
