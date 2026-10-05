# HEADS-UP External Validation Run Report — HU_U01 Multi-Pedestrian

**Sequence ID**: `HU_U01_multiped`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Multi-pedestrian egocentric scene with 3 simultaneous active pedestrians (Agents 1, 2, 3) in camera FOV.
- **Covered Scenarios**: single & multiple pedestrians, cluttered scene
- **Total Valid Frames**: 73
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 2.43 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 10.17 | s |
| **Real Effective Throughput** | **7.18** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 16.18 / 7.28 | FPS |
| **Total Frame Latency (p50 / Median)** | **137.37** | **ms** |
| **Total Frame Latency (p95)** | **161.30** | **ms** |
| **Total Frame Latency (Mean)** | 106.83 | ms |
| **Detector Latency (Mean)** | 21.17 | ms |
| **Tracker Latency (Mean)** | 7.82 | ms |
| **Depth Latency (Mean over all frames)** | 55.70 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 18.66 | ms |
| **TTC Step Latency (Mean)** | 0.04 | ms |
| **Risk Engine Latency (Mean)** | 0.05 | ms |
| **Navigation Latency (Mean)** | 0.04 | ms |
| **Peak GPU VRAM Allocated** | 138.24 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 426 | COCO-80 raw detections before policy |
| **Active Detections** | 311 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 115 (27.0%) | Non-navigation objects filtered out |
| **Person Detections** | 258 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 46 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 7 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 73 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 74 | 24.3% |
| **Receding Observations** | 78 | 25.7% |
| **Static / Stable Observations** | 139 | 45.7% |
| **Unknown / Insufficient Observations** | 13 | 4.3% |
| **Kinematic State Transitions (Approach <-> Recede)** | 4 | Motion direction reversals |
| **Valid TTC Calculations** | 170 | Finite, positive time horizons |
| **Invalid TTC Observations** | 134 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 1.24 s | Closest collision hazard horizon |
| **Median TTC Observed** | 4.29 s | Typical collision horizon |
| **95th Percentile TTC** | 30.00 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 8.21 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (1.4%)
- **CAUTION**: 67 frames (91.8%)
- **WARNING**: 5 frames (6.8%)
- **Risk State Transitions**: 3 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (2.7%)
- **CAUTION**: 33 frames (45.2%)
- **AVOID_LEFT**: 0 frames (0.0%)
- **AVOID_RIGHT**: 38 frames (52.1%)
- **STOP**: 0 frames (0.0%)
- **Navigation State Transitions**: 2 switches
- **Consecutive Frame Consistency**: 97.2%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 35 frames (47.9%)
- **RIGHT**: 38 frames (52.1%)
- **LEFT**: 0 frames (0.0%)
