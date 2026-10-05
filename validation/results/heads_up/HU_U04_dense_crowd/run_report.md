# HEADS-UP External Validation Run Report — HU_U04 Dense Plaza Crowd

**Sequence ID**: `HU_U04_dense_crowd`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Dense public plaza crowd with 5 simultaneous active pedestrians (Agents 12, 15, 16, 17, 18), cluttered background, occlusions.
- **Covered Scenarios**: dense object scene, multiple pedestrians, partial occlusion
- **Total Valid Frames**: 129
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 4.30 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 12.36 | s |
| **Real Effective Throughput** | **10.43** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 17.06 / 7.32 | FPS |
| **Total Frame Latency (p50 / Median)** | **136.57** | **ms** |
| **Total Frame Latency (p95)** | **145.85** | **ms** |
| **Total Frame Latency (Mean)** | 90.08 | ms |
| **Detector Latency (Mean)** | 11.04 | ms |
| **Tracker Latency (Mean)** | 7.58 | ms |
| **Depth Latency (Mean over all frames)** | 51.93 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 17.44 | ms |
| **TTC Step Latency (Mean)** | 0.04 | ms |
| **Risk Engine Latency (Mean)** | 0.07 | ms |
| **Navigation Latency (Mean)** | 0.05 | ms |
| **Peak GPU VRAM Allocated** | 168.80 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 918 | COCO-80 raw detections before policy |
| **Active Detections** | 713 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 205 (22.3%) | Non-navigation objects filtered out |
| **Person Detections** | 660 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 9 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 10 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 129 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 225 | 33.6% |
| **Receding Observations** | 245 | 36.6% |
| **Static / Stable Observations** | 179 | 26.8% |
| **Unknown / Insufficient Observations** | 20 | 3.0% |
| **Kinematic State Transitions (Approach <-> Recede)** | 17 | Motion direction reversals |
| **Valid TTC Calculations** | 256 | Finite, positive time horizons |
| **Invalid TTC Observations** | 413 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.65 s | Closest collision hazard horizon |
| **Median TTC Observed** | 3.67 s | Typical collision horizon |
| **95th Percentile TTC** | 22.41 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 6.83 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (0.8%)
- **CAUTION**: 97 frames (75.2%)
- **WARNING**: 31 frames (24.0%)
- **Risk State Transitions**: 9 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (1.6%)
- **CAUTION**: 0 frames (0.0%)
- **AVOID_LEFT**: 3 frames (2.3%)
- **AVOID_RIGHT**: 121 frames (93.8%)
- **STOP**: 3 frames (2.3%)
- **Navigation State Transitions**: 4 switches
- **Consecutive Frame Consistency**: 96.9%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 5 frames (3.9%)
- **RIGHT**: 121 frames (93.8%)
- **LEFT**: 3 frames (2.3%)
