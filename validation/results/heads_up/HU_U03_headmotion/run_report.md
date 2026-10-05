# HEADS-UP External Validation Run Report — HU_U03 Strong Head Motion

**Sequence ID**: `HU_U03_headmotion`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Strong head scanning / saccadic rotation (>60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m to 4.4m).
- **Covered Scenarios**: strong head motion, camera translation, close hazard
- **Total Valid Frames**: 75
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 2.50 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 7.55 | s |
| **Real Effective Throughput** | **9.93** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 16.83 / 7.30 | FPS |
| **Total Frame Latency (p50 / Median)** | **137.07** | **ms** |
| **Total Frame Latency (p95)** | **148.92** | **ms** |
| **Total Frame Latency (Mean)** | 90.86 | ms |
| **Detector Latency (Mean)** | 11.18 | ms |
| **Tracker Latency (Mean)** | 7.61 | ms |
| **Depth Latency (Mean over all frames)** | 52.28 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 17.51 | ms |
| **TTC Step Latency (Mean)** | 0.05 | ms |
| **Risk Engine Latency (Mean)** | 0.06 | ms |
| **Navigation Latency (Mean)** | 0.04 | ms |
| **Peak GPU VRAM Allocated** | 158.61 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 433 | COCO-80 raw detections before policy |
| **Active Detections** | 421 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 12 (2.8%) | Non-navigation objects filtered out |
| **Person Detections** | 310 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 16 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 14 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 75 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 113 | 34.7% |
| **Receding Observations** | 44 | 13.5% |
| **Static / Stable Observations** | 140 | 42.9% |
| **Unknown / Insufficient Observations** | 29 | 8.9% |
| **Kinematic State Transitions (Approach <-> Recede)** | 3 | Motion direction reversals |
| **Valid TTC Calculations** | 199 | Finite, positive time horizons |
| **Invalid TTC Observations** | 127 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.28 s | Closest collision hazard horizon |
| **Median TTC Observed** | 2.05 s | Typical collision horizon |
| **95th Percentile TTC** | 8.72 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 2.81 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (1.3%)
- **CAUTION**: 2 frames (2.7%)
- **WARNING**: 72 frames (96.0%)
- **Risk State Transitions**: 2 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (2.7%)
- **CAUTION**: 0 frames (0.0%)
- **AVOID_LEFT**: 15 frames (20.0%)
- **AVOID_RIGHT**: 41 frames (54.7%)
- **STOP**: 17 frames (22.7%)
- **Navigation State Transitions**: 4 switches
- **Consecutive Frame Consistency**: 94.6%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 19 frames (25.3%)
- **RIGHT**: 41 frames (54.7%)
- **LEFT**: 15 frames (20.0%)
