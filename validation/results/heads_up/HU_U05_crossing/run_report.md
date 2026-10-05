# HEADS-UP External Validation Run Report — HU_U05 Lateral Crossing

**Sequence ID**: `HU_U05_crossing`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Pedestrian crossing diagonally across user's forward walking corridor (Agent 21, distance ~28.7m -> 32.8m).
- **Covered Scenarios**: lateral/crossing motion, camera translation
- **Total Valid Frames**: 125
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 4.17 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 13.18 | s |
| **Real Effective Throughput** | **9.48** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 15.23 / 7.24 | FPS |
| **Total Frame Latency (p50 / Median)** | **138.18** | **ms** |
| **Total Frame Latency (p95)** | **196.51** | **ms** |
| **Total Frame Latency (Mean)** | 99.31 | ms |
| **Detector Latency (Mean)** | 12.84 | ms |
| **Tracker Latency (Mean)** | 8.74 | ms |
| **Depth Latency (Mean over all frames)** | 54.52 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 20.18 | ms |
| **TTC Step Latency (Mean)** | 0.06 | ms |
| **Risk Engine Latency (Mean)** | 0.07 | ms |
| **Navigation Latency (Mean)** | 0.05 | ms |
| **Peak GPU VRAM Allocated** | 138.24 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 929 | COCO-80 raw detections before policy |
| **Active Detections** | 674 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 255 (27.4%) | Non-navigation objects filtered out |
| **Person Detections** | 591 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 12 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 17 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 109 frames | Max continuous track lifetime (87.2% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 320 | 53.1% |
| **Receding Observations** | 95 | 15.8% |
| **Static / Stable Observations** | 155 | 25.7% |
| **Unknown / Insufficient Observations** | 33 | 5.5% |
| **Kinematic State Transitions (Approach <-> Recede)** | 8 | Motion direction reversals |
| **Valid TTC Calculations** | 384 | Finite, positive time horizons |
| **Invalid TTC Observations** | 219 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.27 s | Closest collision hazard horizon |
| **Median TTC Observed** | 3.99 s | Typical collision horizon |
| **95th Percentile TTC** | 29.95 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 7.44 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (0.8%)
- **CAUTION**: 19 frames (15.2%)
- **WARNING**: 91 frames (72.8%)
- **Risk State Transitions**: 7 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (1.6%)
- **CAUTION**: 15 frames (12.0%)
- **AVOID_LEFT**: 16 frames (12.8%)
- **AVOID_RIGHT**: 36 frames (28.8%)
- **STOP**: 56 frames (44.8%)
- **Navigation State Transitions**: 8 switches
- **Consecutive Frame Consistency**: 93.5%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 73 frames (58.4%)
- **RIGHT**: 36 frames (28.8%)
- **LEFT**: 16 frames (12.8%)
