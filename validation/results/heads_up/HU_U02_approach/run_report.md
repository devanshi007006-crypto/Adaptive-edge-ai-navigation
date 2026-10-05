# HEADS-UP External Validation Run Report — HU_U02 Steady Approach

**Sequence ID**: `HU_U02_approach`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Steady forward walking with closing oncoming pedestrian (Agent 5, closing from 10.5m to 6.5m).
- **Covered Scenarios**: approaching pedestrian, normal walking, sparse scene
- **Total Valid Frames**: 102
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 3.40 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 9.65 | s |
| **Real Effective Throughput** | **10.57** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 17.59 / 15.47 | FPS |
| **Total Frame Latency (p50 / Median)** | **88.25** | **ms** |
| **Total Frame Latency (p95)** | **143.88** | **ms** |
| **Total Frame Latency (Mean)** | 88.36 | ms |
| **Detector Latency (Mean)** | 11.10 | ms |
| **Tracker Latency (Mean)** | 7.06 | ms |
| **Depth Latency (Mean over all frames)** | 51.44 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 17.30 | ms |
| **TTC Step Latency (Mean)** | 0.03 | ms |
| **Risk Engine Latency (Mean)** | 0.04 | ms |
| **Navigation Latency (Mean)** | 0.03 | ms |
| **Peak GPU VRAM Allocated** | 148.42 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 429 | COCO-80 raw detections before policy |
| **Active Detections** | 209 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 220 (51.3%) | Non-navigation objects filtered out |
| **Person Detections** | 157 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 21 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 4 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 102 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 98 | 55.1% |
| **Receding Observations** | 9 | 5.1% |
| **Static / Stable Observations** | 62 | 34.8% |
| **Unknown / Insufficient Observations** | 9 | 5.1% |
| **Kinematic State Transitions (Approach <-> Recede)** | 0 | Motion direction reversals |
| **Valid TTC Calculations** | 125 | Finite, positive time horizons |
| **Invalid TTC Observations** | 53 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 2.30 s | Closest collision hazard horizon |
| **Median TTC Observed** | 3.84 s | Typical collision horizon |
| **95th Percentile TTC** | 16.48 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 5.27 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (1.0%)
- **CAUTION**: 101 frames (99.0%)
- **WARNING**: 0 frames (0.0%)
- **Risk State Transitions**: 1 switches

### Navigation Decisions
- **CONTINUE**: 3 frames (2.9%)
- **CAUTION**: 99 frames (97.1%)
- **AVOID_LEFT**: 0 frames (0.0%)
- **AVOID_RIGHT**: 0 frames (0.0%)
- **STOP**: 0 frames (0.0%)
- **Navigation State Transitions**: 1 switches
- **Consecutive Frame Consistency**: 99.0%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 102 frames (100.0%)
- **RIGHT**: 0 frames (0.0%)
- **LEFT**: 0 frames (0.0%)
