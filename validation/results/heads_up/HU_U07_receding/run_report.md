# HEADS-UP External Validation Run Report — HU_U07 Receding Pedestrians

**Sequence ID**: `HU_U07_receding`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Monotonically receding pedestrians walking ahead in open plaza (Agents 102 & 104, distance expanding 21.4m -> 39.5m).
- **Covered Scenarios**: receding pedestrians, sparse object scene, open space
- **Total Valid Frames**: 126
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 4.20 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 14.98 | s |
| **Real Effective Throughput** | **8.41** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 12.44 / 11.11 | FPS |
| **Total Frame Latency (p50 / Median)** | **109.16** | **ms** |
| **Total Frame Latency (p95)** | **178.37** | **ms** |
| **Total Frame Latency (Mean)** | 111.49 | ms |
| **Detector Latency (Mean)** | 16.97 | ms |
| **Tracker Latency (Mean)** | 11.19 | ms |
| **Depth Latency (Mean over all frames)** | 56.98 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 23.21 | ms |
| **TTC Step Latency (Mean)** | 0.07 | ms |
| **Risk Engine Latency (Mean)** | 0.08 | ms |
| **Navigation Latency (Mean)** | 0.06 | ms |
| **Peak GPU VRAM Allocated** | 156.40 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 691 | COCO-80 raw detections before policy |
| **Active Detections** | 673 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 18 (2.6%) | Non-navigation objects filtered out |
| **Person Detections** | 613 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 8 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 15 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 126 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 354 | 57.0% |
| **Receding Observations** | 76 | 12.2% |
| **Static / Stable Observations** | 162 | 26.1% |
| **Unknown / Insufficient Observations** | 29 | 4.7% |
| **Kinematic State Transitions (Approach <-> Recede)** | 8 | Motion direction reversals |
| **Valid TTC Calculations** | 462 | Finite, positive time horizons |
| **Invalid TTC Observations** | 159 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.41 s | Closest collision hazard horizon |
| **Median TTC Observed** | 2.67 s | Typical collision horizon |
| **95th Percentile TTC** | 13.71 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 4.75 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (0.8%)
- **CAUTION**: 6 frames (4.8%)
- **WARNING**: 119 frames (94.4%)
- **Risk State Transitions**: 2 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (1.6%)
- **CAUTION**: 0 frames (0.0%)
- **AVOID_LEFT**: 92 frames (73.0%)
- **AVOID_RIGHT**: 13 frames (10.3%)
- **STOP**: 19 frames (15.1%)
- **Navigation State Transitions**: 5 switches
- **Consecutive Frame Consistency**: 96.0%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 21 frames (16.7%)
- **RIGHT**: 13 frames (10.3%)
- **LEFT**: 92 frames (73.0%)
