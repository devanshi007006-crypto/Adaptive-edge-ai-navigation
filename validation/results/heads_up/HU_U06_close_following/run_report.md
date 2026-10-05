# HEADS-UP External Validation Run Report — HU_U06 Close Following Cluster

**Sequence ID**: `HU_U06_close_following`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Walking forward in public plaza trailing a dynamic pedestrian cluster (Agents 50, 51, 53, 55).
- **Covered Scenarios**: multi-pedestrian cluster, camera translation, dynamic depth
- **Total Valid Frames**: 110
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 3.67 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 14.06 | s |
| **Real Effective Throughput** | **7.82** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 11.68 / 10.35 | FPS |
| **Total Frame Latency (p50 / Median)** | **112.45** | **ms** |
| **Total Frame Latency (p95)** | **203.24** | **ms** |
| **Total Frame Latency (Mean)** | 118.21 | ms |
| **Detector Latency (Mean)** | 18.13 | ms |
| **Tracker Latency (Mean)** | 11.51 | ms |
| **Depth Latency (Mean over all frames)** | 59.50 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 24.63 | ms |
| **TTC Step Latency (Mean)** | 0.10 | ms |
| **Risk Engine Latency (Mean)** | 0.12 | ms |
| **Navigation Latency (Mean)** | 0.09 | ms |
| **Peak GPU VRAM Allocated** | 146.21 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 1182 | COCO-80 raw detections before policy |
| **Active Detections** | 937 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 245 (20.7%) | Non-navigation objects filtered out |
| **Person Detections** | 812 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 53 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 21 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 110 frames | Max continuous track lifetime (100.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 281 | 32.5% |
| **Receding Observations** | 258 | 29.8% |
| **Static / Stable Observations** | 283 | 32.7% |
| **Unknown / Insufficient Observations** | 43 | 5.0% |
| **Kinematic State Transitions (Approach <-> Recede)** | 14 | Motion direction reversals |
| **Valid TTC Calculations** | 429 | Finite, positive time horizons |
| **Invalid TTC Observations** | 436 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.35 s | Closest collision hazard horizon |
| **Median TTC Observed** | 6.12 s | Typical collision horizon |
| **95th Percentile TTC** | 29.05 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 7.31 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (0.9%)
- **CAUTION**: 16 frames (14.5%)
- **WARNING**: 88 frames (80.0%)
- **Risk State Transitions**: 8 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (1.8%)
- **CAUTION**: 0 frames (0.0%)
- **AVOID_LEFT**: 84 frames (76.4%)
- **AVOID_RIGHT**: 13 frames (11.8%)
- **STOP**: 11 frames (10.0%)
- **Navigation State Transitions**: 5 switches
- **Consecutive Frame Consistency**: 95.4%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 13 frames (11.8%)
- **RIGHT**: 13 frames (11.8%)
- **LEFT**: 84 frames (76.4%)
