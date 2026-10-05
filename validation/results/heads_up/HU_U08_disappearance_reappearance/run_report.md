# HEADS-UP External Validation Run Report — HU_U08 Track Disappearance / Reappearance

**Sequence ID**: `HU_U08_disappearance_reappearance`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `NVIDIA GeForce RTX 4050 Laptop GPU` on `cuda:0` | Python `3.14.4`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: Camera translation through architectural boundary with peripheral pedestrian disappearing behind boundary and re-emerging (Agent 179).
- **Covered Scenarios**: temporary track disappearance/reappearance, camera translation
- **Total Valid Frames**: 112
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: 3.73 seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | 14.27 | s |
| **Real Effective Throughput** | **7.85** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | 11.72 / 9.01 | FPS |
| **Total Frame Latency (p50 / Median)** | **120.98** | **ms** |
| **Total Frame Latency (p95)** | **202.72** | **ms** |
| **Total Frame Latency (Mean)** | 119.32 | ms |
| **Detector Latency (Mean)** | 18.75 | ms |
| **Tracker Latency (Mean)** | 11.53 | ms |
| **Depth Latency (Mean over all frames)** | 60.11 | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | 25.63 | ms |
| **TTC Step Latency (Mean)** | 0.06 | ms |
| **Risk Engine Latency (Mean)** | 0.07 | ms |
| **Navigation Latency (Mean)** | 0.05 | ms |
| **Peak GPU VRAM Allocated** | 166.58 | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | 501 | COCO-80 raw detections before policy |
| **Active Detections** | 392 | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | 109 (21.8%) | Non-navigation objects filtered out |
| **Person Detections** | 322 | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | 31 | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | 10 | Unique persistent IDs across sequence |
| **Longest Track Persistence** | 103 frames | Max continuous track lifetime (92.0% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | 203 | 57.5% |
| **Receding Observations** | 66 | 18.7% |
| **Static / Stable Observations** | 66 | 18.7% |
| **Unknown / Insufficient Observations** | 18 | 5.1% |
| **Kinematic State Transitions (Approach <-> Recede)** | 3 | Motion direction reversals |
| **Valid TTC Calculations** | 233 | Finite, positive time horizons |
| **Invalid TTC Observations** | 120 | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | 0.36 s | Closest collision hazard horizon |
| **Median TTC Observed** | 1.44 s | Typical collision horizon |
| **95th Percentile TTC** | 7.99 s | Long-range hazard horizon |
| **TTC Standard Deviation** | 3.72 s | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: 1 frames (0.9%)
- **CAUTION**: 18 frames (16.1%)
- **WARNING**: 38 frames (33.9%)
- **Risk State Transitions**: 5 switches

### Navigation Decisions
- **CONTINUE**: 2 frames (1.8%)
- **CAUTION**: 0 frames (0.0%)
- **AVOID_LEFT**: 28 frames (25.0%)
- **AVOID_RIGHT**: 49 frames (43.8%)
- **STOP**: 33 frames (29.5%)
- **Navigation State Transitions**: 8 switches
- **Consecutive Frame Consistency**: 92.8%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: 35 frames (31.2%)
- **RIGHT**: 49 frames (43.8%)
- **LEFT**: 28 frames (25.0%)
