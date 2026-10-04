# Behavioral Run Report: S05_crossing_r01

## 1. Video & Execution Overview
- **Scenario**: `crossing` (Dynamic orthogonal / transverse crossing pedestrian)
- **Video Path**: `validation/videos/crossing/S05_crossing_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 908
- **Wall-Clock Runtime**: 114.53 seconds
- **Effective Real FPS**: 7.93 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.95 | 9.67 | 15.29 |
| **BoT-SORT Tracker** | 4.83 | 4.75 | 5.43 |
| **Depth Anything V2 (vits)** | 97.25 | 96.88 | 100.48 |
| **TTC Estimation (Step 9)** | 0.03 | 0.03 | 0.04 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.08 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **123.10** | **121.69** | **129.43** |

## 3. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 18 (IDs: [1, 2, 3, 4, 7, 16, 21, 25, 38, 41, 42, 48, 52, 54, 63, 69, 70, 73])
- **Classes Detected**: {'person': 887, 'chair': 845, 'refrigerator': 100, 'handbag': 4}
- **Continuity & Stability**: Evaluated across 908 consecutive frames.

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 949, 'RECEDING': 770, 'STATIONARY': 0, 'UNKNOWN': 37, 'STABLE': 80}
- **Disparity Consistency**: Evaluated across tracked object instances.

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 949 instances
- **TTC Min / p50 / Max**: 0.69s / 6.07s / 30.0s (only computed when motion is approaching/closing)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 668, 'WARNING': 219, 'CRITICAL': 20}
- **Navigation Decisions**: {'CONTINUE': 70, 'CAUTION': 486, 'STOP': 44, 'AVOID_LEFT': 308}
- **Recommended Safe Directions**: {'NONE': 600, 'LEFT': 308}
- **Spoken Audio Warnings**: 18 utterances generated

## 4. Key Artifacts Generated
- `telemetry.json`: Complete hierarchical execution log.
- `telemetry_frames.csv`: Per-frame timing, motion, warning, and navigation state.
- `telemetry_objects.csv`: Per-object bounding box, depth, TTC, risk score, and warning state.
