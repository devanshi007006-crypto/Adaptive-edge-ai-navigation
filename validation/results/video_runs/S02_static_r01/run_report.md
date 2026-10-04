# Behavioral Run Report: S02_static_r01

## 1. Video & Execution Overview
- **Scenario**: `static_obstacle` (Stationary obstacle in corridor with ego-camera approach)
- **Video Path**: `validation/videos/static_obstacle/S02_static_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 1158
- **Wall-Clock Runtime**: 144.65 seconds
- **Effective Real FPS**: 8.01 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.84 | 9.67 | 15.52 |
| **BoT-SORT Tracker** | 4.36 | 4.39 | 5.07 |
| **Depth Anything V2 (vits)** | 96.82 | 96.62 | 99.39 |
| **TTC Estimation (Step 9)** | 0.03 | 0.02 | 0.04 |
| **Multi-Factor Risk Engine** | 0.04 | 0.03 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.42 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **121.88** | **120.55** | **127.74** |

## 3. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 14 (IDs: [1, 5, 9, 11, 12, 14, 18, 25, 38, 40, 42, 49, 52, 53])
- **Classes Detected**: {'person': 1164, 'cell phone': 2, 'chair': 767, 'suitcase': 141, 'refrigerator': 1, 'couch': 5, 'bed': 3, 'cat': 2}
- **Continuity & Stability**: Evaluated across 1158 consecutive frames.

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 1215, 'RECEDING': 729, 'STATIONARY': 0, 'UNKNOWN': 26, 'STABLE': 115}
- **Disparity Consistency**: Evaluated across tracked object instances.

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 1215 instances
- **TTC Min / p50 / Max**: 0.19s / 2.78s / 30.0s (only computed when motion is approaching/closing)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 662, 'WARNING': 439, 'CRITICAL': 56}
- **Navigation Decisions**: {'CONTINUE': 29, 'CAUTION': 403, 'AVOID_LEFT': 363, 'AVOID_RIGHT': 206, 'STOP': 157}
- **Recommended Safe Directions**: {'NONE': 589, 'LEFT': 363, 'RIGHT': 206}
- **Spoken Audio Warnings**: 50 utterances generated

## 4. Key Artifacts Generated
- `telemetry.json`: Complete hierarchical execution log.
- `telemetry_frames.csv`: Per-frame timing, motion, warning, and navigation state.
- `telemetry_objects.csv`: Per-object bounding box, depth, TTC, risk score, and warning state.
