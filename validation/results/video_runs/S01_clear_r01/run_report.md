# Behavioral Run Report: S01_clear_r01

## 1. Video & Execution Overview
- **Scenario**: `clear_path` (Negative control: clear unobstructed corridor traversal)
- **Video Path**: `validation/videos/clear_path/S01_clear_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 1524
- **Wall-Clock Runtime**: 190.06 seconds
- **Effective Real FPS**: 8.02 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 9.99 | 9.03 | 14.40 |
| **BoT-SORT Tracker** | 4.16 | 4.21 | 4.81 |
| **Depth Anything V2 (vits)** | 98.73 | 97.47 | 105.27 |
| **TTC Estimation (Step 9)** | 0.02 | 0.02 | 0.03 |
| **Multi-Factor Risk Engine** | 0.01 | 0.00 | 0.03 |
| **Spatial Navigation Engine** | 0.01 | 0.01 | 0.02 |
| **User-Facing Audio / TTS Dispatch** | 0.03 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **122.46** | **120.62** | **131.27** |

## 3. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 18 (IDs: [2, 4, 11, 13, 15, 19, 21, 35, 39, 47, 50, 52, 56, 57, 58, 60, 69, 75])
- **Classes Detected**: {'refrigerator': 68, 'toilet': 81, 'cat': 98, 'tv': 7, 'potted plant': 7, 'dog': 3, 'microwave': 2}
- **Continuity & Stability**: Evaluated across 1524 consecutive frames.

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 153, 'RECEDING': 70, 'STATIONARY': 0, 'UNKNOWN': 34, 'STABLE': 9}
- **Disparity Consistency**: Evaluated across tracked object instances.

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 153 instances
- **TTC Min / p50 / Max**: 1.01s / 4.66s / 30.0s (only computed when motion is approaching/closing)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 731, 'CAUTION': 715, 'WARNING': 78, 'CRITICAL': 0}
- **Navigation Decisions**: {'CONTINUE': 794, 'CAUTION': 664, 'AVOID_LEFT': 66}
- **Recommended Safe Directions**: {'NONE': 1458, 'LEFT': 66}
- **Spoken Audio Warnings**: 20 utterances generated

## 4. Key Artifacts Generated
- `telemetry.json`: Complete hierarchical execution log.
- `telemetry_frames.csv`: Per-frame timing, motion, warning, and navigation state.
- `telemetry_objects.csv`: Per-object bounding box, depth, TTC, risk score, and warning state.
