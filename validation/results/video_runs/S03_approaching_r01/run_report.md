# Behavioral Run Report: S03_approaching_r01

## 1. Video & Execution Overview
- **Scenario**: `approaching` (Dynamic approaching obstacle / pedestrian directly closing in)
- **Video Path**: `validation/videos/approaching/S03_approaching_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 788
- **Wall-Clock Runtime**: 98.66 seconds
- **Effective Real FPS**: 7.99 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 11.04 | 9.63 | 15.54 |
| **BoT-SORT Tracker** | 3.96 | 3.86 | 4.91 |
| **Depth Anything V2 (vits)** | 97.12 | 96.66 | 101.16 |
| **TTC Estimation (Step 9)** | 0.03 | 0.02 | 0.04 |
| **Multi-Factor Risk Engine** | 0.03 | 0.03 | 0.04 |
| **Spatial Navigation Engine** | 0.02 | 0.02 | 0.03 |
| **User-Facing Audio / TTS Dispatch** | 0.57 | 0.01 | 0.02 |
| **End-to-End Per-Frame Pipeline** | **121.97** | **120.02** | **127.43** |

## 3. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 9 (IDs: [1, 6, 7, 8, 11, 19, 20, 21, 22])
- **Classes Detected**: {'person': 787, 'chair': 42, 'dining table': 23, 'toilet': 20, 'cell phone': 7}
- **Continuity & Stability**: Evaluated across 788 consecutive frames.

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 546, 'RECEDING': 272, 'STATIONARY': 0, 'UNKNOWN': 17, 'STABLE': 44}
- **Disparity Consistency**: Evaluated across tracked object instances.

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 546 instances
- **TTC Min / p50 / Max**: 0.37s / 2.49s / 30.0s (only computed when motion is approaching/closing)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 319, 'WARNING': 373, 'CRITICAL': 95}
- **Navigation Decisions**: {'CONTINUE': 3, 'CAUTION': 184, 'AVOID_LEFT': 555, 'AVOID_RIGHT': 46}
- **Recommended Safe Directions**: {'NONE': 187, 'LEFT': 555, 'RIGHT': 46}
- **Spoken Audio Warnings**: 38 utterances generated

## 4. Key Artifacts Generated
- `telemetry.json`: Complete hierarchical execution log.
- `telemetry_frames.csv`: Per-frame timing, motion, warning, and navigation state.
- `telemetry_objects.csv`: Per-object bounding box, depth, TTC, risk score, and warning state.
