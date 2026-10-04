# Behavioral Run Report: S04_receding_r01

## 1. Video & Execution Overview
- **Scenario**: `receding` (Object moving away / increasing distance from observer)
- **Video Path**: `validation/videos/receding/S04_receding_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 857
- **Wall-Clock Runtime**: 106.47 seconds
- **Effective Real FPS**: 8.05 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.70 | 9.46 | 15.16 |
| **BoT-SORT Tracker** | 4.37 | 4.45 | 4.99 |
| **Depth Anything V2 (vits)** | 96.73 | 96.39 | 99.73 |
| **TTC Estimation (Step 9)** | 0.03 | 0.03 | 0.04 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.01 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **121.15** | **119.96** | **127.24** |

## 3. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 10 (IDs: [1, 4, 10, 14, 17, 20, 22, 25, 31, 36])
- **Classes Detected**: {'person': 860, 'cell phone': 3, 'chair': 602, 'refrigerator': 4, 'suitcase': 108, 'bench': 5, 'cat': 2}
- **Continuity & Stability**: Evaluated across 857 consecutive frames.

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 864, 'RECEDING': 649, 'STATIONARY': 0, 'UNKNOWN': 19, 'STABLE': 52}
- **Disparity Consistency**: Evaluated across tracked object instances.

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 864 instances
- **TTC Min / p50 / Max**: 0.27s / 3.15s / 30.0s (only computed when motion is approaching/closing)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 3, 'CAUTION': 483, 'WARNING': 276, 'CRITICAL': 95}
- **Navigation Decisions**: {'CONTINUE': 2, 'AVOID_LEFT': 464, 'AVOID_RIGHT': 203, 'STOP': 188}
- **Recommended Safe Directions**: {'NONE': 190, 'LEFT': 464, 'RIGHT': 203}
- **Spoken Audio Warnings**: 14 utterances generated

## 4. Key Artifacts Generated
- `telemetry.json`: Complete hierarchical execution log.
- `telemetry_frames.csv`: Per-frame timing, motion, warning, and navigation state.
- `telemetry_objects.csv`: Per-object bounding box, depth, TTC, risk score, and warning state.
