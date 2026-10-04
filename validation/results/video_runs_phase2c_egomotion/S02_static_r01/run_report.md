# Phase 2C Benchmark Run Report: S02_static_r01

## 1. Video & Execution Overview
- **Scenario**: `static_obstacle` (Stationary obstacle in corridor with ego-camera approach)
- **Video Path**: `validation/videos/static_obstacle/S02_static_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Depth Cadence**: `2:1` (Depth Anything V2 executes every 2 frame(s))
- **Total Frames Processed**: 1158
- **Wall-Clock Runtime**: 97.09 seconds
- **Effective Real FPS**: 11.93 FPS
- **Max VRAM Allocated**: 158.61 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 536.00 MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: 2138
- **Active Hazard Detections**: 2120
- **Policy-Filtered Detections**: 18
- **Raw Detected Classes**: {'person': 1164, 'dog': 1, 'skateboard': 2, 'cell phone': 3, 'chair': 790, 'suitcase': 147, 'refrigerator': 8, 'couch': 8, 'bed': 7, 'bench': 3, 'cat': 5}
- **Active Hazard Classes Tracked**: {'person': 1164, 'dog': 1, 'chair': 790, 'suitcase': 147, 'couch': 8, 'bed': 7, 'bench': 3}
- **Filtered Classes**: {'skateboard': 2, 'cell phone': 3, 'refrigerator': 8, 'cat': 5}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 13.24 | 11.59 | 21.56 |
| **BoT-SORT Tracker** | 5.28 | 4.88 | 8.10 |
| **Depth Anything V2 (vits, cadence 2:1)** | 51.90 | 0.00 | 115.76 |
| **Depth When Executing** | 103.88 ms | - | - |
| **TTC Estimation (Step 9)** | 0.03 | 0.03 | 0.05 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.07 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.06 |
| **User-Facing Audio / TTS Dispatch** | 0.01 | 0.01 | 0.02 |
| **End-to-End Per-Frame Pipeline** | **81.91** | **61.42** | **154.79** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 12 (IDs: [1, 6, 8, 9, 11, 18, 28, 30, 32, 39, 40, 44])
- **Active Track Classes**: {'person': 1164, 'dog': 1, 'chair': 790, 'suitcase': 147, 'couch': 8, 'bed': 7, 'bench': 3}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 745, 'RECEDING': 460, 'STATIONARY': 0, 'STABLE': 848, 'UNKNOWN': 23}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 1190 instances
- **TTC Min / p50 / Max**: 0.44s / 4.22s / 30.00s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 790, 'WARNING': 298, 'CRITICAL': 69}
- **Navigation Decisions**: {'CONTINUE': 36, 'CAUTION': 565, 'AVOID_LEFT': 230, 'AVOID_RIGHT': 170, 'STOP': 157}
- **Recommended Safe Directions**: {'NONE': 758, 'LEFT': 230, 'RIGHT': 170}
- **Spoken Audio Warnings**: 29 utterances generated

