# Phase 2B Behavioral Run Report: S02_static_r01

## 1. Video & Execution Overview
- **Scenario**: `static_obstacle` (Stationary obstacle in corridor with ego-camera approach)
- **Video Path**: `validation/videos/static_obstacle/S02_static_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 1158
- **Wall-Clock Runtime**: 145.55 seconds
- **Effective Real FPS**: 7.96 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

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
| **YOLO11n Object Detector** | 11.16 | 9.75 | 16.98 |
| **BoT-SORT Tracker** | 4.45 | 4.40 | 5.54 |
| **Depth Anything V2 (vits)** | 97.61 | 97.06 | 101.61 |
| **TTC Estimation (Step 9)** | 0.03 | 0.02 | 0.04 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.11 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **123.05** | **121.45** | **131.40** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 12 (IDs: [1, 6, 8, 9, 11, 18, 28, 30, 32, 39, 40, 44])
- **Active Track Classes**: {'person': 1164, 'dog': 1, 'chair': 790, 'suitcase': 147, 'couch': 8, 'bed': 7, 'bench': 3}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 776, 'RECEDING': 450, 'STATIONARY': 0, 'STABLE': 827, 'UNKNOWN': 23}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 1181 instances
- **TTC Min / p50 / Max**: 0.44s / 4.15s / 30.0s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 790, 'WARNING': 300, 'CRITICAL': 67}
- **Navigation Decisions**: {'CONTINUE': 36, 'CAUTION': 556, 'AVOID_LEFT': 230, 'AVOID_RIGHT': 179, 'STOP': 157}
- **Recommended Safe Directions**: {'NONE': 749, 'LEFT': 230, 'RIGHT': 179}
- **Spoken Audio Warnings**: 35 utterances generated
