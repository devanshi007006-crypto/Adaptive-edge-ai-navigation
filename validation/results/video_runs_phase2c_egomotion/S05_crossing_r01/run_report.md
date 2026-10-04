# Phase 2C Benchmark Run Report: S05_crossing_r01

## 1. Video & Execution Overview
- **Scenario**: `crossing` (Dynamic orthogonal / transverse crossing pedestrian)
- **Video Path**: `validation/videos/crossing/S05_crossing_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Depth Cadence**: `2:1` (Depth Anything V2 executes every 2 frame(s))
- **Total Frames Processed**: 908
- **Wall-Clock Runtime**: 67.84 seconds
- **Effective Real FPS**: 13.38 FPS
- **Max VRAM Allocated**: 158.61 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 536.00 MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: 1907
- **Active Hazard Detections**: 1779
- **Policy-Filtered Detections**: 128
- **Raw Detected Classes**: {'person': 894, 'chair': 857, 'refrigerator': 127, 'suitcase': 1, 'dining table': 2, 'handbag': 23, 'bed': 2, 'fire hydrant': 1}
- **Active Hazard Classes Tracked**: {'person': 894, 'chair': 857, 'suitcase': 1, 'dining table': 2, 'handbag': 23, 'bed': 2}
- **Filtered Classes**: {'refrigerator': 127, 'fire hydrant': 1}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.33 | 9.71 | 14.44 |
| **BoT-SORT Tracker** | 4.80 | 4.75 | 5.23 |
| **Depth Anything V2 (vits, cadence 2:1)** | 48.12 | 0.00 | 98.37 |
| **Depth When Executing** | 96.35 ms | - | - |
| **TTC Estimation (Step 9)** | 0.03 | 0.03 | 0.03 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.01 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **72.84** | **34.95** | **125.09** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 11 (IDs: [1, 2, 3, 8, 18, 19, 25, 28, 30, 41, 44])
- **Active Track Classes**: {'person': 894, 'chair': 857, 'suitcase': 1, 'dining table': 2, 'handbag': 23, 'bed': 2}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 337, 'RECEDING': 607, 'STATIONARY': 0, 'STABLE': 770, 'UNKNOWN': 22}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 795 instances
- **TTC Min / p50 / Max**: 1.50s / 8.13s / 30.00s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 808, 'WARNING': 99, 'CRITICAL': 0}
- **Navigation Decisions**: {'CONTINUE': 9, 'CAUTION': 613, 'AVOID_LEFT': 286}
- **Recommended Safe Directions**: {'NONE': 622, 'LEFT': 286}
- **Spoken Audio Warnings**: 32 utterances generated

