# Phase 2B Behavioral Run Report: S05_crossing_r01

## 1. Video & Execution Overview
- **Scenario**: `crossing` (Dynamic orthogonal / transverse crossing pedestrian)
- **Video Path**: `validation/videos/crossing/S05_crossing_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 908
- **Wall-Clock Runtime**: 125.26 seconds
- **Effective Real FPS**: 7.25 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

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
| **YOLO11n Object Detector** | 13.75 | 11.86 | 20.12 |
| **BoT-SORT Tracker** | 5.98 | 5.42 | 8.52 |
| **Depth Anything V2 (vits)** | 101.51 | 100.47 | 108.47 |
| **TTC Estimation (Step 9)** | 0.03 | 0.03 | 0.05 |
| **Multi-Factor Risk Engine** | 0.04 | 0.04 | 0.07 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.06 |
| **User-Facing Audio / TTS Dispatch** | 0.07 | 0.01 | 0.02 |
| **End-to-End Per-Frame Pipeline** | **133.93** | **132.11** | **147.52** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 11 (IDs: [1, 2, 3, 8, 18, 19, 25, 28, 30, 41, 44])
- **Active Track Classes**: {'person': 894, 'chair': 857, 'suitcase': 1, 'dining table': 2, 'handbag': 23, 'bed': 2}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 386, 'RECEDING': 628, 'STATIONARY': 0, 'STABLE': 700, 'UNKNOWN': 22}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 766 instances
- **TTC Min / p50 / Max**: 1.48s / 7.88s / 30.0s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 796, 'WARNING': 111, 'CRITICAL': 0}
- **Navigation Decisions**: {'CONTINUE': 16, 'CAUTION': 611, 'AVOID_LEFT': 281}
- **Recommended Safe Directions**: {'NONE': 627, 'LEFT': 281}
- **Spoken Audio Warnings**: 18 utterances generated
