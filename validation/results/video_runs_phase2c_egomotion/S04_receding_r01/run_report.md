# Phase 2C Benchmark Run Report: S04_receding_r01

## 1. Video & Execution Overview
- **Scenario**: `receding` (Object moving away / increasing distance from observer)
- **Video Path**: `validation/videos/receding/S04_receding_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Depth Cadence**: `2:1` (Depth Anything V2 executes every 2 frame(s))
- **Total Frames Processed**: 857
- **Wall-Clock Runtime**: 62.92 seconds
- **Effective Real FPS**: 13.62 FPS
- **Max VRAM Allocated**: 148.42 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 536.00 MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: 1619
- **Active Hazard Detections**: 1597
- **Policy-Filtered Detections**: 22
- **Raw Detected Classes**: {'person': 860, 'cell phone': 5, 'skis': 1, 'chair': 614, 'refrigerator': 13, 'suitcase': 118, 'bench': 5, 'cat': 3}
- **Active Hazard Classes Tracked**: {'person': 860, 'chair': 614, 'suitcase': 118, 'bench': 5}
- **Filtered Classes**: {'cell phone': 5, 'skis': 1, 'refrigerator': 13, 'cat': 3}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.38 | 9.70 | 14.67 |
| **BoT-SORT Tracker** | 4.33 | 4.36 | 4.98 |
| **Depth Anything V2 (vits, cadence 2:1)** | 47.77 | 46.60 | 96.88 |
| **Depth When Executing** | 95.55 ms | - | - |
| **TTC Estimation (Step 9)** | 0.02 | 0.02 | 0.04 |
| **Multi-Factor Risk Engine** | 0.04 | 0.03 | 0.05 |
| **Spatial Navigation Engine** | 0.03 | 0.03 | 0.04 |
| **User-Facing Audio / TTS Dispatch** | 0.01 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **71.49** | **74.11** | **122.83** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 7 (IDs: [1, 7, 10, 12, 15, 20, 24])
- **Active Track Classes**: {'person': 860, 'chair': 614, 'suitcase': 118, 'bench': 5}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 454, 'RECEDING': 493, 'STATIONARY': 0, 'STABLE': 613, 'UNKNOWN': 14}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 866 instances
- **TTC Min / p50 / Max**: 0.76s / 4.20s / 30.00s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 3, 'CAUTION': 694, 'WARNING': 129, 'CRITICAL': 31}
- **Navigation Decisions**: {'CONTINUE': 2, 'AVOID_LEFT': 464, 'AVOID_RIGHT': 203, 'STOP': 188}
- **Recommended Safe Directions**: {'NONE': 190, 'LEFT': 464, 'RIGHT': 203}
- **Spoken Audio Warnings**: 3 utterances generated

