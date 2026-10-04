# Phase 2C Benchmark Run Report: S03_approaching_r01

## 1. Video & Execution Overview
- **Scenario**: `approaching` (Dynamic approaching obstacle / pedestrian directly closing in)
- **Video Path**: `validation/videos/approaching/S03_approaching_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Depth Cadence**: `2:1` (Depth Anything V2 executes every 2 frame(s))
- **Total Frames Processed**: 788
- **Wall-Clock Runtime**: 60.13 seconds
- **Effective Real FPS**: 13.10 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: 901
- **Active Hazard Detections**: 868
- **Policy-Filtered Detections**: 33
- **Raw Detected Classes**: {'person': 789, 'chair': 55, 'dining table': 24, 'toilet': 25, 'cell phone': 8}
- **Active Hazard Classes Tracked**: {'person': 789, 'chair': 55, 'dining table': 24}
- **Filtered Classes**: {'toilet': 25, 'cell phone': 8}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 10.59 | 9.80 | 15.68 |
| **BoT-SORT Tracker** | 3.95 | 3.91 | 4.90 |
| **Depth Anything V2 (vits, cadence 2:1)** | 48.19 | 0.00 | 98.55 |
| **Depth When Executing** | 96.65 ms | - | - |
| **TTC Estimation (Step 9)** | 0.02 | 0.02 | 0.03 |
| **Multi-Factor Risk Engine** | 0.03 | 0.03 | 0.04 |
| **Spatial Navigation Engine** | 0.02 | 0.02 | 0.03 |
| **User-Facing Audio / TTS Dispatch** | 0.58 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **72.44** | **33.51** | **125.57** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 5 (IDs: [1, 6, 7, 8, 11])
- **Active Track Classes**: {'person': 789, 'chair': 55, 'dining table': 24}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 550, 'RECEDING': 120, 'STATIONARY': 0, 'STABLE': 173, 'UNKNOWN': 9}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 600 instances
- **TTC Min / p50 / Max**: 0.73s / 3.94s / 30.00s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 325, 'WARNING': 426, 'CRITICAL': 36}
- **Navigation Decisions**: {'CONTINUE': 3, 'CAUTION': 246, 'AVOID_LEFT': 489, 'AVOID_RIGHT': 50}
- **Recommended Safe Directions**: {'NONE': 249, 'LEFT': 489, 'RIGHT': 50}
- **Spoken Audio Warnings**: 25 utterances generated

