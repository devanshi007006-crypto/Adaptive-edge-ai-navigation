# Phase 2B Behavioral Run Report: S03_approaching_r01

## 1. Video & Execution Overview
- **Scenario**: `approaching` (Dynamic approaching obstacle / pedestrian directly closing in)
- **Video Path**: `validation/videos/approaching/S03_approaching_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 788
- **Wall-Clock Runtime**: 100.11 seconds
- **Effective Real FPS**: 7.87 FPS
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
| **YOLO11n Object Detector** | 11.57 | 9.75 | 14.98 |
| **BoT-SORT Tracker** | 4.08 | 3.93 | 5.01 |
| **Depth Anything V2 (vits)** | 97.97 | 96.91 | 104.05 |
| **TTC Estimation (Step 9)** | 0.02 | 0.02 | 0.03 |
| **Multi-Factor Risk Engine** | 0.03 | 0.03 | 0.05 |
| **Spatial Navigation Engine** | 0.02 | 0.02 | 0.03 |
| **User-Facing Audio / TTS Dispatch** | 0.14 | 0.01 | 0.02 |
| **End-to-End Per-Frame Pipeline** | **123.42** | **120.81** | **132.20** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 5 (IDs: [1, 6, 7, 8, 11])
- **Active Track Classes**: {'person': 789, 'chair': 55, 'dining table': 24}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 601, 'RECEDING': 106, 'STATIONARY': 0, 'STABLE': 136, 'UNKNOWN': 9}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 615 instances
- **TTC Min / p50 / Max**: 0.66s / 4.08s / 30.0s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1, 'CAUTION': 340, 'WARNING': 390, 'CRITICAL': 57}
- **Navigation Decisions**: {'CONTINUE': 3, 'CAUTION': 254, 'AVOID_LEFT': 482, 'AVOID_RIGHT': 49}
- **Recommended Safe Directions**: {'NONE': 257, 'LEFT': 482, 'RIGHT': 49}
- **Spoken Audio Warnings**: 33 utterances generated
