# Phase 2B Behavioral Run Report: S01_clear_r01

## 1. Video & Execution Overview
- **Scenario**: `clear_path` (Negative control: clear unobstructed corridor traversal)
- **Video Path**: `validation/videos/clear_path/S01_clear_r01.mp4`
- **Execution Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Total Frames Processed**: 1524
- **Wall-Clock Runtime**: 205.59 seconds
- **Effective Real FPS**: 7.41 FPS
- **Max VRAM Allocated**: 138.24 MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: 518.00 MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: 341
- **Active Hazard Detections**: 13
- **Policy-Filtered Detections**: 328
- **Raw Detected Classes**: {'refrigerator': 90, 'toilet': 84, 'bed': 3, 'tv': 14, 'cat': 118, 'person': 1, 'potted plant': 15, 'dog': 7, 'chair': 2, 'microwave': 7}
- **Active Hazard Classes Tracked**: {'bed': 3, 'person': 1, 'dog': 7, 'chair': 2}
- **Filtered Classes**: {'refrigerator': 90, 'toilet': 84, 'tv': 14, 'cat': 118, 'potted plant': 15, 'microwave': 7}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | 13.07 | 11.66 | 20.11 |
| **BoT-SORT Tracker** | 4.99 | 4.51 | 7.46 |
| **Depth Anything V2 (vits)** | 103.21 | 101.98 | 115.27 |
| **TTC Estimation (Step 9)** | 0.00 | 0.00 | 0.01 |
| **Multi-Factor Risk Engine** | 0.00 | 0.00 | 0.01 |
| **Spatial Navigation Engine** | 0.01 | 0.01 | 0.02 |
| **User-Facing Audio / TTS Dispatch** | 0.01 | 0.01 | 0.01 |
| **End-to-End Per-Frame Pipeline** | **133.04** | **130.81** | **152.99** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: 1 (IDs: [10])
- **Active Track Classes**: {'bed': 3, 'person': 1, 'dog': 7, 'chair': 2}

### B. Depth & Motion
- **Approach State Distribution**: {'APPROACHING': 0, 'RECEDING': 0, 'STATIONARY': 0, 'STABLE': 0, 'UNKNOWN': 1}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: 0 instances
- **TTC Min / p50 / Max**: Nones / Nones / Nones (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {'NO_WARNING': 1524, 'CAUTION': 0, 'WARNING': 0, 'CRITICAL': 0}
- **Navigation Decisions**: {'CONTINUE': 1524}
- **Recommended Safe Directions**: {'NONE': 1524}
- **Spoken Audio Warnings**: 0 utterances generated
