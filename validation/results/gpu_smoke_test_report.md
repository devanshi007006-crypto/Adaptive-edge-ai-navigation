# Phase 1A.1 GPU Acceleration Report: NVIDIA GeForce RTX 4050 Laptop GPU

**Execution Timestamp:** 2026-10-04  
**Python Interpreter:** `c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\.venv\Scripts\python.exe`  
**PyTorch Version:** `2.10.0+cu130`  
**CUDA Version:** `13.0`  
**GPU Hardware:** `NVIDIA GeForce RTX 4050 Laptop GPU` (6.0 GB VRAM)  
**Input Clip:** `data/test_clip.mp4` (640×480 @ 30 FPS, 60 frames, 2.00 seconds)  
**Execution Command:**
```powershell
.\.venv\Scripts\python.exe adaptive_navigation/main.py --video data/test_clip.mp4 --headless --device cuda:0 --telemetry-dir validation/logs --telemetry-prefix gpu_telemetry
```

---

## 1. Executive Summary & Verification

Phase 1A.1 successfully transitioned the entire Adaptive Edge-AI Navigation pipeline from CPU fallback to full hardware-accelerated neural inference on the host's **NVIDIA GeForce RTX 4050 Laptop GPU**.

All 60 frames were processed with genuine neural inference, active BoT-SORT tracking, Depth Anything V2 monocular disparity mapping, camera ego-motion compensation, Time-to-Collision (TTC) estimation, multi-factor risk assessment, spatial path corridor steering, and real-time audio dispatch.

| Performance Metric | CPU Baseline (Phase 1A) | GPU Hardware (Phase 1A.1) | Acceleration Factor |
|---|---|---|---|
| **Python Environment** | Global Python 3.14 (CPU) | `.venv\Scripts\python.exe` (cu130) | Verified Virtualenv |
| **Active Inference Device** | `cpu` | `cuda:0` (RTX 4050 Laptop GPU) | GPU Accelerated |
| **YOLO11n Detector Latency** | Mean: 32.25 ms (p50: ~30 ms) | Mean: **18.57 ms** (p50: **9.59 ms**) | **1.7×** (3.4× at p50) |
| **Depth Anything V2 Latency**| Mean: 728.04 ms (p50: ~725 ms)| Mean: **70.49 ms** (p50: **69.25 ms**) | **10.3×** Speedup |
| **End-to-End Frame Latency** | Mean: 772.57 ms (p50: ~765 ms)| Mean: **101.04 ms** (p50: **90.34 ms**)| **7.6×** Overall Speedup |
| **Effective Pipeline FPS** | **1.29 FPS** | **9.90 FPS** (Steady-state: **11.07 FPS**) | **7.7×** Throughput Gain |
| **Total Clip Runtime (60f)** | 46.35 s | **6.06 s** | **7.6×** Faster |
| **GPU VRAM Utilization** | 0 MB (N/A) | **138.25 MB** (Allocated) / **438 MB** (Reserved) | < 7.2% of 6 GB VRAM |

---

## 2. Stage-by-Stage Latency Breakdown (GPU vs CPU)

All latencies are measured from actual wall-clock execution time via `time.perf_counter()` (zero `time.sleep()`, zero synthetic jitter):

| Subsystem / Pipeline Stage | GPU Mean (ms) | GPU p50 (ms) | GPU p95 (ms) | CPU Mean (ms) | Speedup Factor |
|---|---|---|---|---|---|
| **YOLO11n Detection** | 18.57 | 9.59 | 16.15 | 32.25 | **1.7×** (3.4× p50) |
| **BoT-SORT Tracking** | 3.61 | 3.44 | 4.08 | 4.09 | **1.1×** |
| **Depth Anything V2 (ViT-S)** | 70.49 | 69.25 | 76.20 | 728.04 | **10.3×** |
| **Temporal Motion Estimation** | 0.03 | 0.03 | 0.04 | 0.03 | 1.0× (CPU-bound) |
| **Camera Ego-Motion (LK Flow)**| 6.29 | 6.19 | 8.07 | 6.21 | 1.0× (OpenCV CPU) |
| **TTC Estimation** | 0.02 | 0.02 | 0.03 | 0.03 | 1.2× |
| **Multi-Factor Risk Engine** | 0.03 | 0.03 | 0.05 | 0.04 | 1.0× |
| **System Reliability Analysis** | 0.04 | 0.04 | 0.06 | 0.05 | 1.1× |
| **Warning State Machine** | 0.03 | 0.03 | 0.04 | 0.03 | 1.1× |
| **Spatial Navigation Engine** | 0.02 | 0.02 | 0.03 | 0.03 | 1.3× |
| **TTS Speech Dispatch** | 0.54 | 0.01 | 0.02 | 0.50 | 1.0× |
| **End-to-End Total Frame** | **101.04** | **90.34** | **106.75** | **772.57** | **7.6×** Overall |

---

## 3. Algorithmic & Safety Behavior Verification

The transition to CUDA preserved identical mathematical decisions, tracking trajectories, and risk scores:
- **Tracking Continuity:** Track ID 1 (`person`) tracked continuously across 100% of frames (60/60).
- **Depth Range:** Relative disparity values $[\hat{d}_{\min}, \hat{d}_{\max}] = [1.867, 2.480]$ with monotonic growth confirming visual looming.
- **TTC Estimations:** 37 valid kinematic TTC estimates calculated via scale-invariant optical divergence ($\tau = d / \dot{d}$) in range $[1.47\text{s}, 25.13\text{s}]$, median $= 4.94\text{s}$.
- **Warning Decisions:** `CAUTION`: 37 frames (61.7%), `WARNING`: 22 frames (36.7%), `NO_WARNING`: 1 frame (1.7%).
- **Navigation Actions:** `AVOID_LEFT`: 58 frames (96.7%), `CONTINUE`: 2 frames (3.3%).
- **Speech Audio Output:** 12 alerts dispatched via `pyttsx3` without blocking neural inference.

---

## 4. GPU Resource & VRAM Stability

- **Total Dedicated VRAM:** 6,144 MB (6.0 GB GDDR6)
- **VRAM Allocated (Tensors):** Constant **138.25 MB** across the entire 60-frame run.
- **VRAM Reserved (PyTorch Cache):** Constant **438.00 MB**.
- **Memory Leak Check:** Zero VRAM accumulation across frames. Total consumption remained under 7.2% of total GPU capacity, confirming the RTX 4050 has ample memory headroom for concurrent higher-resolution inference or metric scale estimators.

---

## 5. Artifact Traceability

Both CPU and GPU telemetry logs are preserved independently in `validation/logs/`:
1. **GPU Telemetry (Phase 1A.1):**
   - Hierarchical JSON: [`validation/logs/gpu_telemetry.json`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/gpu_telemetry.json)
   - Per-Frame CSV: [`validation/logs/gpu_telemetry_frames.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/gpu_telemetry_frames.csv)
   - Object-Level CSV: [`validation/logs/gpu_telemetry_objects.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/gpu_telemetry_objects.csv)
2. **CPU Telemetry (Phase 1A Baseline):**
   - Hierarchical JSON: [`validation/logs/real_telemetry.json`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/real_telemetry.json)
   - Per-Frame CSV: [`validation/logs/real_telemetry_frames.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/real_telemetry_frames.csv)
   - Object-Level CSV: [`validation/logs/real_telemetry_objects.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/logs/real_telemetry_objects.csv)
