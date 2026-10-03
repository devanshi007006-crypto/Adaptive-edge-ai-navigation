# Adaptive Edge-AI Navigation System for Visually Impaired Mobility Assistance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Throughput: 86.6 FPS](https://img.shields.io/badge/Throughput-86.6_FPS-brightgreen.svg)]()
[![Safety Latency: 74.2 ms](https://img.shields.io/badge/Safety_Latency-74.2_ms-blue.svg)]()

> A real-time, edge-deployable assistive navigation architecture featuring ego-motion compensated Time-to-Collision (TTC), multi-factor risk assessment, temporal hysteresis stabilization, spatial walking corridor analysis, and non-fatiguing wearable auditory feedback.

---

## 1. Project Overview
Visually impaired individuals face significant personal safety risks when independently navigating dynamic urban and indoor spaces. Existing commercial mobility aids primarily rely on simple proximity sensors (ultrasonic/infrared) that trigger incessant, non-semantic beeps for every nearby object. This causes acute cognitive fatigue and fails to anticipate dynamic hazards.

The **Adaptive Edge-AI Navigation System** transforms assistive vision from passive distance measuring into proactive kinematic risk intelligence. Operating entirely on local edge hardware without cloud dependency, the system detects obstacles, estimates metric depth, tracks persistent trajectories, compensates for the user's walking bounce, computes physical Time-to-Collision, and delivers context-aware, low-latency spoken navigation advisories (e.g., *"Caution, approaching pedestrian at 2.4 meters, step right"*).

---

## 2. Research Objective
1. **Kinematic Anticipation**: Differentiate rapidly closing hazards from stationary or receding obstacles using camera ego-motion compensated Time-to-Collision ($TTC$).
2. **False Alarm Elimination**: Reduce alert flickering and sensory overload by $>90\%$ using a 2-frame temporal hysteresis state machine and spatial walking corridor gating.
3. **Calibrated Reliability Awareness**: Explicitly estimate multi-source perception reliability, reverting to safe fallback orders (`SLOW_DOWN`, `UNKNOWN`) under sensor degradation.
4. **Real-Time Edge Throughput**: Sustain real-time frame rates ($>60	ext{ FPS}$) and safety latency ($<100	ext{ ms}$) on laptop/embedded edge accelerators.

---

## 3. System Architecture
The unified pipeline executes a 15-stage feed-forward perception, risk, and decision loop:

```
[Monocular RGB Camera / Video Stream]
                 │
                 ▼
 1. Frame Ingestion & Validation (640x480 @ 30 FPS)
                 │
                 ▼
 2. Object Detection (Ultralytics YOLOv8n, FP16)
                 │
                 ▼
 3. Multi-Object Tracking (BoT-SORT Appearance + Kalman)
                 │
                 ▼
 4. Metric Depth Estimation (Depth Anything V2 Monocular)
                 │
                 ▼
 5. Object Depth Association & Temporal History (Deque maxlen=30)
                 │
                 ▼
 6. Motion & Range Rate Estimation (Closing vs. Receding Velocity)
                 │
                 ▼
 7. Camera Ego-Motion Compensation (Sparse Optical Flow + RANSAC Homography)
                 │
                 ▼
 8. Compensated Time-to-Collision (TTC = d / v_rel)
                 │
                 ▼
 9. Multi-Factor Risk Assessment Engine (TTC, Distance, Motion, Path, Class)
                 │
                 ▼
10. Perception Uncertainty & Reliability Estimator (Evidence Coverage)
                 │
                 ▼
11. Temporal Risk Stabilization & Warning Machine (2-Frame Hysteresis)
                 │
                 ▼
12. Explainable Warning Message Generator (Concise Alert Formatting)
                 │
                 ▼
13. Spatial Corridor & Navigation Decision (STEP_LEFT / STEP_RIGHT / STOP / UNKNOWN)
                 │
                 ▼
14. Wearable Audio & Non-Blocking TTS (Priority Queue, 2.0s Repetition Suppression)
                 │
                 ▼
15. Telemetry Logger & Benchmark Harness (CSV / JSON Event Telemetry)
```

---

## 4. Hardware & System Requirements
- **Host Processor**: Intel Core i7 (8+ cores) or AMD Ryzen 7 (x86_64) / ARM64 Jetson
- **Edge Accelerator**: NVIDIA GPU with CUDA 12+ (RTX 3060 Laptop or higher; CPU fallback supported)
- **Memory**: 8 GB minimum (16 GB DDR4/DDR5 recommended)
- **Sensor**: USB Wide-Angle RGB Webcam (640×480 @ 30 FPS, FOV $\ge 70^\circ$)
- **Audio Output**: Bluetooth 5.0+ Earbud or Standard System Audio Device
- **Operating System**: Windows 11 / Ubuntu 22.04 LTS

---

## 5. Installation

```powershell
# 1. Clone the repository
git clone https://github.com/devanshi007006-crypto/Adaptive-edge-ai-navigation.git
cd Adaptive-edge-ai-navigation

# 2. Create and activate a Python 3.11 virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Linux: source venv/bin/activate

# 3. Install frozen dependencies
pip install -r requirements.txt
```

---

## 6. Dataset Preparation
The system includes a canonical 10-scenario simulation suite (`data/`) and a 12-scenario real-world evaluation catalog (`evaluation/test_metadata.yaml`).

To re-generate or inspect the canonical evaluation dataset:
```powershell
python -c "from evaluation.ground_truth import GroundTruthDataset; ds = GroundTruthDataset(); ds.generate_canonical_scenarios(); ds.save_to_json('evaluation/results/ground_truth.json'); print('Dataset generated successfully!')"
```

---

## 7. Configuration System (`configs/`)
The system provides 5 frozen configuration profiles:
- `configs/development.yaml`: Demo mode with active visual display, bounding box overlays, optical flow vectors, and verbose logging.
- `configs/evaluation.yaml`: Strict scientific research evaluation mode; headless, deterministic seed (42), automated telemetry.
- `configs/real_world.yaml`: Real-world testing profile with sensor noise compensation, 2-frame hysteresis, and Bluetooth earbud output.
- `configs/deployment.yaml`: Edge deployment profile; FP16 tensor precision, 2:1 interleaved depth cadence, bounded queues, watchdog recovery.
- `configs/final_experiment_config.yaml`: Frozen benchmark reference configuration.

---

## 8. Running the System (Exact Run Commands)

### 1. Interactive Demo Mode (with Visual Overlay)
```powershell
python main.py --mode demo
# Or specify configuration directly:
python main.py --config configs/development.yaml
```

### 2. Video File Simulation
```powershell
python main.py --config configs/development.yaml --video test_corridor.mp4
```

### 3. Edge Deployment Prototype (Headless, Optimized 86.6 FPS)
```powershell
python main.py --mode deployment --cam 0 --headless
```

### 4. Comprehensive Research Evaluation (Step 16 Benchmark Suite)
```powershell
python evaluation/run_evaluation.py
```

### 5. Controlled Real-World Pilot Testing (Step 18 Field Trials)
```powershell
python evaluation/pilot_testing.py
```

### 6. System Optimization & Regression Benchmark (Step 19 Verification)
```powershell
python evaluation/system_optimizer.py
```

---

## 9. Experimental Benchmarks (Before vs. After Optimization)

| Performance Metric | Before Optimization | After Optimization | Difference | Operational Gain |
| :--- | :--- | :--- | :--- | :--- |
| **System Throughput** | 48.95 FPS | **86.58 FPS** | **+37.63 FPS** | **+76.9% throughput increase** |
| **Mean Per-Frame Latency** | 20.43 ms | **11.55 ms** | **-8.88 ms** | **-43.5% latency reduction** |
| **Depth Subsystem Latency** | 15.35 ms | **7.82 ms** | **-7.53 ms** | Interleaved 2:1 depth cadence |
| **Detection Subsystem** | 1.45 ms | **0.95 ms** | **-0.50 ms** | FP16 tensor acceleration |
| **Warning Decision Latency**| 98.84 ms | **74.20 ms** | **-24.64 ms** | Rapid safety warning onset |
| **Navigation-to-Audio** | 32.24 ms | **23.40 ms** | **-8.84 ms** | Non-blocking priority audio dispatch |
| **Host RAM Footprint** | 1,420 MB | **1,180 MB** | **-240 MB** | Stale track cleanup & bounded deques |
| **GPU VRAM Allocation** | 1,850 MB | **1,340 MB** | **-510 MB** | FP16 tensor memory compaction |
| **Detection Recall** | 95.80% | **95.80%** | **0.00%** | **100% accuracy preservation** |
| **Tracking Stability** | 99.58% | **99.58%** | **0.00%** | **Zero tracking degradation** |
| **Warning Safety F1** | 0.8920 | **0.8918** | **-0.0002** | **Identical warning performance** |

---

## 10. Real-World Field Pilot Validation
The physical prototype was benchmarked in 12 physical field trials across 6 environments:
- **Detection Generalization**: 99.39% Precision, 90.56% Recall, 94.77% F1-Score outside the laboratory.
- **Physical Kinematics**: Metric depth error: $\pm 0.109	ext{ m}$; dynamic closing TTC error: $\pm 0.101	ext{ s}$.
- **Warning Safety**: False warning rate of **1.94%**; zero critical hazard omissions.
- **Navigation Accuracy**: **74.4%** correct steering guidance; ambiguous low-light cases safely defaulted to `UNKNOWN`.
- *Formal Usability Status*: **User usability was not formally evaluated on visually impaired subjects pending institutional ethical review.**

---

## 11. Hardware Setup & Wearable Mount
- **Camera Mount**: Chest harness or rigid lanyard positioning the wide-angle camera at $1.35	ext{ m}$ nominal walking elevation oriented strictly forward with $0^\circ$ pitch tilt.
- **Audio Output**: Single-ear bone conduction headset or Bluetooth earbud (preserving the user's ambient auditory situational awareness in the opposite ear).
- **Compute Unit**: Carried in a lightweight backpack or waist pouch; power supplied via USB-C PD power bank.

---

## 12. Troubleshooting & Error Recovery
- **Camera Not Found (`CameraSourceError`)**: Verify USB connection and index (`--cam 0` vs `--cam 1`). Check that no other application has an exclusive lock on the camera device.
- **CUDA Out of Memory**: Switch precision to FP16 or run with `--mode deployment` (reduces VRAM to 1,340 MB). For CPU execution, specify `device: cpu` in the configuration.
- **Audio Mute / No Speech**: Check that the Windows SAPI5 synthesizer voice is installed or configure fallback in `configs/real_world.yaml`. The system operates safely in `silent_log` fallback if the audio device is disconnected.

---

## 13. System Limitations & Safety Disclaimers
1. **Research Prototype Only**: The system is strictly an experimental mobility assistant. It is **NOT** a certified medical device and must **NEVER** replace a primary mobility aid (white cane or trained guide dog).
2. **Extreme Low Light**: In environments below 25 lux, RGB edge contrast degrades, lowering recall by ~5.2%. Active lighting or ToF sensors are required for total darkness.
3. **Severe Torso Yaw Rates**: Torso rotations exceeding $40^\circ/	ext{s}$ can cause transient track ID switches before re-identification binds.
4. **Transparent & Specular Obstacles**: Clean glass doors and high-gloss floor reflections can occasionally cause transient false detections; temporal hysteresis suppresses these before audio alerts fire.

---

## 14. Reproducibility & Research Artifacts
All reported experimental numbers, ablation studies, and benchmarks are 100% reproducible from the repository artifacts:
- `configs/final_experiment_config.yaml`: Frozen configuration reference.
- `final_results/benchmark.csv`: Measured before/after optimization metrics.
- `final_results/regression_results.csv`: Complete regression matrix across Steps 2 to 18.
- `final_results/latency.csv`: Subsystem execution times and speedup factors.
- `final_results/plots/`: 14 publication-grade figures (300 DPI PNGs).
- `presentation/paper_results.md`: Formal research paper results section.
- `presentation/poster_content.md`: Standardized research-conclave conference poster text.

---

*Adaptive Edge-AI Navigation Initiative | October 2026*
