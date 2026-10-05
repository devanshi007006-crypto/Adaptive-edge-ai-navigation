# Live Demo Control Center — Operator & Reproduction Guide

**Component:** Phase 5.1 Live Demo Control Center  
**Script Location:** [`scripts/run/run_demo_control_center.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/scripts/run/run_demo_control_center.py)  
**Date:** October 5, 2026  
**Target Hardware:** Laptop Integrated Webcam (`cam 0`) + NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`)  

---

## 1. Overview & Purpose

The **Live Demo Control Center** is an experiment-management desktop GUI application (Tkinter + OpenCV) designed for executing, recording, tagging, and reviewing the six controlled live-camera demonstration scenarios (`S01`–`S06`). 

It wraps the existing production perception, tracking, depth, TTC, risk, navigation, and TTS engines without duplicating neural network architectures or decision logic.

---

## 2. Launching the Control Center

Launch the desktop GUI from the terminal:

```powershell
python scripts/run/run_demo_control_center.py
```

### Startup Sequence
1. The GUI window opens in initializing state.
2. Background thread provisions PyTorch CUDA device (`cuda:0`), loads YOLO11n object detector, BoT-SORT multi-object tracker, and verifies native TensorRT FP16 depth engine (`models/deployment/depth_anything_v2_vits_fp16.engine`).
3. Once initialized, system status indicators update (GPU: `RTX 4050`, Engine: `TRT FP16 Active`), the video feed activates, and the `[ ▶ START TRIAL ]` button is enabled.

---

## 3. The Six Controlled Scenarios

| Scenario ID | Scenario Name | Target Setup & Physical Protocol | Expected System Response |
| :--- | :--- | :--- | :--- |
| **S01** | **Clear Path** | Clear walking corridor with zero physical obstructions or pedestrians. | Primarily `NO_WARNING` warning state and `CONTINUE` navigation action. |
| **S02** | **Static Obstacle** | Place a stationary chair, box, or obstacle 2–3 m ahead in the walking corridor. | Object detection, proximity zone placement, and safe directional avoidance steering (`AVOID_LEFT` / `AVOID_RIGHT`). |
| **S03** | **Person Approaching** | Subject stands 8–10 m away and walks towards camera at controlled speed. | `APPROACHING` motion state, decreasing TTC, escalating risk (`WARNING`/`CRITICAL`), and spoken audio alert. |
| **S04** | **Person Receding** | Subject stands near camera (2 m) and walks away along the corridor. | `RECEDING` motion state, suppressed closing risk, and quiet `NO_WARNING` / `CONTINUE` state. |
| **S05** | **Person Crossing** | Subject walks laterally across the walking corridor at 3–4 m distance. | Lateral motion tracking, corridor overlap evaluation, and path-aware steering response. |
| **S06** | **Head / Camera Motion** | Operator performs deliberate head panning and body walking sway (>30°/s). | Optical-flow ego-motion compensation absorbs gait/pan jitter, maintaining target tracking. |

---

## 4. Session Directory & Output Structure

Every demonstration session creates a unique timestamped session directory:

```
validation/results/live/demo_sessions/
└── YYYY-MM-DD_HH-MM-SS/
    ├── S01_clear_path/
    │   ├── scenario_video.mp4
    │   ├── telemetry.json
    │   ├── event_markers.json
    │   ├── trial_metrics.json
    │   └── trial_report.md
    ├── S02_static_obstacle/
    ├── S03_approaching/
    ├── S04_receding/
    ├── S05_crossing/
    ├── S06_head_motion/
    ├── screenshots/
    ├── session_summary.csv
    ├── session_summary.json
    └── FINAL_DEMO_REPORT.md
```

---

## 5. Event Marker Logging

During an active trial, the operator can log timestamped event markers to trace physical events against telemetry:

- **Supported Tags:**
  - `Hazard Begins`
  - `Subject Starts Moving`
  - `Subject Stops`
  - `Crossing Begins`
  - `Head Motion Begins`
  - `System Warning`
  - `System Failure`
  - `Manual Note`
- Every marker records precise system timestamp, formatted clock time, scenario ID, current trial frame index, tag, and optional text note.

---

## 6. Safety & Intended Use Restrictions

> [!CAUTION]
> **SAFETY NOTICE**:
> 1. This application is designed strictly for **controlled indoor, open-eye, supervised technical demonstrations**.
> 2. **NO BLINDFOLDED TESTING**: Blindfolded testing or unassisted navigation by visually impaired users is strictly prohibited pending formal institutional ethical review and certified technical safety qualification.
> 3. Do NOT describe or present this software as a certified electronic travel aid (ETA) or medical device.
