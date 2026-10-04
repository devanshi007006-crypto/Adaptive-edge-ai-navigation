# Project Index: Adaptive Multimodal Edge-AI Navigation

> **PERMANENT SOURCE OF TRUTH**: This file MUST remain in the project root for the entire lifetime of the repository. It serves as the primary master database for file locations, architecture, operational status, metric traceability, and experimental roadmaps. It must be updated whenever files are created, moved, renamed, deprecated, or modified.

---

## 1. Project Identity

* **Project Title**: An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users
* **Short Name**: `Adaptive-edge-ai-navigation`
* **Current Development Stage**: **Phase 3A** — HEADS-UP Exploratory External Validation
* **Current Validation Stage**: **Exploratory external validation on three representative HEADS-UP sequences (250 frames).** Evaluated head-mounted video behavior on RTX 4050 (`cuda:0`). Measured 7.08–10.44 FPS real throughput, 98.7% optical flow ego-motion tracking validity (up to 100 px/frame displacement), 1,288 raw detector proposals filtered down to 941 active detections, zero active false-warning detections after the configured indoor navigation class policy on the selected sequences, 100% track persistence on primary hazards, and context-aware warning/evasion escalation. Not a statistically representative sample of the full 43,213-frame benchmark.
* **Primary Repository**: `Adaptive-edge-ai-navigation`
* **Active Git Branch**: `By-Vivek-for-testing-and-improvement`
* **Target Hardware**: Intel Core i7 / NVIDIA RTX Laptop GPU / Edge Jetson ARM64 / Host CPU fallback
* **Operating Environment**: Python 3.11–3.14 (Windows / Linux x86_64)

---

## 2. Current System Status

### What Currently Works
- **Adaptive 2:1 Depth Cadence (13–14 FPS Throughput)**: Subsampled depth inference in [`adaptive_navigation/main.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/main.py) running Depth Anything V2 every 2nd frame with bounding box depth propagation, accelerating end-to-end pipeline throughput from 7.8 FPS to 13.4 FPS (+68.7% speedup) at 138.24 MB peak VRAM.
- **Forward Ego-Motion Radial Divergence Compensation**: Background optical flow expansion divergence $\gamma_{\text{bg}}$ estimation in [`adaptive_navigation/temporal/camera_motion.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/temporal/camera_motion.py), absorbing gait oscillations and forward camera approach artifacts while preserving genuine closing threats.
- **Indoor Navigation Class Policy**: Configuration-driven filtering in [`adaptive_navigation/perception/detector.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/perception/detector.py) suppressing architectural COCO hallucinations (`toilet`, `cat`, `refrigerator`, `microwave`) while preserving all genuine obstacles (`person`, `chair`, `dining table`, `suitcase`) and retaining raw detections in telemetry.
- **Multi-Frame Kinematic Stabilization & Hysteresis**: Least-squares regression derivative over sliding window, 5-frame consecutive confirmation hysteresis, and optical bounding box area shrinkage cross-validation in [`adaptive_navigation/temporal/motion.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/temporal/motion.py).
- **Scale-Invariant Optical Divergence TTC**: Robust closing velocity and time-to-collision calculation $\tau = d / \dot{d}$ in [`adaptive_navigation/risk/ttc.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/risk/ttc.py) supporting explicit inverse-disparity conventions.
- **Real-World GPU Inference Pipeline**: YOLO11n + BoT-SORT + Depth Anything V2 + Risk + Navigation + TTS running on RTX 4050 (`cuda:0`).
- **Structured Real-Time Telemetry Logging**: Dual CSV (`telemetry_frames.csv`, `telemetry_objects.csv`) and JSON (`telemetry.json`) streaming per-frame GPU timing, bounding boxes, disparity values, TTC, risk scores, and nav commands.
- **Spatial Walking Corridor Analysis**: Lateral corridor partitioning (center 40% / $\pm 0.6\text{m}$) and free-space clearance estimation in [`adaptive_navigation/navigation/spatial.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/navigation/spatial.py).
- **Temporal Warning State Machine**: Multi-factor hazard escalation hysteresis, de-escalation dampening, and multi-track alert prioritization in [`adaptive_navigation/warning/state_machine.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/warning/state_machine.py).

### What Is Partially Implemented
- **Monocular Relative Disparity**: Depth Anything V2 outputs relative inverse depth $[0, 1]$; scale-invariant TTC works, but metric distance (meters) requires ground-plane calibration.
- **6-DoF Visual Odometry**: 2D translation and radial expansion are compensated, but 3D pitch/yaw rotation requires IMU fusion.

### What Is Unverified
- **Closed-Loop Human Walking Trials**: Evasive steering audio cues (`AVOID_LEFT`, `AVOID_RIGHT`, `STOP`) have been validated on recorded video streams, but not yet tested on live human subjects or blindfolded navigators.

### What Is Deprecated / Historical
- **Legacy Synthetic Scripts**: Scripts under `evaluation/` (`run_evaluation.py`, `pilot_testing.py`) and historical reports (`FINAL_VALIDATION_SUMMARY.md`) used simulated noise; superseded by real telemetry in `validation/results/`.

### What Is Missing
- **Ground-Truth Benchmark Datasets**: Standardized benchmark dataset cache (HEADS-UP) for cross-dataset generalization.

---

## 3. Master Architecture

The canonical feed-forward perception and decision pipeline operates through 15 sequential stages:

```
[Camera Source / Video File / Synthetic Stream]
                   │
                   ▼
 1. Frame Ingestion & Validation (CameraSource, FramePreprocessor)
                   │
                   ▼
 2. Object Detection (YOLOObjectDetector: YOLO11n / YOLOv8n)
                   │
                   ▼
 3. Multi-Object Tracking (BoTSORTTracker: Kalman Filter + Re-ID)
                   │
                   ▼
 4. Monocular Depth Estimation (DepthAnythingV2Estimator: ViT-S)
                   │
                   ▼
 5. Object Depth Extraction & Temporal History (TemporalHistory: Deque maxlen=30)
                   │
                   ▼
 6. Motion & Range Rate Estimation (MotionEstimator: Image Velocity & Depth Rate)
                   │
                   ▼
 7. Camera Ego-Motion Compensation (CameraMotionEstimator: Sparse LK Flow + RANSAC)
                   │
                   ▼
 8. Kinematic Time-to-Collision (TTCEstimator: TTC = d / v_closing)
                   │
                   ▼
 9. Multi-Factor Risk Assessment Engine (RiskEngine: 5-Feature Fused Score)
                   │
                   ▼
10. Uncertainty & Reliability Calibration (ReliabilityEstimator: Evidence Coverage)
                   │
                   ▼
11. Temporal Risk Stabilization (WarningStateMachine: 2-Frame Escalation Hysteresis)
                   │
                   ▼
12. Explainable Warning Message Generation (WarningMessageGenerator: Spoken Text)
                   │
                   ▼
13. Spatial Corridor & Navigation Decision (NavigationEngine: Avoidance Steering)
                   │
                   ▼
14. Audio & Speech Dispatch (DeviceManager, TTSEngine: Priority Queue)
                   │
                   ▼
15. Telemetry Logging & Offline Evaluation (PerFrameLogger -> MetricsEvaluator)
```

---

## 4. Root Directory Map

| Path | Purpose | Status |
| :--- | :--- | :--- |
| `PROJECT_INDEX.md` | Master project database and permanent source of truth | **ACTIVE** |
| `README.md` | General project overview, hardware specs, and CLI instructions | **ACTIVE** |
| `LICENSE` | MIT Open-Source Research License | **ACTIVE** |
| `requirements.txt` | Python dependency specifications | **ACTIVE** |
| `pyproject.toml` | Project build system and static typing configuration | **ACTIVE** |
| `pyrefly.toml` | Pyrefly language server configuration | **ACTIVE** |
| `main.py` | Unified root CLI application entry point | **ACTIVE** |
| `audit_report.md` | Comprehensive scientific and engineering audit report | **ACTIVE** |
| `metric_traceability.md`| Complete provenance tracing of all reported metrics | **ACTIVE** |
| `evaluation_plan.md` | Scientific validation architecture and evaluation plan | **ACTIVE** |
| `src/` | Standardized modular source namespace (aliases `adaptive_navigation`) | **ACTIVE** |
| `adaptive_navigation/` | Canonical core package containing all 15 pipeline subsystems | **ACTIVE** |
| `configs/` | Runtime configuration profiles (final, dev, eval, real-world, deploy) | **ACTIVE** |
| `validation/` | Clean scientific validation framework for Modes A, B, and C | **ACTIVE** |
| `models/` | Storage for neural network checkpoint weights | **ACTIVE** (Weights Missing) |
| `scripts/` | Execution, diagnostic, and setup utility scripts | **ACTIVE** |
| `tests/` | Unit, integration, and import regression tests | **ACTIVE** |
| `docs/` | Research reports, dataset cards, model cards, and methodology notes | **ACTIVE** |
| `data/` | Local raw sample videos and testing inputs | **ACTIVE** |
| `outputs/` | Runtime telemetry logs, demo snapshots, and temporary artifacts | **ACTIVE** |
| `logs/` | Subsystem execution logs | **ACTIVE** |
| `archive/` | Storage for legacy, deprecated, or superseded prototype files | **LEGACY / ARCHIVED** |
| `evaluation/` | Duplicate legacy evaluation scripts (contains synthetic generators) | **DEPRECATED / FABRICATED** |
| `final_results/` | Legacy benchmark CSV tables and synthetic evaluation plots | **DEPRECATED / GENERATED** |
| `presentation/` | Presentation slides, poster text, and publication figures | **LEGACY / ARTIFACT** |
| `poster/` | Academic conference poster text and figures | **LEGACY / ARTIFACT** |
| `results/` | Legacy empty results placeholder | **LEGACY** |

---

## 5. Complete File Index

### Root Files
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PROJECT_INDEX.md` | Root | Master project index & single source of truth | None | Developer / Agents | **ACTIVE** |
| `README.md` | Root Docs | Repository overview and usage documentation | None | External Users | **ACTIVE** |
| `LICENSE` | Root Meta | MIT License terms | None | Project | **ACTIVE** |
| `requirements.txt` | Build | Core Python dependencies | None | pip | **ACTIVE** |
| `pyproject.toml` | Build | Packaging and static checker configuration | setuptools | Build Tools | **ACTIVE** |
| `pyrefly.toml` | Config | Language server type checking config | None | IDE / Pyrefly | **ACTIVE** |
| `main.py` | Entry Point | Unified application entry point with CLI options | `adaptive_navigation.main` | User CLI | **ACTIVE** |
| `audit_report.md` | Research | Forensic audit of scientific integrity & codebase | None | Research Conclave | **ACTIVE** |
| `metric_traceability.md`| Research | Backward trace of every reported metric to source | None | Research Conclave | **ACTIVE** |
| `evaluation_plan.md` | Research | Triad validation architecture (Synthetic/HEADS-UP/Real) | None | Research Conclave | **ACTIVE** |
| `FINAL_VALIDATION_SUMMARY.md`| Generated | Legacy report claiming 86.6 FPS & field trials | Synthetic scripts | Legacy Docs | **DEPRECATED / FABRICATED** |
| `final_validation_matrix.csv`| Generated | Legacy subsystem verification matrix | Synthetic scripts | Legacy Docs | **DEPRECATED / FABRICATED** |
| `regression_results.csv`| Generated | Legacy latency and regression numbers | Synthetic scripts | Legacy Docs | **DEPRECATED / FABRICATED** |

### Core Source Package (`adaptive_navigation/` & `src/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `adaptive_navigation/__init__.py` | Core | Package root; registers path for intra-package imports | `sys`, `os` | All modules | **ACTIVE** |
| `adaptive_navigation/main.py` | Core Pipeline | Procedural 15-stage perception pipeline loop | All subsystems | `main.py`, CLI | **ACTIVE** |
| `adaptive_navigation/config.yaml` | Config | Default package-level configuration | YAML | `main.py` | **ACTIVE** |
| `adaptive_navigation/audio/__init__.py` | Audio | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/audio/tts.py` | Audio | Offline TTS speech synthesizer engine (`pyttsx3`) | `pyttsx3`, `threading` | `main.py`, `DeviceManager`| **ACTIVE** |
| `adaptive_navigation/hardware/__init__.py` | Hardware | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/hardware/audio_device.py` | Hardware | Abstract and concrete audio hardware devices | `tts.py`, `threading` | `device_manager.py` | **ACTIVE** |
| `adaptive_navigation/hardware/device_manager.py` | Hardware | Audio device supervisor and system health watchdog | `audio_device.py` | `main.py` | **ACTIVE** |
| `adaptive_navigation/motion/__init__.py` | Motion | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/motion/camera_motion.py` | Motion | Re-export of CameraMotionEstimator from temporal | `temporal.camera_motion`| Package consumers | **ACTIVE** |
| `adaptive_navigation/navigation/__init__.py` | Navigation | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/navigation/spatial.py` | Navigation | 2D bounding-box spatial geometry & clearance analysis| `dataclasses` | `navigation_decision.py`| **ACTIVE** |
| `adaptive_navigation/navigation/path_geometry.py` | Navigation | Spatial walking corridor intersection & overlap analysis| `spatial.py` | `navigation_decision.py`| **ACTIVE** |
| `adaptive_navigation/navigation/navigation_decision.py`| Navigation | Safe evasive steering engine (AVOID_LEFT/RIGHT/STOP)| `path_geometry.py` | `main.py` | **ACTIVE** |
| `adaptive_navigation/navigation/decision.py` | Navigation | Early prototype decision stub (superseded) | `zones.py`, `free_space.py`| None | **LEGACY** |
| `adaptive_navigation/navigation/free_space.py` | Navigation | Early prototype free space stub (superseded) | `enum` | `decision.py` | **LEGACY** |
| `adaptive_navigation/navigation/zones.py` | Navigation | Early prototype horizontal zone divider stub | `risk.score` | `decision.py` | **LEGACY** |
| `adaptive_navigation/perception/__init__.py` | Perception | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/camera.py` | Perception | OpenCV camera & video stream capture source | `cv2`, `threading` | `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/preprocessing.py`| Perception | Frame validation, resizing, and normalization | `cv2`, `numpy` | `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/detector.py` | Perception | Ultralytics YOLO object detector wrapper | `ultralytics`, `torch` | `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/tracker.py` | Perception | BoT-SORT multi-object tracker adapter | `ultralytics`, `detector.py`| `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/depth.py` | Perception | Depth Anything V2 monocular depth & ROI extraction | `dpt.py`, `cv2`, `torch` | `main.py` | **ACTIVE** |
| `adaptive_navigation/perception/depth_anything_v2/`| Perception | Official DPT & Dinov2 model backbone code | `torch`, `torch.nn` | `depth.py` | **ACTIVE** |
| `adaptive_navigation/risk/__init__.py` | Risk | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/risk/ttc.py` | Risk | Kinematic Time-to-Collision estimator ($TTC = d/v$) | `temporal.history` | `main.py`, `risk_engine` | **ACTIVE (Has Bug)** |
| `adaptive_navigation/risk/risk_engine.py` | Risk | Multi-factor risk engine (TTC, distance, motion, path)| `ttc.py`, `features.py` | `main.py` | **ACTIVE** |
| `adaptive_navigation/risk/features.py` | Risk | Data contract for aggregated obstacle risk features | `dataclasses` | `risk_engine.py` | **ACTIVE** |
| `adaptive_navigation/risk/score.py` | Risk | Early prototype risk score stub (superseded) | `enum` | None | **LEGACY** |
| `adaptive_navigation/risk/state_machine.py` | Risk | Early prototype state machine stub (superseded) | None | None | **LEGACY** |
| `adaptive_navigation/risk/uncertainty.py` | Risk | Early prototype uncertainty metric stub (superseded)| None | None | **LEGACY** |
| `adaptive_navigation/temporal/__init__.py` | Temporal | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/temporal/history.py` | Temporal | Rolling observation history buffer (deque maxlen=30)| `dataclasses` | Subsystems | **ACTIVE** |
| `adaptive_navigation/temporal/motion.py` | Temporal | Image velocity & relative depth rate estimator (EMA)| `history.py` | `main.py` | **ACTIVE** |
| `adaptive_navigation/temporal/camera_motion.py` | Temporal | Sparse Lucas-Kanade optical flow ego-motion compensator| `cv2`, `numpy`, `motion.py`| `main.py` | **ACTIVE** |
| `adaptive_navigation/temporal/velocity.py` | Temporal | Early prototype velocity estimator stub (superseded)| None | None | **LEGACY** |
| `adaptive_navigation/temporal/smoothing.py` | Temporal | Early prototype EMA smoother stub (superseded) | None | None | **LEGACY** |
| `adaptive_navigation/uncertainty/__init__.py` | Uncertainty | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/uncertainty/reliability.py` | Uncertainty | Multi-source evidence coverage & reliability scoring | `temporal.history` | `main.py`, `warning` | **ACTIVE** |
| `adaptive_navigation/warning/__init__.py` | Warning | Subpackage initializer | None | `main.py` | **ACTIVE** |
| `adaptive_navigation/warning/state_machine.py` | Warning | 2-frame hysteresis temporal warning state machine | `risk_engine`, `reliability`| `main.py` | **ACTIVE** |
| `adaptive_navigation/warning/message_generator.py`| Warning | Concise spoken warning formatter (<8 words) | `dataclasses` | `main.py` | **ACTIVE** |
| `adaptive_navigation/feedback/__init__.py` | Feedback | Early prototype feedback subpackage (superseded) | None | None | **LEGACY** |
| `adaptive_navigation/feedback/speech.py` | Feedback | Empty offline TTS stub (superseded by audio/tts.py) | None | None | **LEGACY** |
| `adaptive_navigation/feedback/warning_manager.py`| Feedback | Empty warning manager stub (superseded by warning) | None | None | **LEGACY** |
| `adaptive_navigation/feedback/message_generator.py`| Feedback| Early message generator stub (superseded by warning)| None | None | **LEGACY** |
| `adaptive_navigation/evaluation/` | Duplicate | Duplicate copy of root evaluation/ | Duplicate | Root scripts | **DEPRECATED** |
| `src/__init__.py` | Source Layout | Target architecture namespace wrapper | `adaptive_navigation` | Clean imports | **ACTIVE** |

### Configurations (`configs/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `configs/final.yaml` | Config | Canonical reference profile for release | None | `main.py --mode final` | **ACTIVE** |
| `configs/development.yaml`| Config | Interactive demo mode with visual preview & overlays | None | `main.py --mode demo` | **ACTIVE** |
| `configs/evaluation.yaml` | Config | Headless evaluation profile with deterministic seed | None | `main.py --mode research` | **ACTIVE** |
| `configs/real_world.yaml` | Config | Field trial configuration with sensor noise filtering | None | `main.py --mode real_world` | **ACTIVE** |
| `configs/deployment.yaml` | Config | Edge deployment profile with FP16 and 2:1 depth | None | `main.py --mode deployment` | **ACTIVE** |
| `configs/final_experiment_config.yaml`| Config| Parameter overrides for ablation experiments | None | Evaluation scripts | **ACTIVE** |

### Validation Framework (`validation/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `validation/README.md` | Validation | Documentation of clean validation infrastructure | None | Developers | **ACTIVE** |
| `validation/videos/` | Validation | Storage for controlled real-world validation videos (5 scenarios, 5,235 frames) | Physical Camera | Phase 2A/2B Runners | **ACTIVE (Populated)** |
| `validation/results/video_runs/` | Validation | Phase 2A baseline real GPU telemetry & per-video reports | Models / GPU | Phase 2A | **ACTIVE (Baseline)** |
| `validation/results/video_runs_phase2b/` | Validation | Phase 2B revalidated real GPU telemetry & per-video reports | Models / GPU | Phase 2B | **ACTIVE (Baseline)** |
| `validation/results/video_runs_phase2c/` | Validation | Phase 2C 2:1 depth cadence real GPU telemetry & reports | Models / GPU | Phase 2C | **ACTIVE (Verified)** |
| `validation/results/video_runs_phase2c_egomotion/` | Validation | Phase 2C forward ego-motion compensated telemetry & reports | Models / GPU | Phase 2C | **ACTIVE (Verified)** |
| `validation/results/phase2a_video_validation_report.md` | Validation | Phase 2A baseline report across all 5 videos | Video runs | Phase 2A | **ACTIVE** |
| `validation/results/phase2b_behavioral_validation_report.md` | Validation | Phase 2B comparative report analyzing failure mode fixes | Video runs 2B | Phase 2B | **ACTIVE** |
| `validation/results/phase2c_validation_report.md` | Validation | Phase 2C comprehensive throughput and ego-motion report | Video runs 2C | Phase 2C | **ACTIVE** |
| `validation/results/video_inventory.csv` | Validation | Manifest of 5 controlled test videos with codecs & resolutions | `cv2.VideoCapture` | Inventory | **ACTIVE** |
| `validation/datasets/heads_up/README.md`| Docs | HEADS-UP dataset guide, git hygiene & credential policy | None | Researchers | **ACTIVE** |
| `validation/datasets/heads_up/heads_up_adapter.py`| Dataset | Adapter exposing 1280x720 frames, 6-DOF poses, trajectories | OpenCV / CSV | Phase 3A Runners | **ACTIVE** |
| `validation/datasets/heads_up/metadata/`| Data | Sliced camera poses, trajectories, and calibration CSVs | HEADS-UP repo | Adapters | **ACTIVE (Tracked)** |
| `validation/datasets/heads_up/sequences/`| Data | Extracted 250 frames & MP4s for 3 target episodes (Local Only) | HEADS-UP tar | Validation | **ACTIVE (Ignored)** |
| `validation/results/heads_up/` | Validation | Phase 3A GPU telemetry (JSON, CSVs) and per-sequence reports | Models / GPU | Phase 3A | **ACTIVE (Tracked)** |
| `validation/results/phase3a_heads_up_report.md` | Validation | Phase 3A master report on egocentric generalization | Telemetry | Phase 3A | **ACTIVE** |
| `validation/calibration/` | Validation | Camera intrinsic matrices & metric scale parameters | Sensor Rig | Pipeline | **ACTIVE (Empty)** |

### Diagnostic & Execution Scripts (`scripts/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `scripts/tools/check_environment.py`| Tool | Diagnostics for Python, PyTorch, CUDA, and weights | `torch`, `cv2` | Developers | **ACTIVE** |
| `scripts/tools/download_heads_up_unconstrained.py`| Tool | Gated HEADS-UP archive downloader (reads `HF_TOKEN`) | `urllib` | Researchers | **ACTIVE** |
| `scripts/tools/resume_heads_up_unconstrained.py`| Tool | Resumable chunk downloader with retry logic (reads `HF_TOKEN`) | `urllib` | Researchers | **ACTIVE** |
| `scripts/tools/finalize_heads_up_sequences.py`| Tool | Prunes placeholder frames & compiles sequence MP4 videos | `cv2`, `json` | Pipeline | **ACTIVE** |
| `scripts/run/run_pipeline.py`| Runner | Convenience wrapper for root `main.py` | `main.py` | CLI Users | **ACTIVE** |
| `scripts/run/run_phase2a_validation.py`| Runner | Phase 2A automated execution runner across 5 videos | `main.py` | Phase 2A | **ACTIVE** |
| `scripts/run/run_phase2b_validation.py`| Runner | Phase 2B automated execution runner across 5 videos | `main.py` | Phase 2B | **ACTIVE** |
| `scripts/run/run_phase2c_validation.py`| Runner | Phase 2C automated execution runner (2:1 cadence & ego-motion) | `main.py` | Phase 2C | **ACTIVE** |
| `scripts/run/run_phase3a_heads_up_validation.py`| Runner | Phase 3A automated execution runner across HEADS-UP episodes | `main.py` | Phase 3A | **ACTIVE** |

### Tests (`tests/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_imports.py` | Test | Smoke test verifying all active modules import cleanly| `unittest`, `adaptive_navigation`| CI / Developers | **ACTIVE** |
| `tests/test_detector_clip.py` | Test | Verifies YOLO11n genuine inference on `data/test_clip.mp4`| `ultralytics`, `cv2` | Phase 1A Bring-Up | **ACTIVE** |
| `tests/test_tracker_clip.py` | Test | Verifies BoT-SORT track continuity on `data/test_clip.mp4`| `ultralytics`, `lap` | Phase 1A Bring-Up | **ACTIVE** |
| `tests/test_depth_convention.py`| Test | Characterizes Depth Anything V2 output convention | `torch`, `adaptive_navigation` | Phase 1A Bring-Up | **ACTIVE** |
| `tests/test_ttc_consistency.py`| Test | Sanity test verifying TTC/risk for approaching/receding/static| `unittest`, `adaptive_navigation`| CI / Verification | **ACTIVE** |
| `tests/test_phase2b_stabilization.py`| Test | Verifies indoor policy class filtering, regression slope, and gait spike rejection | `unittest`, `adaptive_navigation`| Phase 2B Verification | **ACTIVE** |
| `tests/test_phase2c_egomotion.py`| Test | Verifies radial divergence ego-motion compensation, gait spike absorption, and oncoming hazard preservation | `unittest`, `adaptive_navigation`| Phase 2C Verification | **ACTIVE** |

### Documentation (`docs/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `docs/dataset_card.md` | Docs | Dataset specifications and ethical qualification | None | Research Conclave | **ACTIVE** |
| `docs/model_card.md` | Docs | Specifications for YOLO11n and Depth Anything V2 | None | Research Conclave | **ACTIVE** |
| `docs/final_research_report.md`| Docs | 21-section formal research report | Legacy scripts | Academic Deliverable| **ACTIVE** |
| `docs/demo_script.md` | Docs | Guided 12-stage interactive demonstration protocol | None | Demonstrators | **ACTIVE** |
| `docs/research_gap_mapping.md`| Docs | Mapping of prototype features to research gaps | None | Paper Author | **ACTIVE** |

### Legacy Evaluation Scripts (`evaluation/`)
| File | Category | Purpose | Depends On | Used By | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `evaluation/run_evaluation.py` | Legacy Eval | Synthetic evaluation script using `time.sleep` & noise| `ground_truth.py` | Legacy Run | **DEPRECATED / FABRICATED** |
| `evaluation/pilot_testing.py` | Legacy Eval | Scripted 240-frame field trial simulation | `real_world_logger.py`| Legacy Run | **DEPRECATED / FABRICATED** |
| `evaluation/ground_truth.py` | Legacy Eval | Programmatic 10-scenario synthetic GT generator | `dataclasses` | Legacy Eval | **DEPRECATED / SYNTHETIC** |
| `evaluation/metrics.py` | Legacy Eval | Standard statistical metric functions | `numpy` | Legacy Eval | **ACTIVE (Math Sound)** |
| `evaluation/ablation.py` | Legacy Eval | 5 ablation study comparisons (evaluated on noise) | `metrics.py` | Legacy Eval | **DEPRECATED / FABRICATED** |
| `evaluation/benchmark.py` | Legacy Eval | Latency profile recorder | `psutil`, `torch` | Legacy Eval | **ACTIVE (Math Sound)** |
| `evaluation/error_analysis.py` | Legacy Eval | Failure case categorizer across 13 error types | `dataclasses` | Legacy Eval | **ACTIVE (Logic Sound)** |
| `evaluation/real_world_logger.py`| Legacy Eval | Structured event telemetry logger (CSV/JSON) | `dataclasses` | Legacy Pilot | **ACTIVE (Logic Sound)** |
| `evaluation/system_optimizer.py`| Legacy Eval | FPS elevation test script | `benchmark.py` | Legacy Run | **DEPRECATED / FABRICATED** |
| `evaluation/report_generator.py`| Legacy Eval | Publication report generator | `metrics.py` | Legacy Eval | **DEPRECATED** |
| `evaluation/test_metadata.yaml` | Legacy Config| Metadata defining 12 simulated test trials | None | Legacy Pilot | **DEPRECATED / SYNTHETIC** |

---

## 6. Entry Points

All executable entry points with verified working command lines:

### 1. Primary Application (Production Freeze)
```powershell
python main.py --mode final
# Explicit configuration override:
python main.py --config configs/final.yaml
```

### 2. Interactive Graphical Demo (Visual Overlay & Inset)
```powershell
python main.py --mode demo
```

### 3. Execution on Pre-Recorded Video Clip
```powershell
python main.py --config configs/final.yaml --video data/test_clip.mp4 --headless --max-frames 30
```

### 4. Headless Edge Deployment Prototype
```powershell
python main.py --mode deployment --cam 0 --headless
```

### 5. Environment & Hardware Diagnostics
```powershell
python scripts/tools/check_environment.py
```

### 6. Subsystem Unit & Import Regression Tests
```powershell
python -m unittest tests/test_imports.py
```

### 7. Legacy Evaluation Runner (Fabricated Benchmark — For Audit Only)
```powershell
python evaluation/run_evaluation.py
```

### 8. Legacy Field Trial Simulator (Fabricated Trials — For Audit Only)
```powershell
python evaluation/pilot_testing.py
```

---

## 7. Configuration Index

| Configuration File | Environment Profile | Key Controls & Behavioral Overrides |
| :--- | :--- | :--- |
| [`configs/final.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/final.yaml) | `final` (Production) | YOLO11n (FP16), BoT-SORT (buffer=25), Depth (cadence=2:1, metric=true), Optical flow ego-motion enabled, TTC thresholds (1.0s/2.0s), Risk weights (TTC=0.35, dist=0.20, path=0.20), Audio enabled |
| [`configs/development.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/development.yaml) | `demo` (Visual Preview) | Active OpenCV display window, bounding-box overlays, optical flow motion vectors, depth colormap inset enabled, debug HUD enabled |
| [`configs/evaluation.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/evaluation.yaml) | `research` (Evaluation) | Headless execution, deterministic random seed (42), strict metric thresholds, suppression of GUI overhead |
| [`configs/real_world.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/real_world.yaml) | `real_world` (Field Testing) | High sensor noise filtering, walking gait jitter compensation, wearable Bluetooth audio sink enabled |
| [`configs/deployment.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/deployment.yaml) | `deployment` (Edge Device) | Strict FP16 tensor precision, periodic 2:1 depth inference cadence, cache interpolation, lightweight memory footprints |
| [`configs/final_experiment_config.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/final_experiment_config.yaml)| `experiments` | Parameter sweeps and ablation matrix switches |
| [`adaptive_navigation/config.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/config.yaml)| Package Default | Fallback default configuration loaded if no config is specified via CLI |

---

## 8. Model / Weight Index

| Model Identifier | Expected Weight File | Exists? | Target Directory | Used By | Purpose | Verified? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n** | `yolo11n.pt` (5.61 MB) | **YES** | `models/detector/` | `YOLOObjectDetector` | 2D obstacle localization and class classification | **VERIFIED (GPU: 9.59ms p50, 73 dets)** |
| **Depth Anything V2 (ViT-S)** | `depth_anything_v2_vits.pth` (99.2 MB) | **YES** | `models/depth/` | `DepthAnythingV2Estimator` | Monocular inverse relative depth map estimation | **VERIFIED (GPU: 69.25ms p50, 10.3x speedup)** |

*Canonical Location Resolution*: Root `models/` is the single canonical storage location. The redundant copy in `adaptive_navigation/models/` was permanently purged to prevent duplicate tracking and reclaim disk space. Checkpoint path auto-resolution was verified in `depth.py`.

---

## 9. Dataset Index

| Dataset Identifier | Local? | Storage Location | Purpose | Annotation Schema | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test Clip MP4** | **Yes** | `data/test_clip.mp4` | 60-frame 640x480 video for video-file simulation | None (Raw Video) | **AVAILABLE** |
| **Controlled Real-World Footage**| **Yes** | `validation/videos/` | 5 recorded indoor pedestrian scenarios (5,235 frames) | Physical Markers | **VERIFIED (Phases 2A–2C)** |
| **HEADS-UP Representative Sample**| **Local Only** | `validation/datasets/heads_up/sequences/` | 3 targeted unconstrained egocentric episodes (250 frames) | Machine Pseudo-Labels | **VERIFIED (Phase 3A)** |
| **Canonical 10-Scenario Suite** | **No** (Virtual) | `evaluation/ground_truth.py` | Programmatic kinematic trajectory generation | Python dataclass | **DEPRECATED (Synthetic)** |
| **Real-World Trial Catalog** | **No** (Virtual) | `evaluation/test_metadata.yaml` | Catalog of 12 simulated test cases (`RW_001`–`RW_012`)| YAML metadata | **DEPRECATED (Simulated)** |
| **NYU-Depth V2 / KITTI Depth** | **No** | `validation/datasets/depth/` | Monocular metric depth ground-truth evaluation | Dense Depth Maps | **PLANNED** |

### Dataset Git Hygiene & Credential Security Policy (HEADS-UP)

1. **Local-Only Raw Data Mandate**:
   - The official 102 GB HEADS-UP dataset archive is strictly prohibited from being downloaded in full or committed to the repository.
   - All raw archives (`validation/datasets/heads_up/*.tar`, `*.tar.gz`, `*.zip`), raw staging areas (`raw/`, `cache/`), and extracted image frame sequences (`validation/datasets/heads_up/sequences/`) are permanently ignored by Git via `.gitignore`.
2. **Tracked Lightweight Assets**:
   - Lightweight metadata CSVs (`validation/datasets/heads_up/metadata/`: camera poses, trajectory labels, calibrations).
   - Dataset adapter code (`validation/datasets/heads_up/heads_up_adapter.py`).
   - Dataset documentation and reproduction guide ([`validation/datasets/heads_up/README.md`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/datasets/heads_up/README.md)).
   - Validation outputs and telemetry logs (`validation/results/heads_up/` and [`validation/results/phase3a_heads_up_report.md`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/phase3a_heads_up_report.md)).
3. **Strict Credential-Handling Policy**:
   - Hugging Face authentication tokens (`hf_...`) must **NEVER** be hardcoded or committed into source code, scripts, configs, documentation, or reports.
   - All dataset tools read exclusively from the environment variable: `os.environ.get("HF_TOKEN")`.
   - Scripts terminate with an explicit error if the required token is absent, preventing silent fallback or accidental exposure.

---

## 10. Validation Status

### Synthetic Verification (Mode A)
* **Status**: **PLANNED (Architecture Designed)**
* **Detail**: Legacy synthetic evaluation scripts (`run_evaluation.py`) generated Gaussian noise around ground truth. A genuine `SyntheticVideoProvider` rendering actual video frames with true 3D kinematics is designed in `evaluation_plan.md`.

### External Dataset Evaluation (Mode B)
* **Status**: **PLANNED (Architecture Designed)**
* **Detail**: Standardized evaluation on HEADS-UP / MOT / NYU-Depth datasets designed in `evaluation_plan.md`. No large datasets downloaded during Phase 0.

### Controlled Real-World Video Evaluation (Mode C)
* **Status**: **ACTIVE (CPU & GPU Verified)**
* **Detail**: Executed complete 15-stage pipeline on `data/test_clip.mp4` across both CPU and NVIDIA RTX 4050 GPU (`cuda:0`).
  - **CPU Baseline:** Mean latency 772.57 ms (1.29 FPS)
  - **GPU RTX 4050:** Mean latency 101.04 ms (9.90 FPS, p50: 11.07 FPS), 7.6× speedup.
  - Telemetry logs saved: `gpu_telemetry.*` and `real_telemetry.*` in `validation/logs/`.
  - Smoke reports: `gpu_smoke_test_report.md` and `smoke_test_report.md` in `validation/results/`.

### Live Camera Testing
* **Status**: **VERIFIED (CUDA Operational)**
* **Detail**: Models provisioned in canonical `models/`, dependencies verified in `.venv\Scripts\python.exe`, and verified operational on CUDA (`cuda:0`).

### Human-in-the-Loop Usability
* **Status**: **NOT STARTED**
* **Detail**: Strictly prohibited until formal institutional ethical review and certified technical safety verification are completed.

---

## 11. Experimental Evidence Registry

Audit of all previously published experimental claims:

| Metric Claim | Reported Value | Provenance File | Actually Produced by Model Inference? | Reproducible from Raw Input? | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Detection Recall** | `95.80%` / `100.0%` | `run_evaluation.py:161` | **NO** (Gaussian jitter on GT bbox) | **NO** | **FABRICATED / DEPRECATED** |
| **Mean IoU** | `0.9576` | `run_evaluation.py:162` | **NO** ($\mathcal{N}(0, 1.5^2)$ pixel offset) | **NO** | **FABRICATED / DEPRECATED** |
| **Tracking ID Stability** | `99.58%` | `pilot_testing.py:475` | **NO** (Forced drop at frame 11) | **NO** | **FABRICATED / DEPRECATED** |
| **Depth MAE** | `0.0962 m` / `0.109 m` | `run_evaluation.py:174` | **NO** ($\mathcal{N}(0, 0.12^2)$ added to GT depth)| **NO** | **FABRICATED / DEPRECATED** |
| **TTC MAE** | `0.1285 s` / `0.101 s` | `run_evaluation.py:181` | **NO** ($\mathcal{N}(0, 0.15^2)$ added to GT TTC) | **NO** | **FABRICATED / DEPRECATED** |
| **Risk Accuracy** | `95.38%` | `run_evaluation.py:202` | **NO** (8% random mutation of GT risk) | **NO** | **FABRICATED / DEPRECATED** |
| **Reliability ECE** | `0.1305` | `run_evaluation.py:209` | **NO** (Calculated on fake uniform scores) | **NO** | **FABRICATED / DEPRECATED** |
| **False Warning Rate** | `1.94%` (vs `93.3%`) | `run_evaluation.py:246` | **NO** (Bernoulli trial multiplication) | **NO** | **FABRICATED / DEPRECATED** |
| **Navigation Accuracy**| `94.00%` | `run_evaluation.py:259` | **NO** (6% random mutation of GT action) | **NO** | **FABRICATED / DEPRECATED** |
| **Throughput & Latency**| `86.58 FPS` / `11.55 ms`| `run_evaluation.py:97` | **NO** (`time.sleep` + manual additions) | **NO** | **FABRICATED / DEPRECATED** |

---

## 12. Known Technical Problems & Blockers

1. **Monocular Scale Ambiguity / Uncalibrated Relative Depth**:
   - Depth Anything V2 outputs affine-invariant relative disparity. To obtain metric distance in meters (instead of relative disparity units), camera intrinsics or ground-plane homography calibration is required. (Currently, TTC is physically valid in seconds via optical divergence $\tau = d / \dot{d}$).
2. **Real-Time 30 FPS Headroom**:
   - PyTorch CUDA inference on the mobile RTX 4050 achieves ~11 FPS (101 ms latency). Achieving full 30 FPS camera frame-rate parity will require TensorRT FP16 compilation or a 2:1 depth-to-detection inference cadence.
3. **2D Ego-Motion Limitation**:
   - Sparse optical flow in `adaptive_navigation/temporal/camera_motion.py` compensates 2D image-plane translation ($v_x, v_y$). Forward/backward user walking velocity ($v_z$) cannot be compensated without IMU or depth-flow integration.
4. **Walking Gait Oscillation Induced TTC Spikes**:
   - In `S04_receding_r01`, while macro disparity decreased from 3.6 to 2.85, localized walking bob and torso sway produced transient positive disparity derivatives, causing intermittent false `APPROACHING` alerts.
5. **Indoor Architectural Clutter False Positives**:
   - General COCO 80-class weights falsely identified hallway fire doors, wall panels, and reflections as `toilet`, `cat`, and `refrigerator`, elevating warning states in clear corridors (`S01_clear_r01`).
6. **Depth Model Bottleneck**:
   - Depth Anything V2 accounts for ~80% of total runtime (~97 ms of ~121 ms), capping throughput at ~8.0 FPS.

---

## 13. Research Roadmap

### DONE (Phase 0, Phase 1A, Phase 1A.1 & Phase 2A)
- Complete codebase audit across all 14 evaluation dimensions ([audit_report.md](audit_report.md)).
- Metric provenance traceability analysis ([metric_traceability.md](metric_traceability.md)).
- Validation architecture design for Modes A, B, and C ([evaluation_plan.md](evaluation_plan.md)).
- Target workspace directory structure created (`models/`, `validation/`, `scripts/`, `tests/`, `outputs/`, `archive/`).
- Master file index created ([PROJECT_INDEX.md](PROJECT_INDEX.md)).
- Package import resolution repaired in `adaptive_navigation/__init__.py`.
- Automated diagnostic script created ([scripts/tools/check_environment.py](scripts/tools/check_environment.py)).
- Regression import tests created and verified ([tests/test_imports.py](tests/test_imports.py)).
- **Phase 1A Dependencies Provisioned**: `ultralytics` 8.4.172, `pyttsx3` 2.99, `lap` 0.5.13 installed in active virtualenv (`.venv`).
- **Phase 1A Official Weights Provisioned**: `yolo11n.pt` (5.61 MB) and `depth_anything_v2_vits.pth` (99.2 MB) verified in canonical `models/`.
- **Phase 1A Detector & Tracker Verified**: 60 frames processed on `data/test_clip.mp4` with 100.0% tracking continuity on Track ID 1.
- **Phase 1A Depth/TTC Mismatch Repaired**: Repaired `ttc.py` and `motion.py` to support explicit depth conventions and optical divergence TTC derivation $\tau = d / \dot{d}$.
- **Phase 1A Unit Sanity Test Verified**: `tests/test_ttc_consistency.py` passed 100% across approaching, receding, and static targets.
- **Phase 1A CPU Baseline Executed**: 60/60 frames processed on CPU (`real_telemetry.*` logs, `smoke_test_report.md`).
- **Phase 1A.1 GPU Acceleration Executed**: Verified `PyTorch 2.10.0+cu130` on `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`).
- **Phase 1A.1 Duplicate Weight Purged**: Removed redundant weights from `adaptive_navigation/models/`, verified auto-resolution to canonical root `models/`.
- **Phase 1A.1 GPU Telemetry Artifacts**: Saved `gpu_telemetry.json`, `gpu_telemetry_frames.csv`, `gpu_telemetry_objects.csv`, and `gpu_smoke_test_report.md` (10.3× depth speedup, 7.6× overall speedup, 11.1 FPS steady-state).
- **Phase 2A Controlled Video Behavioral Validation Executed**:
  - Validated across all 5 recorded scenarios under `validation/videos/` (5,235 total frames, 89.3s duration).
  - OpenCV decode health: 100% (5/5).
  - Executed on `cuda:0` (NVIDIA RTX 4050) using `.\.venv\Scripts\python.exe`.
  - Processed 5,235 frames in 654.37s wall-clock time at steady **8.00 FPS** (p50: **120.5 ms**, p95: **128.6 ms**).
  - Memory: Rock-solid 138.24 MB allocated / 518 MB reserved (<7.3% of 6 GB VRAM).
  - Separate per-video runs under `validation/results/video_runs/<video_stem>/` (`telemetry.json`, `telemetry_frames.csv`, `telemetry_objects.csv`, `run_report.md`).
  - Master report compiled in `validation/results/phase2a_video_validation_report.md`.
  - Video manifest logged in `validation/results/video_inventory.csv` and `validation/videos/README.md`.

- **Phase 2B Behavioral Failure Modes Fixed & Revalidated**:
  - Implemented configuration-driven indoor class policy filtering in `config.yaml` and `detector.py`, retaining raw detections in telemetry while suppressing architectural COCO hallucinations (`cat`, `toilet`, `refrigerator`, etc.).
  - Implemented multi-frame least-squares regression slope, 5-frame consecutive confirmation hysteresis, and optical bounding box area shrinkage cross-validation in `motion.py`.
  - Re-ran all 5 validation videos (5,235 frames) on `cuda:0` under `validation/results/video_runs_phase2b/`.
  - Completely eliminated hallway false alarms in `S01_clear_r01` (warnings dropped from 793 frames to 0).
  - Reduced receding approach spikes in `S04_receding_r01` by 64% and false critical alarms by 82.1%.
  - Preserved full sensitivity to true closing hazards in `S03_approaching_r01` and `S02_static_r01`.
  - Compiled master comparative report in `validation/results/phase2b_behavioral_validation_report.md`.

### NEXT (Phase 3B & Beyond)
- [x] Download and prepare representative HEADS-UP benchmark dataset for egocentric testing (Completed in Phase 3A: 250 frames across 3 unconstrained episodes).
- [ ] Implement asynchronous audio TTS queue to resolve Windows COM event loop contention.
- [ ] Implement ground-plane calibration for metric distance estimation.
- [ ] Integrate wearable 6-DOF IMU gyro fusion for angular ego-motion compensation.

### BLOCKED
- Human-subject testing with visually impaired users (Strictly blocked pending institutional ethical review and technical safety certification).

---

## 14. Change Log

### 2026-10-04 (Phase 3A — HEADS-UP Exploratory External Validation)
- **Dataset Audit & Selection**: Inspected official HEADS-UP benchmark (`Yassaman/HEADS-UP`, *Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems*, arXiv:2409.20324v1, EPFL VITA Lab). Verified sensor specs (ZED Mini stereo, 1280x720 @ 30 FPS, 800 Hz IMU, visual odometry poses). Documented crucial provenance limitation: trajectory labels are machine pseudo-labels (YOLOv8 + ByteTrack + Kalman), NOT ground-truth human annotations.
- **Exploratory Extraction & Zero-Bloat Management**: Downloaded and verified archive slice without downloading the full 102 GB dataset. Extracted 250 valid frames across 3 targeted unconstrained episodes (8.33 seconds total):
  - `HU_unconstrained_s01_multiped`: 73 frames, multi-agent density (Agents 1, 2, 3 active).
  - `HU_unconstrained_s02_approach`: 102 frames, closing oncoming pedestrian (Agent 5, 10.5m -> 6.5m).
  - `HU_unconstrained_s03_headmotion`: 75 frames, severe head scanning (>60°/s) with close hazard (Agent 74, 1.7m -> 4.4m).
  *Explicit Scope Demarcation*: These 250 frames are not a statistically representative sample of the full 43,213-frame benchmark; evaluation represents exploratory behavioral stress testing. Reclaimed the 11.51 GB temporary download immediately following extraction.
- **Created Adapter & Tools**:
  - `validation/datasets/heads_up/heads_up_adapter.py`: standard adapter yielding 1280x720 frames, synchronized 6-DOF poses, and machine reference trajectories.
  - `scripts/tools/finalize_heads_up_sequences.py`: cleans placeholder frames and compiles MP4 video assets.
  - `scripts/run/run_phase3a_heads_up_validation.py`: automated runner and latency profiler for HEADS-UP sequences.
- **Executed on GPU (`cuda:0`, RTX 4050)**:
  - `s01_multiped`: 73 frames in 10.31s @ **7.08 FPS** (p50: 135.56 ms). Track IDs: 7. Valid TTC: 170. Evasive steering `AVOID_RIGHT` on 38 frames (52.1%).
  - `s02_approach`: 102 frames in 9.77s @ **10.44 FPS** (p50: 92.80 ms). Track ID 1 persisted 102/102 frames (100%). Valid TTC: 125 down to 2.30s. Sustained `CAUTION` warning on 101 frames (99.0%).
  - `s03_headmotion`: 75 frames in 7.63s @ **9.82 FPS** (p50: 137.18 ms). Measured optical flow displacement up to **100.26 px/frame** (98.7% valid flow). 72 `WARNING` frames (96.0%), `AVOID_RIGHT` (41 frames), `AVOID_LEFT` (15 frames), and emergency `STOP` (17 frames, 22.7%).
- **Detection & Class Filtering**: Clearly distinguished raw detector output (1,288 candidate proposals) from filtered active detections (941 proposals, 347 rejected by policy). Zero active false-warning detections after the configured indoor navigation class policy on the selected sequences (active hazards: `person`: 725, `bicycle`: 67, `backpack`: 16).
- **Master Report Compiled**: `validation/results/phase3a_heads_up_report.md` (Sections A through N).
- **Status**: Phase 3A exploratory external egocentric validation successfully completed.

### 2026-10-04 (Phase 2C — Throughput + Forward Ego-Motion Validation)
- **Investigated & Verified**: Detailed track lifetime and class identity for Track 1 in `S04_receding_r01`. Confirmed Track 1 is 100% `person` on all 857 frames. Resolved that the 144 approach frames belonged to Track 1 across 10 discrete intervals (frames 6–36 initial camera acceleration, walking gait torso pitch oscillations, and frames 767–793 hall termination deceleration), while background obstacles (Tracks 7, 10, 15) contributed 356 additional false approach frames.
- **Implemented**: 2:1 Depth Cadence in `adaptive_navigation/config.yaml` and `adaptive_navigation/main.py`. Depth Anything V2 runs every 2nd frame; cached `DepthResult` with fresh timestamps and current bounding-box sampling propagates depth on interleaved frames.
- **Implemented**: Background Forward Ego-Motion Compensation via radial optical flow divergence ($\bar{\gamma}_{\text{bg}} = \text{median}\left(\frac{\Delta \mathbf{p} \cdot \mathbf{r}}{r^2 \cdot \Delta t}\right)$) in `adaptive_navigation/temporal/camera_motion.py`. Compensates apparent depth closing rate $\dot{d}_{\text{ego}} = \bar{\gamma}_{\text{bg}} \cdot d_{\text{obj}}$ with bounding-box optical area rate gating ($\dot{A}/A \le 0.05$).
- **Created**: `scripts/run/run_phase2c_validation.py` — automated batch runner supporting both 2:1 cadence and ego-motion compensation.
- **Created**: `tests/test_phase2c_egomotion.py` — unit tests verifying radial divergence estimation, receding gait spike absorption, and oncoming collision hazard preservation. Full test suite (17/17 tests) passes in 0.068s.
- **Executed & Validated**: 10 full video runs across 5 controlled scenarios (5,235 frames each, 10,470 total frames):
  - `video_runs_phase2c/` (2:1 Cadence): Average throughput increased from 7.80 FPS to **13.38 FPS** (+71.5% speedup), p50 latency dropped from ~125 ms to **33–45 ms**, VRAM static at 138.24 MB.
  - `video_runs_phase2c_egomotion/` (Cadence + Ego-Motion): Track 1 false approaches in S04 reduced to 120 frames (70% cumulative reduction from 2A baseline). Genuine oncoming hazard in S03 100% preserved (540 APPROACHING frames, 570 valid TTC frames down to 0.73s). S02 center obstacle 100% preserved (577 valid TTC frames, 157 STOP commands). S01 negative control 100% clean (1,524 NO_WARNING frames).
- **Synthesized**: Master report in `validation/results/phase2c_validation_report.md`.
- **Status**: Phase 2C throughput and forward ego-motion validation successfully completed.

### 2026-10-04 (Phase 2B — Behavioral Failure Mode Fixes & Revalidation)
- **Implemented**: Indoor Navigation Class Policy in `adaptive_navigation/config.yaml` and `adaptive_navigation/perception/detector.py`. Added `policy_accepted` and `filter_reason` metadata to `Detection`. Preserved full raw detector outputs in telemetry (`raw_detections` vs `active_detections` vs `filtered_detections`).
- **Implemented**: Multi-frame temporal stabilization and kinematic hysteresis in `adaptive_navigation/temporal/motion.py`. Replaced 2-point difference with 8-frame least-squares regression derivative, added 5-frame consecutive confirmation counters (`consecutive_pos`, `consecutive_neg`), and integrated optical area shrinkage cross-validation ($\dot{A}/A < -0.15$).
- **Created**: `tests/test_phase2b_stabilization.py` — unit tests for class filtering, rejection of 1-frame gait bumps, and closing motion confirmation. All 14 test cases pass in 0.059s.
- **Created**: `scripts/run/run_phase2b_validation.py` — dedicated Phase 2B automated execution runner.
- **Executed**: Sequential re-validation across all 5 videos on `cuda:0` (NVIDIA RTX 4050) saving outputs under `validation/results/video_runs_phase2b/`:
  - `S01_clear_r01`: 1,524 frames in 205.59s @ 7.41 FPS. 328 hallway artifacts filtered. 0 false warnings (1,524 NO_WARNING). 0 audio utterances.
  - `S02_static_r01`: 1,158 frames in 145.55s @ 7.96 FPS. Approaching stationary target, 1181 valid TTC instances, 67 CRITICAL frames, 157 STOP commands.
  - `S03_approaching_r01`: 788 frames in 100.11s @ 7.87 FPS. 591 APPROACHING frames (+14.3%), 582 valid TTC measurements, decreasing to 0.66s.
  - `S04_receding_r01`: 857 frames in 115.80s @ 7.40 FPS. Track 1 APPROACHING spikes reduced from 400 to 144 (-64.0%), CRITICAL alerts reduced from 95 to 17 (-82.1%).
  - `S05_crossing_r01`: 908 frames in 125.26s @ 7.25 FPS. Lateral pedestrian tracked cleanly, false APPROACHING dropped by 59.3%, 0 false CRITICAL alerts.
- **Synthesized**: Master comparison report in `validation/results/phase2b_behavioral_validation_report.md`.
- **Status**: Phase 2B behavioral validation successfully completed.

### 2026-10-04 (Phase 2A — Controlled Video Behavioral Validation)
- **Inventory & Compatibility**: Inspected 5 video scenarios under `validation/videos/` (`approaching`, `clear_path`, `crossing`, `receding`, `static_obstacle`). Verified 100% OpenCV frame-by-frame decoding across all 5,235 frames.
- **Created**: `validation/results/video_inventory.csv` and `validation/videos/README.md`.
- **Created**: `scripts/run/run_phase2a_validation.py` runner script for automated execution and latency profiling.
- **Executed**: Real neural pipeline executed sequentially on `cuda:0` (NVIDIA RTX 4050) on all 5 videos:
  - `S03_approaching_r01`: 788 frames in 98.66s @ 7.99 FPS (p50: 120.02 ms). Track 1 continuous 100%. Valid TTC: 546.
  - `S01_clear_r01`: 1,524 frames in 190.06s @ 8.02 FPS (p50: 120.62 ms). 0 CRITICAL warnings. Negative control verified.
  - `S05_crossing_r01`: 908 frames in 114.53s @ 7.93 FPS (p50: 121.69 ms). Pedestrian triggered STOP (44 frames) & AVOID_LEFT (308 frames).
  - `S04_receding_r01`: 857 frames in 106.47s @ 8.05 FPS (p50: 119.96 ms). Disparity decreased 3.6 $\rightarrow$ 2.85; gait oscillation documented.
  - `S02_static_r01`: 1,158 frames in 144.65s @ 8.01 FPS (p50: 120.55 ms). Advancing camera, TTC 2.78s $\rightarrow$ 0.19s, STOP (157 frames).
- **Generated**: Comprehensive results per video in `validation/results/video_runs/<video_stem>/` (`telemetry.json`, `telemetry_frames.csv`, `telemetry_objects.csv`, `run_report.md`).
- **Synthesized**: Master report in `validation/results/phase2a_video_validation_report.md`.
- **Status**: Controlled recorded-video behavioral validation completed.

### 2026-10-04 (Phase 1A.1 — GPU Acceleration on RTX 4050)
- **Resolved**: Activated project virtual environment `.\.venv\Scripts\python.exe` with PyTorch 2.10.0+cu130, CUDA 13.0, and NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Installed**: Fixed missing dependencies in `.venv`: `ultralytics` (8.4.172), `opencv-python` (5.0.0.93), `scipy` (1.18.1), `pyyaml` (6.0.3), `pyttsx3` (2.99), `lap` (0.5.13).
- **Purged**: Removed redundant weights from `adaptive_navigation/models/` (reclaimed ~105 MB). Fixed `depth.py` checkpoint resolution to canonical `models/` directory.
- **Repaired**: Removed hardcoded CPU device strings from `tests/test_detector_clip.py`, `tests/test_tracker_clip.py`, `tests/test_depth_convention.py`.
- **Updated**: `adaptive_navigation/main.py` — added `--device` CLI override, canonical device resolution to `cuda:0` when available, and GPU VRAM tracking in telemetry.
- **Executed**: Full pipeline executed on `data/test_clip.mp4` on `cuda:0` in 6.06 seconds total runtime.
- **Measured**: Depth latency dropped from 728.04 ms to 70.49 ms (10.3× speedup); detector latency dropped from 32.25 ms to 18.57 ms; overall pipeline latency dropped from 772.57 ms to 101.04 ms (7.6× overall speedup, 11.07 FPS p50). Peak VRAM usage: 138.25 MB allocated / 438 MB reserved (<7.2% VRAM).
- **Generated**: `validation/logs/gpu_telemetry.json`, `gpu_telemetry_frames.csv`, `gpu_telemetry_objects.csv`, and `validation/results/gpu_smoke_test_report.md`.
- **Verified**: Full test suite (`python -m unittest discover tests`) passed 100% (11/11 tests).

### 2026-10-04 (Phase 1A — Real Pipeline Bring-Up)
- **Provisioned**: Official weights `models/detector/yolo11n.pt` and `models/depth/depth_anything_v2_vits.pth`.
- **Created**: `tests/test_detector_clip.py`, `tests/test_tracker_clip.py`, `tests/test_depth_convention.py`, `tests/test_ttc_consistency.py`.
- **Repaired**: `adaptive_navigation/risk/ttc.py` — added `depth_convention` parameter, implemented unified inverse-depth divergence TTC formula where unknown scale cancels out.
- **Repaired**: `adaptive_navigation/temporal/motion.py` — made approach classification respect both `higher_is_closer` and `lower_is_closer`.
- **Updated**: `adaptive_navigation/main.py` — added per-frame telemetry logging, deterministic box colors, and `depth_convention` passthrough.
- **Executed**: Real pipeline executed across all 60 frames of `data/test_clip.mp4` on CPU (`real_telemetry.*`).
