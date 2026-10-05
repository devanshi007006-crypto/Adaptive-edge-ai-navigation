# Project File Manifest: Adaptive Multimodal Edge-AI Navigation

> **SECOND-LEVEL MASTER FILE MANIFEST**: This document serves as the second-level map beneath [`PROJECT_INDEX.md`](../PROJECT_INDEX.md). It records every active, archived, model, dataset, test, and documentation file in the repository, along with its owner area, purpose, status, and references.

---

## 1. Master File & Directory Index

| File / Directory Path | Purpose | Status | Owner Area | Referenced By | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `PROJECT_INDEX.md` | Master project index & single source of truth | **ACTIVE** | Root Docs | Developer / AI Assistants | Permanent source of truth |
| `README.md` | General repository overview & quickstart guide | **ACTIVE** | Root Docs | External Users / GitHub | Public landing documentation |
| `LICENSE` | Open-source MIT license terms | **ACTIVE** | Root Meta | Project | MIT Open Source License |
| `requirements.txt` | Core Python dependencies specification | **ACTIVE** | Environment | `pip install` | PyTorch, TensorRT, OpenCV, etc. |
| `pyproject.toml` | Build system & static checker configuration | **ACTIVE** | Build | setuptools / IDE | Project packaging configuration |
| `pyrefly.toml` | Pyrefly language server configuration | **ACTIVE** | IDE Config | Pyrefly IDE | Type checker rules |
| `.gitignore` | Git exclusion rules for large models & binaries | **ACTIVE** | Version Control | Git | Excludes `.venv`, `*.engine`, `raw/` |
| `main.py` | Unified CLI application entry point | **ACTIVE** | Core Pipeline | CLI Users | CLI wrapper calling `adaptive_navigation.main` |
| `audit_report.md` | Forensic scientific & codebase audit report | **ACTIVE** | Documentation | Research Conclave | Phase 0 audit findings |
| `metric_traceability.md` | Metric provenance tracing database | **ACTIVE** | Documentation | Research Conclave | Backward trace of all metrics |
| `evaluation_plan.md` | Scientific triad validation architecture | **ACTIVE** | Documentation | Research Conclave | Mode A/B/C validation plan |

---

## 2. Core Packages (`adaptive_navigation/` & `src/`)

| File / Directory Path | Purpose | Status | Owner Area | Referenced By | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `adaptive_navigation/__init__.py` | Core package initializer & path registration | **ACTIVE** | Core Package | All modules | Exposes package version `1.0.0` |
| `adaptive_navigation/main.py` | Procedural 15-stage perception pipeline loop | **ACTIVE** | Core Pipeline | `main.py`, CLI | Full pipeline orchestrator |
| `adaptive_navigation/config.yaml` | Package fallback default configuration | **ACTIVE** | Configuration | `adaptive_navigation.main` | Package default YAML |
| `adaptive_navigation/audio/` | Speech synthesis & TTS output module | **ACTIVE** | Audio Subsystem | `main.py`, `run_live_camera.py` | Pyttsx3 async queue engine |
| `adaptive_navigation/hardware/` | Audio hardware device supervisor & watchdog | **ACTIVE** | Hardware Subsystem| `main.py` | Device manager & sink abstraction |
| `adaptive_navigation/motion/` | Camera motion re-export package | **ACTIVE** | Motion Subsystem | Package consumers | Re-exports `CameraMotionEstimator` |
| `adaptive_navigation/navigation/` | Spatial clearance & path geometry engine | **ACTIVE** | Navigation | `main.py`, `run_live_camera.py` | Evasive steering logic & corridor |
| `adaptive_navigation/perception/` | Vision ingest, YOLO, BoT-SORT & TRT Depth | **ACTIVE** | Perception | `main.py`, `run_live_camera.py` | OpenCV, YOLO11n, TRT Depth Anything |
| `adaptive_navigation/risk/` | Scale-invariant TTC & multi-factor risk engine | **ACTIVE** | Risk Subsystem | `main.py`, `run_live_camera.py` | Fused 5-feature risk score |
| `adaptive_navigation/temporal/` | Observation history & camera flow compensation | **ACTIVE** | Temporal Subsystem| `main.py`, `run_live_camera.py` | Sparse optical flow & motion EMA |
| `adaptive_navigation/uncertainty/` | Multi-source reliability assessment engine | **ACTIVE** | Uncertainty | `main.py`, `run_live_camera.py` | Evidence coverage & reliability |
| `adaptive_navigation/warning/` | Temporal warning state machine & speech formatter | **ACTIVE** | Warning Subsystem | `main.py`, `run_live_camera.py` | 2-frame escalation hysteresis |
| `src/__init__.py` | Target architecture namespace wrapper | **ACTIVE** | Source Layout | Package consumers | Maps `src` to `adaptive_navigation` |

---

## 3. Configuration Profiles (`configs/`)

| File Path | Profile Purpose | Status | Owner Area | Referenced By | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `configs/final.yaml` | Production reference profile | **ACTIVE** | Configuration | `main.py --mode final` | Canonical release profile |
| `configs/development.yaml` | Interactive GUI demo profile | **ACTIVE** | Configuration | `main.py --mode demo` | Visual window with HUD |
| `configs/evaluation.yaml` | Headless deterministic evaluation profile | **ACTIVE** | Configuration | `main.py --mode research` | Fixed seed evaluation profile |
| `configs/real_world.yaml` | Field testing profile with gait filtering | **ACTIVE** | Configuration | `main.py --mode real_world` | Wearable bluetooth profile |
| `configs/deployment.yaml` | Edge deployment profile with FP16 & 2:1 depth | **ACTIVE** | Configuration | `main.py --mode deployment` | Lightweight edge profile |
| `configs/final_experiment_config.yaml`| Parameter overrides for ablation matrix | **ACTIVE** | Configuration | Benchmark scripts | Parameter sweep config |

---

## 4. Models & Deployment Artifacts (`models/`)

| File Path | Purpose | Precision | Status | Owner Area | Current Usage |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `models/detector/yolo11n.pt` | Ultralytics YOLO11n PyTorch weights | FP32 | **ACTIVE MODEL** | Detector | 2D obstacle detection |
| `models/detector/yolo11n.onnx` | Exported YOLO11n ONNX representation | FP32 | **ACTIVE MODEL** | Deployment | TensorRT engine compilation |
| `models/depth/depth_anything_v2_vits.pth` | Depth Anything V2 ViT-S PyTorch weights | FP32 | **ACTIVE MODEL** | Depth | PyTorch reference depth |
| `models/deployment/yolo11n.onnx` | Deployment YOLO11n ONNX file | FP32 | **DEPLOYMENT** | Deployment | ORT CUDA / TensorRT |
| `models/deployment/yolo11n_fp16.engine` | Native TensorRT FP16 YOLO11n Engine | FP16 | **DEPLOYMENT** | Deployment | Native TensorRT 11.3 |
| `models/deployment/depth_anything_v2_vits.onnx` | Deployment Depth Anything V2 ONNX file | FP32 | **DEPLOYMENT** | Deployment | ORT CUDA / TensorRT |
| `models/deployment/depth_anything_v2_vits_fp16.engine`| Native TensorRT FP16 Depth Engine | FP16 | **DEPLOYMENT** | Deployment | Native TRT (33.8 ms) |

---

## 5. Execution & Utility Scripts (`scripts/`)

| File Path | Category | Purpose | Status | Owner Area | Notes |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `scripts/run/run_live_camera.py` | Live Runner | Interactive live webcam navigation prototype | **ACTIVE** | Execution | Webcam 0, TensorRT Depth, TTS |
| `scripts/run/run_pipeline.py` | Pipeline Runner | Main pipeline wrapper | **ACTIVE** | Execution | Root CLI helper |
| `scripts/run/run_phase2a_validation.py` | Benchmark | Phase 2A baseline real GPU runner | **ACTIVE** | Validation | 5 videos (5,235 frames) |
| `scripts/run/run_phase2b_validation.py` | Benchmark | Phase 2B failure mode revalidation runner | **ACTIVE** | Validation | Class policy & regression |
| `scripts/run/run_phase2c_validation.py` | Benchmark | Phase 2C 2:1 depth & ego-motion runner | **ACTIVE** | Validation | Subsampled depth cadence |
| `scripts/run/run_phase3a_heads_up_validation.py` | Benchmark | Phase 3A HEADS-UP exploratory runner | **ACTIVE** | Validation | 250 egocentric frames |
| `scripts/run/run_phase3b_heads_up_validation.py` | Benchmark | Phase 3B clean HEADS-UP runner | **ACTIVE** | Validation | 1,418 metadata frame slots |
| `scripts/run/run_phase3b1_temporal_revalidation.py`| Benchmark | Phase 3B.1 timestamp-aware runner | **ACTIVE** | Validation | Source frame timing correction |
| `scripts/run/run_phase4a_depth_calibration.py` | Benchmark | Phase 4A metric depth calibration runner | **ACTIVE** | Validation | Affine inverse scale fit |
| `scripts/run/run_phase4b_external_depth_validation.py`| Benchmark | Phase 4B external depth validation runner | **ACTIVE** | Validation | 3,919 external observations |
| `scripts/run/run_phase4c_tensorrt_optimization.py` | Benchmark | Phase 4C ORT CUDA FP16 runner | **ACTIVE** | Validation | ORT CUDA FP16 benchmark |
| `scripts/run/run_phase4c1_native_tensorrt_audit.py` | Benchmark | Phase 4C.1 Three-Way Benchmark runner | **ACTIVE** | Validation | PyTorch vs ORT vs Native TRT |
| `scripts/tools/check_environment.py` | Diagnostic Tool | System, CUDA, PyTorch & weight checker | **ACTIVE** | Tools | Pre-flight diagnostics |
| `scripts/tools/download_clean_single_stream.py` | Dataset Tool | Clean HEADS-UP archive slice downloader | **ACTIVE** | Tools | Stream archive downloader |
| `scripts/tools/download_heads_up_unconstrained.py` | Dataset Tool | Gated HEADS-UP archive downloader | **ACTIVE** | Tools | Reads `HF_TOKEN` from env |
| `scripts/tools/extract_heads_up_phase3b.py` | Dataset Tool | Verified frame range extractor | **ACTIVE** | Tools | Extracts 8 verified ranges |
| `scripts/tools/extract_heads_up_sequences.py` | Dataset Tool | Sequence extractor utility | **ACTIVE** | Tools | Image frame extractor |
| `scripts/tools/finalize_heads_up_sequences.py` | Dataset Tool | Sequence MP4 video compiler | **ACTIVE** | Tools | Compiles sequence MP4s |
| `scripts/tools/generate_phase3b_plots.py` | Diagnostic Tool | Plot generator for Phase 3B | **ACTIVE** | Tools | Visual plot generator |
| `scripts/tools/resume_heads_up_unconstrained.py` | Dataset Tool | Resumable chunk downloader | **ACTIVE** | Tools | Retry logic for downloads |

---

## 6. Test Suite (`tests/`)

| File Path | Purpose | Status | Owner Area | Verification Method |
| :--- | :--- | :---: | :--- | :--- |
| `tests/test_imports.py` | Package module import regression test | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_detector_clip.py` | YOLO11n inference verification on test clip | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_tracker_clip.py` | BoT-SORT track continuity verification | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_depth_convention.py` | Monocular depth output convention test | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_ttc_consistency.py` | Scale-invariant optical TTC sanity test | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_phase2b_stabilization.py` | Class policy & gait spike rejection test | **ACTIVE** | Test Suite | `unittest discover` |
| `tests/test_phase2c_egomotion.py` | Optical flow ego-motion compensation test | **ACTIVE** | Test Suite | `unittest discover` |

---

## 7. Validation Infrastructure (`validation/`)

| Path | Purpose | Status | Owner Area | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `validation/videos/` | 5 controlled indoor real-world test videos | **ACTIVE DATA** | Validation | 5,235 frames (89.3s total) |
| `validation/datasets/heads_up/` | HEADS-UP dataset adapter & metadata CSVs | **ACTIVE DATA** | Validation | Tracked lightweight metadata |
| `validation/results/video_runs/` | Phase 2A baseline real GPU telemetry | **ACTIVE EVIDENCE** | Validation | 5 per-video folders & report |
| `validation/results/video_runs_phase2b/` | Phase 2B revalidated telemetry | **ACTIVE EVIDENCE** | Validation | 5 per-video folders & report |
| `validation/results/video_runs_phase2c/` | Phase 2C 2:1 depth telemetry | **ACTIVE EVIDENCE** | Validation | 5 per-video folders & report |
| `validation/results/video_runs_phase2c_egomotion/`| Phase 2C ego-motion compensated telemetry | **ACTIVE EVIDENCE** | Validation | 5 per-video folders & report |
| `validation/results/heads_up/` | Phase 3A/3B/3B.1 GPU telemetry & reports | **ACTIVE EVIDENCE** | Validation | 8 sequence folders & reports |
| `validation/results/phase4/` | Phase 4A/4B/4C/4C.1 reports, CSVs, JSONs, plots | **ACTIVE EVIDENCE** | Validation | Calibration, ORT & TRT audit |
| `validation/results/live/` | Live webcam prototype reports, logs & frames | **ACTIVE EVIDENCE** | Validation | Baseline/fixed JSONs & frames |

---

## 8. Archived & Historical Assets (`archive/`)

| Path | Category | Purpose | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `archive/debug/` | Diagnostic Scripts | 20 one-off diagnostic scripts from `scratch/` | **ARCHIVED DEBUG** | Diagnostic history preserved |
| `archive/obsolete/evaluation/` | Legacy Code | Pre-Phase 0 synthetic evaluation scripts | **ARCHIVED OBSOLETE**| Synthetic noise generators |
| `archive/obsolete/final_results/` | Legacy CSVs | Legacy synthetic evaluation CSVs | **ARCHIVED OBSOLETE**| Synthetic metrics tables |
| `archive/obsolete/adaptive_navigation_evaluation/`| Legacy Code | Duplicate copy of synthetic eval subpackage | **ARCHIVED OBSOLETE**| Deprecated subpackage |
| `archive/obsolete/FINAL_VALIDATION_SUMMARY.md`| Legacy Report | Report claiming synthetic 86.6 FPS | **ARCHIVED OBSOLETE**| Superseded by real GPU |
| `archive/legacy/` | Legacy Prototypes | Early stage 1 prototype files | **ARCHIVED LEGACY** | Historical prototype code |
| `presentation/` | Presentation | Presentation slides & figures | **ARCHIVED ASSET** | Academic presentation text |
| `poster/` | Poster | Conference poster text & figures | **ARCHIVED ASSET** | Academic poster text |
