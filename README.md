# Adaptive-edge-ai-navigation
An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users.

## Pipeline Architecture
CAMERA -> PERCEPTION (YOLO + BoT-SORT + Depth) -> TEMPORAL (History + Motion + Camera Compensation) -> RISK (TTC + Multi-factor Risk + Uncertainty/Reliability) -> NAVIGATION -> FEEDBACK -> SPEAKER

## Implementation Progress
- [x] **Step 1 — Project Skeleton & Setup**: Structured package layout, configs, and typed data interfaces.
- [x] **Step 2 — Camera & Preprocessing Layer**: Thread-safe frame acquisition, resolution resizing, illumination validation, timestamping.
- [x] **Step 3 — YOLO Object Detection**: Ultralytics YOLO11n integration with CPU/CUDA execution, bounding box normalization, and class filtering.
- [x] **Step 4 — BoT-SORT Object Tracking**: Persistent track IDs across consecutive frames with trajectory buffer and lost track handling.
- [x] **Step 5 — Depth Estimation (Depth Anything V2)**: Lightweight ViT-S monocular depth estimation, median obstacle depth extraction, and explicit relative vs metric typing.
- [x] **Step 6 — Temporal History Buffer**: Rolling observation queues, FIFO eviction, missing frame tracking, and temporal metrics.
- [x] **Step 7 — Velocity, Approach Estimation & Smoothing**: Bounding-box image plane velocity, relative depth rate, approach classification (APPROACHING, RECEDING, STATIONARY), and EMA temporal smoothing.
- [x] **Step 8 — Camera Motion Compensation (Optical Flow)**: Lucas-Kanade sparse feature tracking, RANSAC rigid affine transformation, and ego-motion subtraction.
- [x] **Step 9 — Time-to-Collision (TTC) Estimation**: Constant-velocity projection, calibrated metric seconds calculation, and explicit relative closing flags (RELATIVE_DEPTH_ONLY).
- [x] **Step 10 — Multi-Factor Risk Assessment Engine**: Multi-criteria hazard scoring fusing TTC, depth, approach dynamics, path corridor relevance, class criticality weights, and evidence coverage.
- [x] **Step 11 — Uncertainty & Reliability Estimation Layer**: 8-component runtime evidence quality evaluation, composite heuristic reliability and uncertainty indicators, consistency conflict detection, and global system health monitoring.
- [x] **Step 12 — Temporal Risk Stabilization & Warning Decision State Machine**: Per-track risk history, persistence counters, hysteresis thresholds, reliability gating, track disappearance grace period, and global threat priority selector.
- [x] **Step 13 — User-Facing Warning & Audio/TTS Layer**: Structured natural language alert message generator, repeat suppression, priority preemption, and offline-first TTS engine (pyttsx3/SAPI5).
- [x] **Step 14 — Spatial Position, Path Geometry & Safe Navigation Decision Engine**: Normalized 2D spatial zoning, walking corridor overlap, lateral free-space occupancy estimation, two-sided safe-path validation, and direction switching hysteresis.
