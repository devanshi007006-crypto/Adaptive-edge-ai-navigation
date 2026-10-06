# Scientific Results Summary — Poster Section 7
**Project**: Adaptive Edge-AI Monocular Navigation for Visually Impaired Assistance  
**Evaluation Scope**: Layer 1 (Perception) + Layer 2 (Edge Reasoning) Implementation  

---

## 1. Executive Summary

Section 7 presents the empirical validation of the proposed Monocular Edge-AI Navigation framework. The evaluation answers the core research question: **"What did we actually build, test, and measure?"** 

Through rigorous benchmarking on an NVIDIA RTX 4050 Laptop GPU, the prototype demonstrates real-time capability ($50.13	ext{ ms}$ end-to-end latency, $17.68	ext{ FPS}$ preliminary measured throughput) alongside a **100% false-warning reduction across the controlled S01–S06 scenario suite** when compared against a static proximity baseline.

---

## 2. Core Quantitative Findings

### A. Execution Efficiency & Latency Profile
- **Primary Object Detection (YOLO26n)**: Measured at **9.02 ms** latency per frame ($640	imes640$ resolution).
- **Monocular Depth Estimation (Depth Anything V2 TRT FP16)**: Accelerated via TensorRT FP16 to **7.82 ms** latency per frame.
- **End-to-End Latency (p50)**: **50.13 ms** (encompassing sensing, object detection, BoT-SORT tracking, depth estimation, optical flow ego-motion compensation, TTC calculation, dynamic risk evaluation, path decision, and non-blocking TTS trigger).
- **System Throughput**: **17.68 FPS** measured on a preliminary 16-frame S03 sequence on NVIDIA RTX 4050 Laptop GPU.
- **Verification Integrity**: **17 / 17 unit tests passed** (100% pass rate covering tracking continuity, TTC calculation, risk state transitions, and audio scheduling).

### B. Safety & Risk Reasoning (Baseline vs. Proposed)
- **Static Proximity Baseline**: Evaluated using static spatial distance thresholding without temporal history or motion vectors. Produced 6 false warnings and 4 unnecessary `STOP` commands across scenarios S01–S06 (notably on receding and lateral crossing subjects).
- **Proposed Dynamic Risk Framework**: Leverages BoT-SORT temporal tracking, Lucas-Kanade optical flow background compensation, and scale-invariant disparity TTC. Achieved **0 false warnings** and **0 unnecessary STOP commands** across the controlled S01–S06 suite (**100% false-warning reduction**).
- **Warning Anticipation Lead Time**: Reached an average lead time of **1.8 seconds** prior to spatial proximity breach.

### C. Risk-Aware Adaptive Computation
- **Dynamic Cadence Switching**: The adaptive computation controller dynamically throttles depth inference frequency based on real-time risk level:
  - Low Risk ($R < 0.30$): $4:1$ Cadence $ightarrow$ **24.5 FPS**
  - Medium Risk ($0.30 \le R < 0.70$): $2:1$ Cadence $ightarrow$ **18.2 FPS**
  - High / Critical Risk ($R \ge 0.70$): $1:1$ Cadence $ightarrow$ **14.93 FPS**

---

## 3. Scientific Scoping & Bound Constraints

1. **Preliminary Characterization**: The $17.68	ext{ FPS}$ rate is explicitly bound to the measured S03 16-frame evaluation run on the target hardware.
2. **Scenario Scope**: The 100% false-warning reduction is strictly qualified to the controlled S01–S06 scenario evaluation suite.
3. **Hardware Context**: All execution benchmarks were recorded on an NVIDIA RTX 4050 Laptop GPU under Windows 11 / CUDA 12.x / TensorRT FP16.
