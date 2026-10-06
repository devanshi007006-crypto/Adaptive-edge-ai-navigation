# Research Conclave Judge Demonstration Script (2-3 Minutes)

> **Document ID:** `validation/results/final_research_evidence/judge_demo/judge_demo_script.md`

---

## Demonstration Sequence

### 0:00 - 0:20: System Overview
- Present system title: *An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction*.
- Point out Layer 1 (Laptop Webcam) + Layer 2 (Edge-AI GPU Pipeline).

### 0:20 - 0:50: Live Pipeline Execution & YOLO26n
- Launch `python scripts/run/run_final_prototype.py`.
- Highlight primary object detector: **YOLO26n** ($9.0	ext{ ms}$ inference) and **TensorRT FP16 Monocular Depth Engine** ($7.8	ext{ ms}$).

### 0:50 - 1:30: Dynamic Collision Risk & TTC (Approaching Scenario S03)
- Play approaching person clip.
- Show **Scale-Invariant Time-to-Collision ($	au = d/\dot{d}$)** decreasing dynamically as risk escalates (`NO_WARNING` -> `CAUTION` -> `WARNING` -> `CRITICAL`).
- Demonstrate non-blocking audio advisory: *"Warning. Person approaching. Stop."*

### 1:30 - 2:00: Receding Suppression & Steering Guidance (S04 / S05)
- Show receding person: system detects positive depth derivative ($\dot{d} > 0$) and **suppresses false warnings**.
- Show crossing person: system evaluates spatial walking corridor and issues directional steering: **`LEFT`** / **`RIGHT`**.

### 2:00 - 2:30: Risk-Aware Adaptive Computation
- Explain adaptive cadence controller: LOW RISK triggers 4:1 light depth cadence ($24.5	ext{ FPS}$ throughput), saving compute while maintaining safety.

### 2:30 - 3:00: Poster Graphs & Conclusion
- Review poster comparative plots showing **100% false alert reduction** over static range sensors.
