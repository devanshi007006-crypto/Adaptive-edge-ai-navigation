# Baseline Comparison Protocol: Static Proximity vs Dynamic Edge-AI Navigation

> **Document ID:** `docs/research/baseline_comparison_protocol.md`  
> **Status:** Active Protocol Standard  
> **Date:** October 5, 2026  
> **Target System:** Adaptive Monocular Edge-AI Navigation Framework (`v1.4.0-final`)

---

## 1. Executive Summary & Objective

To prove the core scientific hypothesis of this research—that **dynamic Time-to-Collision (TTC), camera ego-motion compensation, and spatial walking corridor analysis reduce false warnings and increase warning lead time compared to conventional range/proximity sensors**—this protocol defines a rigorous baseline comparison study.

We compare a representative **Static Proximity Baseline** against the **Proposed Dynamic Edge-AI Framework** across identical video streams and physical trajectories.

---

## 2. System Definitions

### 2.1 Baseline Architecture (Static Proximity + Single-Frame Detection)
Simulates traditional ultrasonic/LiDAR or static vision-based Electronic Travel Aids (ETAs):
- **Component 1**: Object Detection (YOLO11n, single frame).
- **Component 2**: Monocular Depth Estimation (Depth Anything V2, single frame).
- **Component 3**: Static Threshold Warning:
  - If any detected object depth $d \le 1.5\text{ m} \implies \text{CRITICAL}$ (Continuous Beep/Alert).
  - If any detected object depth $d \in (1.5, 3.0]\text{ m} \implies \text{CAUTION}$.
  - Otherwise $\implies \text{NO\_WARNING}$.
- **Omissions**: No tracking (BoT-SORT disabled), no temporal velocity/slope estimation, no camera motion compensation, no TTC calculation, no lateral corridor spatial analysis.

### 2.2 Proposed Framework (Full Dynamic Multimodal Architecture)
- **Full Pipeline**: YOLO11n + BoT-SORT Tracking + Depth Anything V2 + Temporal History (maxlen=25) + Motion Compensation (Lucas-Kanade Flow) + Scale-Invariant TTC ($\tau = d / \dot{d}$) + Dynamic Risk Engine + Spatial Path Engine + Hysteresis Warning State Machine.

---

## 3. Comparison Metrics & Evaluation Criteria

| Metric | Scientific Significance | Target Outcome (Proposed vs Baseline) |
|:---|:---|:---|
| **Warning Lead Time ($\Delta t_{\text{lead}}$)** | Time available for user to react before hazard boundary ($1.5\text{ m}$) | $\ge 0.8\text{ s}$ earlier alert for fast-approaching threats |
| **False Warning Rate** | Acoustic fatigue & alert flickering rate in clear/receding paths | $\ge 70\%$ reduction in false/unnecessary alerts |
| **Receding Object Suppression** | Ability to ignore objects moving away from the user | $100\%$ suppression of alerts for receding pedestrians |
| **Head/Camera Sway Resilience** | False positive rate during head panning/walking gait | $\ge 80\%$ reduction in false alerts caused by body motion |
| **Navigation Direction Accuracy** | Correctness of spatial avoidance guidance (`AVOID_LEFT`/`RIGHT`) | $\ge 85\%$ directional compliance with safe corridor |
| **End-to-End Latency** | Computational burden of dynamic temporal pipeline | Pipeline latency $< 55\text{ ms}$ (throughput $> 14\text{ FPS}$) |

---

## 4. Controlled Scenario Evaluation Plan

Both systems process identical recorded video streams from the S01–S06 scenario suite:

1. **S01 — Clear Path**: Measure false positive alert rate per minute.
2. **S02 — Static Obstacle**: Measure distance accuracy and stability of warning boundary.
3. **S03 — Person Approaching**: Measure warning lead time $\Delta t_{\text{lead}}$ when person approaches at $v \approx 1.2\text{ m/s}$.
4. **S04 — Person Receding**: Measure alert suppression (Baseline will incorrectly trigger; Proposed should remain silent).
5. **S05 — Person Crossing**: Measure lateral path analysis and safe steering direction advisory.
6. **S06 — Camera Movement**: Measure false alert suppression during head tilt/pan.

---

## 5. Implementation via Configuration Modular Switches

The comparison is executed without altering codebase logic by running `main.py` with modular config overrides:

```bash
# 1. Run Baseline System Evaluation
python main.py --config configs/baseline_static.yaml --video data/scenarios/S03_approaching.mp4 --output outputs/baseline_S03.json

# 2. Run Proposed Dynamic System Evaluation
python main.py --config configs/final.yaml --video data/scenarios/S03_approaching.mp4 --output outputs/proposed_S03.json
```

---

## 6. Output Deliverables & Reporting Format

Results are compiled into `validation/results/baseline_comparison_report.md` with comparative summary tables, precision-recall curves, and lead-time boxplots.
