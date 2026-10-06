# Final Results Forensic Reconciliation Report
**Project**: Adaptive Edge-AI Monocular Navigation  
**Document ID**: `validation/results/final_research_evidence/reports/RESULT_RECONCILIATION.md`  
**Date**: October 6, 2026  
**Status**: Formal Forensic Audit Complete  
**Authoritative Evidence Source**: `validation/results/phase6/ground_truth_trials.csv` & `phase6_metrics.json`

---

## 1. Executive Summary & Audit Purpose

This document provides a ground-truth forensic reconciliation of all quantitative metrics across the evaluation pipeline. A discrepancy audit was conducted to resolve conflicting values between raw telemetry CSVs, Phase 6 evaluation logs, evidence audit reports, and informal poster summary text drafts.

### Authoritative Hierarchy Applied:
1. **Raw Telemetry & Event Logs** (`ground_truth_trials.csv`, `ground_truth_events.csv`)
2. **Phase 6 Structured Metrics** (`phase6_metrics.json`)
3. **Formal Phase 6 Audit Report** (`phase6_report.md`)
4. **Final Evidence Audit Report** (`FINAL_EVIDENCE_AUDIT.md`)
5. **Poster Summary Text & Markdown Drafts** *(Informal text drafts overrideable by raw CSVs)*

---

## 2. Reconciled Metrics Audit Table

| Metric | Raw Telemetry / CSV Value | Informal Text Summary Value | Resolved Reconciled Value | Source Experiment & Denominator | Poster-Safe Status | Mandatory Qualifier Tag |
|---|:---:|:---:|:---:|---|:---:|---|
| **False Warning Rate** | **91.60 alerts/min** (229 frames / 2.5 min) | "6 false warnings" | **91.60 alerts/min $\rightarrow$ 0.00** | Phase 6 (30 trials, 150s total duration across S01–S06) | **YES** | `CONTROLLED PROTOCOL • S01–S06` |
| **Unnecessary STOP Advisories** | **9 frames** (S04 receding trials) | "4 unnecessary STOPs" | **9 frames $\rightarrow$ 0 frames** | Phase 6 (30 trials, 5.0s per trial, S04 receding suite) | **YES** | `CONTROLLED PROTOCOL • S04 RECEDING` |
| **Warning Lead Time** | **1.50s (Proposed) vs 1.43s (Baseline)** | "1.8s lead time" | **+0.07s earlier median reaction** ($1.50\text{s}$ vs $1.43\text{s}$) | Phase 6 (Median across S03 approaching & S05/S06 trials) | **YES** | `CONTROLLED PROTOCOL • S03 APPROACH` |
| **Missed Hazards** | **Baseline: 5 \| Proposed: 10** | "0 missed hazards" | **Baseline: 5 \| Proposed: 10** | Phase 6 (S02 static obstacle: 5 miss; S06 camera sway: 5 miss) | **YES** *(Honest Trade-off)* | `CONTROLLED PROTOCOL • LIMITATION` |
| **Navigation Accuracy** | **Baseline: 45.6% \| Proposed: 53.5%** | "53.5% accuracy" | **45.6% $\rightarrow$ 53.5% (+7.9% gain)** | Phase 6 (30 trials, directional guidance evaluation) | **YES** | `CONTROLLED PROTOCOL • S01–S06` |
| **YOLO26n Latency** | **9.02 ms** | "9.02 ms" | **9.02 ms** | S03 evaluation benchmark (NVIDIA RTX 4050 GPU) | **YES** | `MEASURED • RTX 4050` |
| **Depth TRT FP16 Latency** | **7.82 ms** | "7.82 ms" | **7.82 ms** | TensorRT FP16 Engine benchmark (RTX 4050 GPU) | **YES** | `MEASURED • TENSORRT FP16` |
| **p50 End-to-End Latency** | **50.13 ms** | "50.13 ms" | **50.13 ms** | S03 16-frame preliminary evaluation run | **YES** | `MEASURED • S03 PRELIMINARY RUN` |
| **System Throughput** | **17.68 FPS** | "17.68 FPS" | **17.68 FPS** | S03 preliminary 16-frame evaluation sequence | **YES** | `MEASURED • S03 PRELIMINARY RUN` |
| **Unit Verification Rate** | **17 / 17 (100%)** | "17/17 tests passed" | **17 / 17 passed** | Automated Pytest suite (`tests/`) | **YES** | `MEASURED • VERIFIED SUITE` |

---

## 3. Forensic Investigation of Specific Discrepancies

### A. False Warnings: `91.60 alerts/min vs 0.00` vs `"6 false warnings"`
- **Origin of `91.60 alerts/min`**: In `ground_truth_trials.csv`, System A (Static Baseline) generated 229 false warning frames across 30 trials (150 seconds total evaluation time). $229 / (150 / 60) = 91.60\text{ alerts/min}$. System B (Proposed) produced 0 false warning frames ($0.00\text{ alerts/min}$).
- **Origin of `"6 false warnings"`**: An informal text copy draft summarized "6 scenarios" or "6 baseline alert episodes" as "6 false warnings".
- **Resolution**: **91.60 alerts/min vs 0.00 (100% false warning reduction across S01–S06)** is the authoritative, mathematically exact metric. The informal text phrase "6 false warnings" is declared **UNSAFE** and replaced.

### B. Unnecessary STOP Advisories: `9 frames vs 0` vs `"4 unnecessary STOPs"`
- **Origin of `9 frames`**: In `ground_truth_trials.csv`, S04 (Person Receding) generated 9 unnecessary `STOP` advisory frames under System A (Baseline: T01=1, T02=2, T03=3, T04=0, T05=3 frames). System B (Proposed) produced 0 unnecessary `STOP` frames.
- **Origin of `"4 unnecessary STOPs"`**: Informal copy draft estimated 4 event occurrences instead of counting exact frame instances.
- **Resolution**: **9 frames vs 0 frames (100% unnecessary STOP elimination)** is the authoritative metric. The text phrase "4 unnecessary STOPs" is declared **UNSAFE** and replaced.

### C. Warning Lead Time: `1.50s vs 1.43s (+0.07s)` vs `"1.8s lead time"`
- **Origin of `1.50s vs 1.43s`**: Calculated from `ground_truth_trials.csv` as $\Delta t_{\text{lead}} = t_{\text{GT\_hazard}} - t_{\text{first\_warning}}$. Baseline median lead time across valid hazard trials was $1.43\text{s}$; Proposed median lead time was $1.50\text{s}$ (net improvement of $+0.07\text{s}$ earlier reaction window).
- **Origin of `"1.8s lead time"`**: Single-frame uncalibrated estimate from an early preliminary S03 single-run chart.
- **Resolution**: **+0.07s earlier median reaction window ($1.50\text{s}$ Proposed vs $1.43\text{s}$ Baseline)** is the authoritative metric. The uncalibrated text "1.8s lead time" is declared **UNSAFE** and replaced.

### D. Missed Hazards Investigation: `Baseline = 5` vs `Proposed = 10`
- **Forensic Audit Findings**:
  - **Static Baseline (System A)**: Missed 5 hazards, all occurring in Scenario S02 (Stationary Obstacle). Because the obstacle was stationary at $d > 3.0\text{m}$, the static proximity threshold ($d \le 3.0\text{m}$) did not trigger during the $2.0\text{s}$ hazard window ($5 \times 1 = 5$ missed hazards).
  - **Proposed System (System B)**: Missed 10 hazards: 5 in S02 (Stationary Obstacle) + 5 in S06 (Head / Camera Sway).
  - **Root Cause of S06 Missed Hazards**: In S06, the user experienced severe head/camera sway while approaching a stationary obstacle. The Proposed system's Lucas-Kanade optical flow background motion compensation correctly identified background motion, BUT because the obstacle was static relative to the room, the ego-motion compensation filtered out the relative displacement as camera sway, suppressing the alert (`first_warning_sec = None`).
- **Scientific Conclusion**: This is **REAL SYSTEM BEHAVIOR** and an important algorithmic trade-off. Lucas-Kanade background motion compensation successfully eliminates false alerts during head shake (S06), but can temporarily suppress warnings on stationary obstacles during intense camera sway.
- **Poster Policy**: In accordance with scientific integrity rules, **this trade-off MUST NOT be hidden**. It is included in the reconciled findings as a documented framework limitation.

### E. Navigation Accuracy: `Baseline = 45.6%` vs `Proposed = 53.5%`
- **Origin**: Computed as the proportion of frames where the issued steering advisory matched the optimal ground-truth corridor direction (`FORWARD`, `LEFT`, `RIGHT`, `STOP`).
- **Result**: Baseline achieved $45.64\%$; Proposed achieved $53.53\%$ (+7.89% net gain), driven primarily by successful `AVOID_LEFT` steering advisories in S05 (Crossing Person).

---

## 4. Telemetry Plot Audit (`poster_box_07/real_ttc_plot.png`)

- **Telemetry Source**: 16-frame sequence from S03 approaching run.
- **Audit Classification**: The curve data represents an **EMPIRICAL S03 TELEMETRY SERIES** recorded during S03 evaluation.
- **Provenance Label**: `MEASURED • S03 APPROACHING SEQUENCE`.
- **Integrity Status**: Fully verified. No synthetic or theoretical curves are represented as raw sensor telemetry.
