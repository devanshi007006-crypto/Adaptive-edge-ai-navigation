# Final Canva/InDesign Text Copy — Section 7: PROTOTYPE & RESULTS

*Copy and paste the exact text blocks below into Canva or InDesign poster text containers.*

---

## 1. Section Banner Text

**SECTION TITLE**: 7. PROTOTYPE & RESULTS  
**SUBTITLE**: Empirical Evaluation of Monocular Edge-AI Risk Reasoning Framework

---

## 2. Block A — Performance KPI Strip Text

### Card 1: Detection Latency
```text
YOLO26n DETECTOR
9.02 ms
Primary Object Detection Latency
[MEASURED • RTX 4050 LAPTOP GPU]
```

### Card 2: Depth Estimation Latency
```text
DEPTH TRT FP16
7.82 ms
TensorRT Monocular Depth Latency
[MEASURED • TENSORRT FP16 ENGINE]
```

### Card 3: End-to-End Pipeline Latency
```text
50.13 ms
p50 End-to-End Latency
Full Sensing to Audio Command Pipeline
[MEASURED • S03 PRELIMINARY RUN]
```

### Card 4: System Throughput
```text
17.68 FPS*
Preliminary Measured Throughput
(*S03 16-frame evaluation sequence)
[MEASURED • S03 PRELIMINARY RUN]
```

### Card 5: Engineering Verification
```text
17 / 17
Unit Tests Passing
100% Core Verification Rate
[MEASURED • VERIFIED SUITE]
```

---

## 3. Block B — Temporal Risk / TTC Telemetry Copy

**BLOCK TITLE**: Temporal Risk & Time-To-Collision (TTC) Behavior  
**TAG**: `MEASURED • S03 APPROACHING PERSON`

**EXPLANATORY TEXT**:
> Frame-by-frame measured telemetry during controlled sequence S03 (approaching person). As the target subject advances toward the camera, scale-invariant disparity TTC decreases continuously from 6.5s to 0.6s. The framework issues its initial acoustic warning at TTC = 3.0s, triggering dynamic risk escalation from LOW to CRITICAL prior to physical proximity threshold breach.

---

## 4. Block C — Baseline vs. Proposed Copy

**BLOCK TITLE**: Comparative Research Validation: Static Baseline vs. Proposed Framework  
**TAG**: `CONTROLLED PROTOCOL • S01–S06 SCENARIO SUITE`

**HEADLINE**:
> **100% false-warning reduction across the controlled S01–S06 scenario suite.**

**METRIC COMPARISON SUMMARY**:
- **False Warnings**: Static Proximity Baseline = 6 warnings | Proposed Dynamic Framework = **0 warnings**
- **Unnecessary STOP Commands**: Static Proximity Baseline = 4 commands | Proposed Dynamic Framework = **0 commands**
- **Warning Lead Time**: Static Baseline = 0.0s (reacts after proximity breach) | Proposed Dynamic Framework = **1.8s anticipation lead time**
- **Crossing/Receding Handling**: Static Baseline = Triggers false hazard | Proposed Dynamic Framework = **Correctly filters non-collision paths**

---

## 5. Block D — Risk-Aware Adaptive Computation Copy

**BLOCK TITLE**: Adaptive Depth Computation Scheduling  
**TAG**: `MEASURED • ADAPTIVE CADENCE`

**CAPTION & DATA**:
> Implemented risk-aware depth scheduling dynamically adjusts depth inference cadence based on real-time risk level:
> - **LOW RISK**: 4:1 Depth Cadence (Depth every 4th frame) → **24.5 FPS** throughput
> - **MEDIUM RISK**: 2:1 Depth Cadence (Depth every 2nd frame) → **18.2 FPS** throughput
> - **HIGH / CRITICAL RISK**: 1:1 Depth Cadence (Full per-frame depth) → **14.93 FPS** throughput

---

## 6. Block E — Real Prototype Montage Text

**BLOCK TITLE**: Real-Time Edge Prototype Execution Sequence  
**TAG**: `REAL PROTOTYPE EXECUTION • S03 RUN`

**PANEL CAPTIONS**:
1. **01_Detection**: YOLO26n detects approaching target (`person`, conf = 0.91).
2. **02_Tracking**: BoT-SORT establishes persistent track ID #1 across frames.
3. **03_Depth**: Depth Anything V2 TRT FP16 computes relative disparity map.
4. **04_TTC**: Timestamp-aware TTC estimator computes decreasing collision time ($TTC = 2.4	ext{s}$).
5. **05_Risk**: Dynamic Risk Engine escalates risk state to `HIGH` ($R = 0.85$).
6. **06_Navigation**: Spatial path engine issues `STOP / AVOID` via non-blocking TTS audio.

---

## 7. Mandatory Disclaimer & Provenance Notes

```text
PROVENANCE DISCLAIMER:
- Latency and throughput figures represent measured benchmark runs on an NVIDIA RTX 4050 Laptop GPU.
- Throughput of 17.68 FPS is derived from a preliminary 16-frame evaluation run on sequence S03 and is not presented as a universal system throughput across all deployment conditions.
- False warning reduction of 100% applies specifically to the controlled S01–S06 scenario evaluation suite.
```
