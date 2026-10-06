# Poster Section 7 Caption Manifest

*Exact captions to be placed directly beneath each corresponding visual figure on the research poster.*

---

### Figure 7.1 — Measured S03 Temporal Risk & TTC Progression
> **Figure 7.1: Measured Temporal Risk and TTC Telemetry.** Frame-by-frame measured time-to-collision (cyan) and dynamic risk score (red) recorded during controlled test sequence S03 (approaching person). As the subject advances, TTC decreases monotonically from 6.5s to 0.6s, triggering the initial acoustic warning at TTC = 3.0s (1.8s prior to proximity threshold breach).  
> *Tag*: `[MEASURED • S03 APPROACHING RUN • RTX 4050]`

---

### Figure 7.2 — Baseline vs. Proposed Framework Performance
> **Figure 7.2: Comparative Evaluation Across Controlled Scenario Suite S01–S06.** Quantitative performance comparison between the Static Proximity Baseline (detection + fixed depth threshold) and the Proposed Dynamic Framework (YOLO26n + BoT-SORT + TTC + dynamic risk). The proposed system eliminates 100% of false warnings and unnecessary STOP commands across the controlled suite.  
> *Tag*: `[CONTROLLED PROTOCOL • S01–S06 SCENARIO SUITE]`

---

### Figure 7.3 — Risk-Aware Adaptive Depth Computation
> **Figure 7.3: Implemented Risk-Aware Depth Scheduling Modes.** System throughput and depth inference cadence dynamically adapt to evaluated risk level: Low Risk (4:1 cadence, 24.5 FPS), Medium Risk (2:1 cadence, 18.2 FPS), and High/Critical Risk (1:1 cadence, 14.93 FPS).  
> *Tag*: `[MEASURED • ADAPTIVE CADENCE EVALUATION]`

---

### Figure 7.4 — Controlled Scenario Suite Risk Distribution
> **Figure 7.4: Test Scenario Risk Coverage.** Peak risk level distribution across the controlled evaluation suite: S01 Clear Path (Nominal), S02 Static Obstacle (Low), S03 Approaching Person (Critical), S04 Receding Person (Low/Filtered), S05 Crossing Subject (Medium), and S06 Head/Camera Motion (Compensated).  
> *Tag*: `[CONTROLLED PROTOCOL • S01–S06]`

---

### Figure 7.5 — Real Prototype Pipeline Montage
> **Figure 7.5: End-to-End Edge Prototype Pipeline Execution.** Sequential execution of the 15-stage monocular edge navigation architecture on real S03 video frames: (1) YOLO26n Detection, (2) BoT-SORT Tracking, (3) TensorRT FP16 Monocular Depth, (4) Scale-Invariant TTC, (5) Dynamic Risk Engine, and (6) Path Decision & Audio TTS.  
> *Tag*: `[REAL PROTOTYPE EXECUTION • RTX 4050 GPU]`
