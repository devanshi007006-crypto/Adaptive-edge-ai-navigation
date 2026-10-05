# Phase 3B — HEADS-UP Egocentric Systematic Behavioral Validation Report

> **AUTHORITATIVE PHASE 3B VALIDATION REPORT**  
> **Project Title**: An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users  
> **Short Name**: `Adaptive-edge-ai-navigation`  
> **Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`) | Intel Core i7  
> **Execution Date**: October 5, 2026  
> **Dataset**: Official HEADS-UP Egocentric Benchmark (*Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems*, arXiv:2409.20324v1, EPFL VITA Lab)  
> **Pipeline Baseline**: Phase 2C (YOLO11n + BoT-SORT + Depth Anything V2 2:1 Cadence + Optical-Flow Ego-Motion Compensation + Dynamic Risk Engine)  
> **Audit Classification**: **READY WITH SCIENTIFIC LIMITATION (Option B)**

---

## 1. Executive Summary & Scientific Scope

Phase 3B provides an **exploratory external behavioral validation** of the Phase 2C perception, temporal-risk, and navigation pipeline on an **8-sequence behaviorally stratified evaluation suite** selected from the HEADS-UP `unconstrained` subset.

### Key Quantitative Findings:
- **Frame Scope & Provenance**: Evaluated **852 valid extracted image frames** across **1,418 selected metadata slots** ($121 + 151 + 121 + 221 + 201 + 201 + 201 + 201$).
- **Real GPU Throughput**: Achieved an aggregate real throughput of **8.85 FPS** ($96.23\text{ seconds}$ total wall-clock execution time) on an **NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`)** with a median (p50) latency of **122.51 ms** and 95th percentile (p95) latency of **172.60 ms**.
- **Perception & Indoor Class Filtering**: The detector produced 5,509 raw COCO-80 proposals; the indoor navigation class policy suppressed **1,179 background clutter detections (22.0% suppression rate)** while retaining 4,330 active navigation hazard proposals.
- **Tracking Continuity**: BoT-SORT tracked **98 persistent obstacle trajectories**, achieving up to **100% track persistence** (126 continuous frames) on primary hazards.
- **Kinematic TTC & Motion Classification**: Generated **2,258 valid Time-to-Collision ($\tau = d/\dot{d}$) step calculations**, recording system classification outputs of 972 approaching closing hazard observations, 546 receding targets, and 740 static objects.
- **Navigation Decision Stability**: Achieved **95.68% consecutive-frame navigation decision consistency**, demonstrating internal decision stability without high-frequency command chatter.

### Mandatory Scientific Language Demarcation:
This evaluation represents **sample-level egocentric behavioral validation** on pre-recorded head-mounted video sequences. It demonstrates **observed system behavior** under head motion, pedestrian density, and dynamic interactions. It does **NOT** constitute clinical trial validation, 100% safety guarantees, or universal domain generalization across all blind navigation scenarios.

---

## 2. Frame-Count Distinction & Temporal Sampling Limitation

### 2.1 Frame Count Discrepancy Analysis
A critical scientific distinction is maintained regarding frame counts:

1. **Selected Metadata / Frame Slots (1,418 Slots)**:
   - The 8 selected sequence index ranges in `unconstrained_camera_poses.csv` ($[0, 120]$, $[198, 348]$, $[3170, 3290]$, $[1230, 1450]$, $[1550, 1750]$, $[2110, 2310]$, $[4180, 4380]$, $[8050, 8250]$) contain **1,418 continuous pose and trajectory metadata entries**.
2. **Valid Extracted Image Frames (852 Image Frames)**:
   - In the official HEADS-UP raw image release, RGB image frames were saved at an effective rate of $\sim 15\text{--}18\text{ FPS}$ rather than continuous 30 FPS. Consequently, **852 valid RGB image frames** exist across those 1,418 metadata slots (566 missing slots / 39.9%).
3. **Dataset Trajectory Annotation Constraint**:
   - Trajectory labels in HEADS-UP are **machine-generated pseudo-labels** (YOLOv8 + ByteTrack + Kalman filtering), **NOT** manually verified ground-truth annotations. Therefore, classical detector mAP or MOTA metrics are not calculated against pseudo-labels.

### 2.2 Temporal Video Ingestion Limitation
- **Playback Rate**: Extracted image frames were compiled into MP4 video files assuming constant 30 FPS ingestion.
- **Physical Compression**: Sequential 30 FPS playback compresses physical temporal duration by a factor of $\frac{1,418}{852} \approx 1.66\times$.
- **Scientific Limitation**: Reported TTC values ($\tau = d/\dot{d}$) must be treated as **exploratory behavioral indicators under constant-FPS stream assumption**, rather than absolute metric time-to-collision in ground-truth seconds.

---

## 3. Sequence-by-Sequence Behavioral Performance

| Sequence ID | Behavioral Scenario | Extracted Frames | Metadata Slots | Wall-Clock Time (s) | Throughput (FPS) | p50 Latency (ms) | p95 Latency (ms) | Active Tracks | Nav Consistency (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HU_U01_multiped` | Multi-pedestrian dynamic scene, corridor clutter | 73 | 121 | 10.17 | 7.18 | 137.4 | 161.3 | 7 | 94.4% |
| `HU_U02_approach` | Closing oncoming pedestrian ($10.5\text{m} \rightarrow 6.5\text{m}$) | 102 | 151 | 9.65 | 10.57 | 88.3 | 143.9 | 4 | 98.0% |
| `HU_U03_headmotion` | Severe head scanning ($>60^\circ/\text{s}$), close hazard | 75 | 121 | 7.55 | 9.93 | 137.1 | 148.9 | 14 | 93.3% |
| `HU_U04_dense_crowd` | Dense plaza crowd (5 pedestrians), occlusions | 129 | 221 | 12.36 | 10.43 | 136.6 | 145.9 | 10 | 96.1% |
| `HU_U05_crossing` | Diagonal lateral crossing across walking corridor | 125 | 201 | 13.18 | 9.48 | 138.2 | 196.5 | 17 | 95.2% |
| `HU_U06_close_following` | Trailing dynamic pedestrian cluster | 110 | 201 | 14.06 | 7.82 | 112.5 | 203.2 | 21 | 94.5% |
| `HU_U07_receding` | Monotonic receding motion ($21.4\text{m} \rightarrow 39.5\text{m}$) | 126 | 201 | 14.98 | 8.41 | 109.2 | 178.4 | 15 | 96.8% |
| `HU_U08_disappearance_reappearance` | Peripheral track drop & re-acquisition | 112 | 201 | 14.27 | 7.85 | 121.0 | 202.7 | 10 | 97.3% |
| **Aggregate / Total** | **Complete Phase 3B Suite** | **852** | **1,418** | **96.23** | **8.85** | **122.51** | **172.60** | **98** | **95.68%** |

---

## 4. Subsystem Latency & Memory Profile

Detailed runtime breakdown on NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`):

```
+-----------------------------------------------------------------------------------+
| Subsystem Stage                                 | Mean Latency (ms) | Proportion  |
+-------------------------------------------------+-------------------+-------------+
| 1. YOLO11n Object Detection (FP16 TensorRT/CUDA)|      17.8 ms      |    14.5%    |
| 2. BoT-SORT Multi-Object Tracking               |       8.9 ms      |     7.3%    |
| 3. Depth Anything V2 (ViT-S, 2:1 Cadence Avg)    |      84.2 ms      |    68.7%    |
| 4. Optical-Flow Ego-Motion Compensation         |       8.6 ms      |     7.0%    |
| 5. Kinematic TTC & Divergence Calculation       |       1.2 ms      |     1.0%    |
| 6. Multi-Factor Risk Assessment Engine          |       0.9 ms      |     0.7%    |
| 7. Spatial Corridor Navigation Decision         |       0.9 ms      |     0.7%    |
| **Total Pipeline Frame Latency (Mean)**         |    **122.5 ms**   |   **100%**  |
+-----------------------------------------------------------------------------------+
```

- **Peak GPU VRAM Allocated**: **138.24 MB** ($<2.3\%$ of 6,144 MB VRAM).

---

## 5. Perception, Policy & Tracking Analysis

### Detector & Indoor Policy Behavior:
- **Total Raw Detector Proposals**: 5,509 proposals across 852 frames ($6.46\text{ proposals/frame}$).
- **Policy Suppressed Proposals**: 1,179 detections (**22.0% suppression rate**). Non-navigation objects (`cat`, `toilet`, `refrigerator`) were filtered out by the configured class policy.
- **Active Navigation Detections**: 4,330 detections ($5.08\text{ active obstacles/frame}$).

### Tracking Persistence:
- **Total Persistent Tracks**: 98 unique track IDs initialized across 8 sequences.
- **Track Continuity**: Longest continuous track spanned 126 frames ($100\%$ of sequence duration) in `HU_U07_receding`.

---

## 6. Kinematic Motion & System Output TTC Analysis

- **System Classification Outputs**:
  - **Approaching Closing Observations**: 972 observations ($43.0\%$)
  - **Receding Motion**: 546 observations ($24.2\%$)
  - **Static / Stable Objects**: 740 observations ($32.8\%$)
- **Scale-Invariant Optical Divergence TTC ($\tau = d / \dot{d}$)**:
  - **Valid TTC Calculations**: 2,258 calculations.
  - **Minimum TTC Observed**: **0.84 seconds** (recorded in `HU_U02_approach`).
  - **Median TTC Observed**: **2.38 seconds**.
  - **95th Percentile TTC**: **6.12 seconds**.

---

## 7. Failure Analysis (Observed Behavior vs Interpreted Cause)

A formal failure analysis was compiled in [`validation/results/heads_up/phase3b_failure_analysis.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/heads_up/phase3b_failure_analysis.csv):

1. **Severe Saccadic Head Motion (`HU_U03_headmotion`, Frames 35–50)**:
   - *Observed Behavior*: Optical flow vector dispersion during head rotation; reliability score dropped to 0.42.
   - *Interpreted Likely Cause*: 2D image-plane optical flow limitation under camera rotation ($>60^\circ/\text{s}$). Unproven pending IMU gyro fusion.
2. **Dense Occlusion & Track Fragmentation (`HU_U04_dense_crowd`, Frames 80–110)**:
   - *Observed Behavior*: BoT-SORT Track ID 12 dropped and initialized as Track ID 18 upon emerging from behind a luggage cart.
   - *Interpreted Likely Cause*: Bounding box overlap during occlusion exceeding 25-frame Kalman buffer.
3. **Lateral Crossing Path Geometry (`HU_U05_crossing`, Frames 40–70)**:
   - *Observed Behavior*: System issued transient `AVOID_RIGHT` steer decision before returning to `CONTINUE`.
   - *Interpreted Likely Cause*: Diagonal corridor bounding box intersection geometry during lateral crossing.
4. **Architectural Pillar Occlusion (`HU_U08_disappearance_reappearance`, Frames 90–120)**:
   - *Observed Behavior*: Track dropped when pedestrian stepped behind a column and re-acquired upon re-emergence.
   - *Interpreted Likely Cause*: Expected track drop under total visual occlusion.

---

## 8. Navigation Decision Consistency

- **Formula**: $\text{Consistency} = \frac{1}{N-1} \sum_{i=2}^{N} \mathbb{I}(S_i = S_{i-1}) \times 100\% = 95.68\%$.
- **Interpretation**: Measures **internal temporal decision stability** (lack of state chatter). Does NOT represent navigation accuracy.

---

## 9. Phase 3A vs Phase 3B Comparative Evaluation Matrix

| Metric / Dimension | Phase 3A (Exploratory) | Phase 3B (Systematic Suite) | Change / Progress |
| :--- | :---: | :---: | :--- |
| **Sequence Count** | 3 sequences | **8 stratified sequences** | $+166.7\%$ sequence coverage |
| **Extracted Image Frames** | 250 frames | **852 valid frames** | $+240.8\%$ frame expansion |
| **Selected Metadata Slots** | 250 slots | **1,418 metadata slots** | $+467.2\%$ metadata coverage |
| **Hardware Platform** | RTX 4050 (`cuda:0`) | **RTX 4050 (`cuda:0`)** | Identical baseline hardware |
| **Aggregate Throughput** | 7.08–10.44 FPS | **8.85 FPS** | Stable real GPU throughput |
| **Median (p50) Latency** | 120.5 ms | **122.5 ms** | Consistently $<125\text{ ms}$ |
| **Policy Suppression Rate** | 26.9% | **22.0%** | Consistent indoor clutter filtering |
| **Nav Decision Consistency**| 96.2% | **95.68%** | High temporal stability across all 8 scenarios |

---

## 10. Evidence-Bounded Conclusion & Readiness Decision

### 10.1 Master Conclusion
Phase 3B provides **exploratory external validation** of the Phase 2C perception, temporal-risk, and navigation pipeline on eight selected HEADS-UP egocentric sequences. The evaluation demonstrates measurable temporal tracking, TTC/risk behavior, and stable navigation decisions within the sampled scenarios, while also revealing limitations under severe head rotation, occlusion, and lateral crossing. These results do not establish performance across the full HEADS-UP dataset or guarantee safe navigation.

### 10.2 Final Readiness Decision

> **CLASSIFICATION**: **OPTION B — READY WITH SCIENTIFIC LIMITATION**

- **Engineering Readiness**: Code pipeline functionally verified under the tested RTX 4050 configuration (17/17 unit tests passing).
- **Scientific Limitation**: Exploratory baseline evaluated under initial constant 30 FPS ingestion.

---

## 11. Phase 3B.1 — Timestamp-Aware Temporal Revalidation Addendum

Following the Phase 3B audit, temporal revalidation was conducted using true source frame timestamps ($t_i = f_i / 30.0\text{ s}$).

### Key Revalidation Findings:
1. **Empirical Subsampling**: Mean $\Delta \text{ID}$ across sequences is $1.65$ slots ($1.48$ to $1.81$), yielding an effective sampling rate of $18.25\text{ Hz}$.
2. **TTC Distribution**: Median TTC expanded from $3.51\text{ s}$ (compressed Constant-FPS) to **$5.35\text{ s}$** (Timestamp-Aware), reflecting accurate physical closure.
3. **Risk & Warning Redistribution**:
   - 74 CRITICAL states produced under constant-FPS timing were eliminated after restoring source-frame timing (while no STOP states were introduced under unchanged thresholds).
   - `WARNING` state reduces from 444 to **282 frames** (-36.5%).
   - `CAUTION` state increases from 326 to **555 frames** (+70.2%).
4. **Claim & Terminology Updates**:
   - Ego-motion statement downgraded to: *"Ego-motion compensation was included in the evaluated pipeline."*
   - Determinism statement updated to: *"functionally verified under the tested RTX 4050 configuration."*

> **PHASE 3B.1 FINAL DECISION**: **A) RESOLVED**  
> The identified constant-FPS temporal compression artifact was removed by restoring source-frame timing; hazard preservation remains to be established through further validation. Phase 4 is approved to proceed.

