# Phase 3B — Scientific Audit & Temporal Integrity Report

> **AUTHORITATIVE SCIENTIFIC AUDIT REPORT**  
> **Project Title**: An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users  
> **Repository**: `Adaptive-edge-ai-navigation`  
> **Target Dataset**: Official HEADS-UP Egocentric Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
> **Audit Date**: October 5, 2026  
> **Audit Focus**: Temporal Sampling Integrity, Frame Subsampling Analysis, Claim Verification, and Evidence-Bounded Language Standardization

---

## 1. Frame & Temporal Integrity Audit

### 1.1 Overview of Frame Slot Discrepancy
The Phase 3B sequence extraction selected 8 behaviorally stratified ranges spanning **1,418 metadata frame slots** in `unconstrained_camera_poses.csv`. However, the extraction pipeline yielded **852 valid extracted RGB image files**.

- **Cause of Discrepancy**: The official HEADS-UP benchmark release logs camera poses and trajectory CSV metadata at **30.0 Hz continuous frame intervals** (1,418 slots across the 8 selected ranges). However, the raw RGB image stream in `rgb_unconstrained.tar.gz` was recorded with **variable frame subsampling** (effective image frame rate of $\sim 15\text{--}18\text{ FPS}$).
- **Impact**: Exactly **566 metadata frame slots (39.9%)** do not possess a corresponding raw RGB PNG image file in the benchmark release.

### 1.2 Sequence-by-Sequence Temporal Audit Breakdown

| Sequence ID | Selected Range $[S, E]$ | Expected Metadata Slots ($E-S+1$) | Valid Extracted Images | Missing Image Slots | Contiguous Image IDs? | Frame ID Delta Distribution ($\Delta = f_{i} - f_{i-1}$) | Missing Slot Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `HU_U01_multiped` | $[0, 120]$ | 121 | **73** | 48 | **NO** | $\Delta=1: 43$, $\Delta=2: 17$, $\Delta=3: 9$, $\Delta=4: 2$, $\Delta=6: 1$ | 39.7% |
| `HU_U02_approach` | $[198, 348]$ | 151 | **102** | 49 | **NO** | $\Delta=1: 69$, $\Delta=2: 21$, $\Delta=3: 8$, $\Delta=4: 1$, $\Delta=5: 2$ | 32.5% |
| `HU_U03_headmotion` | $[3170, 3290]$ | 121 | **75** | 46 | **NO** | $\Delta=1: 48$, $\Delta=2: 14$, $\Delta=3: 7$, $\Delta=4: 4$, $\Delta=6: 1$ | 38.0% |
| `HU_U04_dense_crowd` | $[1230, 1450]$ | 221 | **129** | 92 | **NO** | $\Delta=1: 82$, $\Delta=2: 22$, $\Delta=3: 13$, $\Delta=4: 6$, $\Delta=5: 3$, $\Delta=6: 1$, $\Delta=8: 1$ | 41.6% |
| `HU_U05_crossing` | $[1550, 1750]$ | 201 | **125** | 76 | **NO** | $\Delta=1: 73$, $\Delta=2: 34$, $\Delta=3: 11$, $\Delta=4: 4$, $\Delta=5: 2$ | 37.8% |
| `HU_U06_close_following` | $[2110, 2310]$ | 201 | **110** | 91 | **NO** | $\Delta=1: 57$, $\Delta=2: 31$, $\Delta=3: 12$, $\Delta=4: 5$, $\Delta=5: 2$, $\Delta=6: 2$ | 45.3% |
| `HU_U07_receding` | $[4180, 4380]$ | 201 | **126** | 75 | **NO** | $\Delta=1: 77$, $\Delta=2: 32$, $\Delta=3: 12$, $\Delta=4: 2$, $\Delta=5: 1$, $\Delta=6: 1$ | 37.3% |
| `HU_U08_disappearance_reappearance` | $[8050, 8250]$ | 201 | **112** | 89 | **NO** | $\Delta=1: 61$, $\Delta=2: 32$, $\Delta=3: 8$, $\Delta=4: 6$, $\Delta=5: 2$, $\Delta=6: 1$, $\Delta=8: 1$ | 44.3% |
| **Total Suite** | — | **1,418** | **852** | **566** | **NO** | Non-contiguous frame steps across all 8 sequences | **39.9%** |

### 1.3 Identification of Video Runner Timing Assumption

- **Video Ingestion Mechanism**: In [`extract_heads_up_phase3b.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/scripts/tools/extract_heads_up_phase3b.py), the 852 valid extracted images were sequentially compiled into MP4 video files using `cv2.VideoWriter(..., 30.0, ...)`. In [`run_phase3b_heads_up_validation.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/scripts/run/run_phase3b_heads_up_validation.py), OpenCV `VideoCapture` ingests the MP4 sequentially frame $i \rightarrow i+1$.
- **Timing Model**: The execution pipeline treats every extracted image frame as a **uniform $1/30\text{-second}$ ($\Delta t = 33.3\text{ ms}$) timestep** (Option B: assumed constant 30 FPS spacing).
- **Physical Scale Compression**: Because source frame IDs have non-unit gaps ($\Delta \text{frame} \in \{1, 2, 3, 4, 5, 6, 8\}$), sequential 30 FPS playback compresses physical temporal duration by a factor of $\frac{1,418}{852} \approx 1.66\times$.

> [!WARNING]
> **MANDATORY SCIENTIFIC LIMITATION DECLARATION**:  
> Because image frames are non-contiguous while the video runner assumes uniform 30 FPS spacing, computed kinematic velocities $\dot{d}$ and Time-to-Collision metrics $\tau = d / \dot{d}$ reflect an accelerated temporal scale. These TTC values MUST be interpreted strictly as **EXPLORATORY BEHAVIORAL INDICATORS UNDER CONSTANT-FPS INGESTION**, rather than absolute metric time-to-collision in ground-truth wall-clock seconds.

---

## 2. Scientific Claim Audit & Language Standardization

To ensure rigorous scientific integrity, all claims in Phase 3B reports have been audited to eliminate overstatements:

| Unsubstantiated Claim / Overstatement | Audit Disposition | Approved Evidence-Bounded Replacement |
| :--- | :---: | :--- |
| *"100% sensitivity to genuine obstacles"* | **REMOVED** | *"Preserved detection proposals for dynamic pedestrians and spatial obstacles within the sampled frames"* |
| *"Correctly classifying approaching/receding targets"* | **REMOVED** | *"Observed system classification outputs of 972 approaching, 546 receding, and 740 static target observations"* |
| *"Successfully prevented false TTC alarms"* | **REMOVED** | *"Ego-motion compensation suppressed gait-induced radial divergence spikes relative to uncompensated optical flow"* |
| *"FULLY VALIDATED"* | **REMOVED** | *"Exploratory external behavioral validation on selected HEADS-UP sequences"* |
| Any claim of full HEADS-UP dataset validation | **REMOVED** | *"Sample-level evaluation across eight selected egocentric sequences (852 valid images / 1,418 metadata slots)"* |
| Any implication of clinical or blind navigation safety | **REMOVED** | *"Exploratory edge-AI framework evaluation strictly limited to offline pre-recorded video sequences"* |

---

## 3. System Outputs vs Ground Truth Audit

### 3.1 Target Kinematic State Counts
The recorded motion state counts:
- **Approaching Observations**: 972
- **Receding Observations**: 546
- **Static / Stable Observations**: 740

These numbers represent **measured system classification outputs** generated by the kinematic estimator in [`adaptive_navigation/temporal/motion.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/temporal/motion.py). Because HEADS-UP trajectory annotations are machine pseudo-labels rather than human-verified 3D motion ground truth, these counts are reported strictly as observed system behavior, **NOT** as verified ground-truth accuracy.

### 3.2 Time-to-Collision ($\tau$) Statistics
- **Valid TTC Calculations**: 2,258 calculations
- **Minimum TTC**: 0.84 seconds
- **Median TTC**: 2.38 seconds
- **95th Percentile TTC**: 6.12 seconds

These values represent **measured system output metrics** produced by scale-invariant optical divergence $\tau = d/\dot{d}$ under the constant 30 FPS playback assumption.

---

## 4. Policy Suppression Behavior Audit

- **Raw Detector Proposals**: 5,509 proposals ($6.46\text{ proposals/frame}$)
- **Active Navigation Detections**: 4,330 detections ($5.08\text{ detections/frame}$)
- **Policy Suppressed Detections**: 1,179 detections (**22.0% suppression rate**)

This suppression rate characterizes **detector-policy filtering behavior** configured in [`configs/final.yaml`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/configs/final.yaml) and [`adaptive_navigation/perception/detector.py`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/adaptive_navigation/perception/detector.py). It demonstrates that the configured class filter suppresses background furniture/architectural clutter (`cat`, `toilet`, `refrigerator`), but it is **NOT** described as detector precision/recall or mAP.

---

## 5. Navigation Decision Consistency Metric Audit

The system recorded a **95.68% consecutive-frame navigation decision consistency**.

### Mathematical Definition:
$$\text{Consistency} = \frac{1}{N-1} \sum_{i=2}^{N} \mathbb{I}(S_i = S_{i-1}) \times 100\% = 95.68\%$$
where $S_i \in \{\text{CONTINUE}, \text{CAUTION}, \text{AVOID\_LEFT}, \text{AVOID\_RIGHT}, \text{STOP}\}$ represents the discrete navigation state output at frame $i$, and $\mathbb{I}(\cdot)$ is the indicator function.

- **Interpretation**: This metric quantifies **internal temporal decision stability** and confirms that the directional hysteresis state machine prevents high-frequency command oscillation. It does **NOT** represent navigation accuracy or path optimality.

---

## 6. Failure Analysis Differentiation Audit

The 4 documented stress cases in [`validation/results/heads_up/phase3b_failure_analysis.csv`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/heads_up/phase3b_failure_analysis.csv) have been audited to strictly differentiate **Observed System Behavior** from **Interpreted Likely Cause**:

1. **`HU_U03_headmotion` (Frames 35–50)**:
   - *Observed System Behavior*: Optical flow vector dispersion during head movement; reliability score dropped to 0.42.
   - *Interpreted Likely Cause*: 2D image-plane optical flow limitation under fast 3D camera rotation ($>60^\circ/\text{s}$). Unproven pending IMU gyro fusion.
2. **`HU_U04_dense_crowd` (Frames 80–110)**:
   - *Observed System Behavior*: BoT-SORT Track ID 12 dropped and initialized as Track ID 18 upon emerging from behind a luggage cart.
   - *Interpreted Likely Cause*: Bounding box overlap during occlusion exceeding the 25-frame Kalman buffer.
3. **`HU_U05_crossing` (Frames 40–70)**:
   - *Observed System Behavior*: System issued transient `AVOID_RIGHT` steer decision before returning to `CONTINUE`.
   - *Interpreted Likely Cause*: Diagonal corridor bounding box intersection geometry during lateral pedestrian crossing.
4. **`HU_U08_disappearance_reappearance` (Frames 90–120)**:
   - *Observed System Behavior*: Track dropped when pedestrian stepped behind an architectural column and re-acquired upon re-emergence.
   - *Interpreted Likely Cause*: Expected track drop under total line-of-sight visual occlusion.

---

## 7. Evidence-Bounded Conclusion & Readiness Decision

### 7.1 Revised Master Conclusion
Phase 3B provides **exploratory external validation** of the Phase 2C perception, temporal-risk, and navigation pipeline on eight selected HEADS-UP egocentric sequences (852 valid extracted image frames across 1,418 metadata slots). The evaluation demonstrates measurable temporal tracking, TTC/risk behavior, and stable navigation decisions within the sampled scenarios, while also revealing limitations under severe head rotation, occlusion, and lateral crossing. These results do not establish performance across the full HEADS-UP dataset or guarantee safe real-world navigation.

### 7.2 Final Readiness Classification

> **CLASSIFICATION**: **OPTION B — READY WITH SCIENTIFIC LIMITATION**

- **Engineering Status**: The perception, tracking, TTC, risk, and navigation code pipeline is fully functional, deterministic, and verified (17/17 unit tests passing).
- **Scientific Limitation**: Because image frames are non-contiguous while the video runner assumes uniform 30 FPS ingestion, TTC and motion metrics reflect an accelerated temporal scale ($1.66\times$). All reported TTC metrics must be treated as **exploratory behavioral indicators under constant-FPS stream assumption**.

---

## 8. Required Actions Completed

1. Created [`validation/results/heads_up/phase3b_scientific_audit.md`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/heads_up/phase3b_scientific_audit.md).
2. Updated [`validation/results/heads_up/phase3b_heads_up_report.md`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/validation/results/heads_up/phase3b_heads_up_report.md) with evidence-bounded claims and temporal limitation declarations.
3. Updated [`PROJECT_INDEX.md`](file:///c:/My%20sep_stuffs/Research%20Conclave/Adaptive-edge-ai-navigation/PROJECT_INDEX.md) with evidence-bounded status and Option B classification.
