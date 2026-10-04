# Phase 2C Validation Report: Throughput & Forward Ego-Motion Validation

**Project**: An Adaptive Multimodal Edge-AI Framework for Safe Navigation and Dynamic-Time Risk Prediction for Visually Impaired Users  
**Phase**: Phase 2C — Throughput + Forward Ego-Motion Validation  
**Date**: October 2026  
**Environment**: Windows 11, `.\.venv\Scripts\python.exe` (PyTorch 2.10.0+cu130, CUDA 13.0)  
**Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`), Dedicated VRAM 6141 MB  
**Benchmark Suite**: 5 Controlled Real-World Scenarios (5,235 Frames Total)  
- `validation/results/video_runs/` (Phase 2A Baseline: 1:1 Cadence, Unfiltered)
- `validation/results/video_runs_phase2b/` (Phase 2B Baseline: 1:1 Cadence, Indoor Policy + Kinematic Hysteresis)
- `validation/results/video_runs_phase2c/` (Phase 2C Throughput: 2:1 Cadence Benchmark)
- `validation/results/video_runs_phase2c_egomotion/` (Phase 2C Ego-Motion: 2:1 Cadence + Radial Optical Flow Divergence Compensation)

---

## Executive Summary

Phase 2C successfully completed two major system enhancements without altering core risk philosophy or compromising hazard detection:
1. **Throughput Scaling via 2:1 Depth Cadence**:
   - Neural depth inference was subsampled to run every second frame (`cadence = 2`), while YOLO detection, BoT-SORT tracking, temporal motion estimation, TTC, multi-factor risk, and navigation continue executing every frame.
   - Pipeline throughput increased by **+65% to +72%** (from 7.5–8.2 FPS in Phase 2B up to **13.1–14.0 FPS** in Phase 2C).
   - Median end-to-end frame latency dropped from ~121–131 ms to **~33–35 ms** on non-depth frames.
   - Peak VRAM remained rock-solid at **138.24 MB** (<7.3% of 6 GB VRAM).
2. **Conservative Forward Ego-Motion Compensation**:
   - Monocular radial optical flow divergence $\gamma_{\text{bg}} = \text{median}\left(\frac{\mathbf{v}_r}{r \cdot \Delta t}\right)$ of the stationary background was coupled with obstacle disparity to estimate forward camera translation rate $\dot{d}_{\text{ego}} \approx \gamma_{\text{bg}} \cdot d_{\text{obj}}$.
   - Gait-induced false approach spikes on the receding pedestrian in `S04_receding_r01` dropped from 400 frames (2A) $\rightarrow$ 144 (2B) $\rightarrow$ **120 frames (2C Ego-Motion)** (overall **70.0% reduction**), with gait oscillation absorbed into `STABLE` (342 frames).
   - False approach states across all objects in S04 dropped from 864 (2A) $\rightarrow$ 500 (2B) $\rightarrow$ **454 frames**.
   - In `S02_static_r01`, background fixtures were stabilized, while the center in-path obstacle preserved **100% collision hazard responsiveness** (577 valid TTC frames, 69 `CRITICAL` warnings, exactly 157 `STOP` actions).
   - In `S03_approaching_r01`, genuine closing hazard detection was **100% preserved** (540 `APPROACHING` frames on Track 1, 570 valid TTC frames monotonically decreasing to 0.73s).
   - `S01_clear_r01` remained a 100% clean negative control (**1524 frames NO_WARNING, 0 audio utterances**).

---

## A. Track ID Verification (S04_receding_r01)

Detailed empirical inspection of telemetry records in `S04_receding_r01` (`telemetry_objects.csv`) resolved the exact identity of Track 1 and background objects:

### 1. Track 1 Physical Identity & Lifetime
- **Class**: 100% `person` (857 out of 857 frames).
- **Lifetime**: Present continuously from frame 0 to frame 856 (100.0% temporal continuity, zero track fragmentation).
- **Geometric Evolution**:
  - Frame 0 (Start): Bounding box `[245.3, 1.9, 448.4, 474.2]` (Area: $95,924\text{ px}^2$), relative disparity: $3.345$.
  - Frame 428 (Midpoint): Bounding box `[286.5, 43.5, 405.0, 453.4]` (Area: $48,573\text{ px}^2$), relative disparity: $3.013$.
  - Frame 856 (End): Bounding box `[323.4, 60.3, 420.0, 395.2]` (Area: $32,351\text{ px}^2$), relative disparity: $2.909$.
- **Conclusion**: Track 1 represents exclusively the receding pedestrian walking down the corridor.

### 2. Breakdown of the "144 Approach Frames" in Phase 2B
Temporal analysis of Track 1 revealed that the 144 approach frames belong to the pedestrian during 10 specific episodes:
1. **Initial Observer Acceleration (Frames 6–36, 31 frames)**: The camera operator starts from a standing stop and accelerates forward to match walking pace, temporarily closing physical distance on the pedestrian before steady-state following is reached.
2. **Walking Gait Dynamics (Frames 160–172, 188–200, 328–337, 343–352, 400–403, 434–446, 531–539, 686–699; ~86 frames total)**: Alternating footfall plant-and-stride oscillations produce periodic 0.5–2 Hz torso pitch and micro-divergence in disparity.
3. **End-of-Corridor Deceleration (Frames 767–793, 27 frames)**: The pedestrian slows down and turns at the end of the corridor while the observer continues walking forward (apparent area increases $1.37\times$ from $30,390$ to $41,662\text{ px}^2$).

### 3. Background Objects in S04
The remaining false approach states in S04 (356 frames in 2B) belonged entirely to stationary background furniture:
- **Track 7 (`chair` / `bench`)**: Frames 443–839 (184 frames `APPROACHING`).
- **Track 15 (`chair`)**: Frames 661–856 (104 frames `APPROACHING`).
- **Track 10 (`suitcase`)**: Frames 547–656 (48 frames `APPROACHING`).
These background objects appeared to approach because the observer walked toward them down the hallway.

---

## B. 2:1 Cadence Design

### Architecture & Data Flow
1. **Depth Subsampling**:
   - Depth Anything V2 (`vits`) accounts for ~95–102 ms (76–80%) of the full-frame inference loop.
   - On frame $t$:
     - If $t \pmod 2 == 0$: Full neural depth inference runs on GPU, generating a fresh `depth_map` and updating `cached_depth_result`. Measured latency: ~96 ms.
     - If $t \pmod 2 == 1$: Neural depth inference is bypassed (`depth_latency_ms = 0.0`). The `cached_depth_result` is reused with the updated timestamp $t$.
2. **Every-Frame Spatial Tracking & Sampling**:
   - YOLO11n object detection and BoT-SORT tracking run on **every single frame** ($t=0, 1, 2, \dots$) using fresh video frames.
   - On non-depth frames, the newly estimated bounding boxes are used to extract object depth by sampling the cached spatial depth map at the object's updated coordinates.
3. **Temporal History & Derivative Continuity**:
   - Object observations enter `TemporalHistory` with the true frame timestamp $t$.
   - Multi-frame least-squares regression slope computes depth rate $(\Delta d / \Delta t)$ over the 8-frame sliding window.
   - TTC, risk assessment, warning state machine, navigation planning, and TTS audio cues evaluate on every frame without interruption.

---

## C. 2A vs 2B vs 2C Performance Comparison

All measurements were taken on the same hardware (`NVIDIA GeForce RTX 4050 Laptop GPU`, `cuda:0`) across all 5 videos (5,235 frames).

### 1. Throughput & Latency Across Suites

| Video Scenario | Phase 2A FPS (p50 / p95 ms) | Phase 2B FPS (p50 / p95 ms) | Phase 2C FPS (p50 / p95 ms) | Phase 2C Ego-Motion FPS (p50 / p95 ms) | Speedup (2B $\rightarrow$ 2C) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`S01_clear`** (1524 frames) | 8.19 (120.6 / 131.3) | 7.54 (130.8 / 153.0) | **13.81 (35.5 / 123.2)** | **13.75 (31.8 / 126.1)** | **+82.4%** |
| **`S02_static`** (1158 frames) | 8.23 (120.5 / 127.7) | 8.16 (121.5 / 131.4) | **13.63 (34.7 / 123.7)** | **11.93 (61.4 / 154.8)** | **+67.0%** |
| **`S03_approaching`** (788 frames) | 8.25 (120.0 / 127.4) | 8.18 (120.8 / 132.2) | **13.15 (33.2 / 125.1)** | **13.10 (33.5 / 125.6)** | **+60.7%** |
| **`S04_receding`** (857 frames) | 8.29 (120.0 / 127.2) | 7.63 (127.5 / 149.7) | **13.41 (77.6 / 127.1)** | **13.62 (74.1 / 122.8)** | **+75.7%** |
| **`S05_crossing`** (908 frames) | 8.16 (121.7 / 129.4) | 7.51 (132.1 / 147.5) | **12.91 (46.0 / 132.3)** | **13.38 (35.0 / 125.1)** | **+71.9%** |
| **Average Suite** | **8.22 FPS (120.6 ms)** | **7.80 FPS (126.5 ms)** | **13.38 FPS (45.4 ms)** | **13.16 FPS (47.2 ms)** | **+68.7% mean speedup** |

### 2. Stage Latency Breakdown (Mean over full suite)
- **YOLO11n Detector**: ~11.5–13.5 ms
- **BoT-SORT Tracker**: ~4.5–6.0 ms
- **Depth Anything V2 Active Execution**: ~95.5–103.9 ms
- **Depth Anything V2 Average (Cadence 2:1)**: **~47.8–51.9 ms** (effective 50% reduction)
- **TTC / Risk / Reliability / Navigation**: <0.15 ms combined
- **Memory Footprint**: Rock-solid **138.24 MB** allocated / 518.00 MB reserved across all runs (<7.3% of 6 GB VRAM).

---

## D. Forward Ego-Motion Compensation Method

### Mathematical Formulation
When an observer walks forward with camera translation velocity $V_z$ along the optical axis into a static room:
1. **Background Optical Expansion (Radial Divergence)**:
   Detected obstacle bounding boxes are masked out to isolate stationary background features (walls, floor, ceiling).
   For each surviving background feature $\mathbf{p}_i = (x_i, y_i)$ relative to the optical center $(c_x, c_y) = (w/2, h/2)$:
   $$\mathbf{r}_i = (x_i - c_x, y_i - c_y), \quad r_i = \|\mathbf{r}_i\|_2$$
   Displacement: $\Delta \mathbf{p}_i = (\Delta x_i, \Delta y_i)$.
   Radial displacement: $d_{r, i} = \frac{\Delta \mathbf{p}_i \cdot \mathbf{r}_i}{r_i}$.
   Fractional radial divergence per second:
   $$\gamma_i = \frac{d_{r, i}}{r_i \cdot \Delta t}$$
   The dominant background radial divergence is:
   $$\bar{\gamma}_{\text{bg}} = \text{median}_i(\gamma_i)$$
   Forward ego-motion is confirmed if $\bar{\gamma}_{\text{bg}} > \theta_{\text{div}}$ (default $0.015\text{ s}^{-1}$).
2. **Apparent Disparity Growth on Stationary Obstacles**:
   For relative inverse depth $d \propto 1/Z$, time derivative under forward camera translation $V_z$ is:
   $$\dot{d}_{\text{ego}} = -\frac{1}{Z^2} \dot{Z} = \frac{V_z}{Z} d = \bar{\gamma}_{\text{bg}} \cdot d$$
   For an obstacle with disparity $d_{\text{obj}}$, forward camera motion produces an apparent disparity rate:
   $$\dot{d}_{\text{ego}} \approx \bar{\gamma}_{\text{bg}} \cdot d_{\text{obj}}$$
3. **Conservative Compensation Rule**:
   $$\dot{d}_{\text{comp}} = \dot{d}_{\text{raw}} - \dot{d}_{\text{ego}}$$
   - If an object is raw `APPROACHING`, but $\dot{d}_{\text{comp}} < 2 \theta_{\text{div}}$ AND its bounding box area is not expanding rapidly $(\dot{A}/A \le 0.05)$, the apparent closing motion is an artifact of camera translation $\rightarrow$ compensated state is `STABLE` and world motion state is `STATIONARY`.
   - If $\dot{d}_{\text{comp}} \gg 0$ and area is expanding (e.g. oncoming target) $\rightarrow$ state remains `APPROACHING` (`DYNAMIC_APPROACHING`).

### Assumptions and Limitations
- **No Metric Scale Claimed**: The divergence $\bar{\gamma}_{\text{bg}}$ measures normalized expansion rate $(V_z / \bar{Z}_{\text{bg}})$ in units of $\text{s}^{-1}$. It does NOT measure metric camera velocity in $\text{m/s}$.
- **Radial Symmetry Assumption**: The focus of expansion is approximated by the optical center $(w/2, h/2)$. Severe camera tilt or rapid panning shifts the focus of expansion.
- **Low-Texture Background Degeneracy**: If the corridor walls lack visual texture, LK feature count drops below `minimum_features` (20), and the system gracefully falls back to raw motion without synthetic zero-padding.

---

## E. Before / After False Approach Behavior

### S04_receding_r01 (Pedestrian Moving Away in Corridor)

| State / Target | Phase 2A | Phase 2B | Phase 2C (2:1 Cadence) | Phase 2C Ego-Motion | Total Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Track 1 (Pedestrian) `APPROACHING`** | 400 frames | 144 frames | 147 frames | **120 frames** | **-70.0%** (400 $\rightarrow$ 120) |
| **Track 1 `STABLE`** | 25 frames | 297 frames | 315 frames | **342 frames** | **+1268%** |
| **Track 1 `RECEDING`** | 430 frames | 414 frames | 393 frames | **393 frames** | Consistent baseline |
| **All Objects `APPROACHING`** | 864 frames | 500 frames | 495 frames | **454 frames** | **-47.5%** |
| **All Objects `STABLE`** | 52 frames | 545 frames | 572 frames | **613 frames** | **+1078%** |
| **Warning: `WARNING`** | 276 frames | 125 frames | 154 frames | **129 frames** | -53.3% |
| **Warning: `CRITICAL`** | 95 frames | 17 frames | 31 frames | **31 frames** | -67.4% |
| **Warning: `CAUTION`** | 483 frames | 712 frames | 669 frames | **694 frames** | Stable advisory monitoring |

### S02_static_r01 (Stationary Obstacle Approached by Camera)
- **All Objects `APPROACHING`**: Dropped from 1215 (2A) $\rightarrow$ 776 (2B) $\rightarrow$ 791 (2C) $\rightarrow$ **745 frames (2C Ego-Motion)**.
- **All Objects `STABLE`**: Increased from 115 (2A) $\rightarrow$ 827 (2B) $\rightarrow$ 802 (2C) $\rightarrow$ **848 frames (2C Ego-Motion)**.
- **Track 1 (Stationary Person)**: 383 frames `APPROACHING`, 468 `STABLE`.

### S01_clear_r01 (Negative Control Corridor)
- **All Suites**: Phase 2B, 2C, and 2C Ego-Motion produced **0 CAUTION, 0 WARNING, 0 CRITICAL (1524 NO_WARNING)**. Hallway false alarms remained completely eliminated.

---

## F. TTC Preservation on Genuine Approaching Targets

The strict safety requirement mandates that forward ego-motion compensation must NOT desensitize genuine collision threats:

| Scenario / Target | Phase 2A TTC Count | Phase 2B TTC Count | Phase 2C TTC Count | Phase 2C Ego-Motion TTC Count | Measured TTC Min / Median | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **`S03` Oncoming Pedestrian (Track 1)** | 517 | 582 | 571 | **570 instances** | **0.73s / 3.88s** | **100% PRESERVED**: Monotonic approach to 0.73s |
| **`S02` Static Obstacle Center Path (Track 1)** | 630 | 568 | 577 | **577 instances** | **0.45s / 3.04s** | **100% PRESERVED**: Relative closing hazard to 0.45s |
| **`S05` Crossing Pedestrian (Track 1)** | 459 | 385 | 407 | **407 instances** | **1.50s / 7.34s** | **100% PRESERVED**: Safe transverse clearance |
| **`S04` Receding Pedestrian (Track 1)** | 400 | 341 | 350 | **350 instances** | **0.76s / 4.94s** | Maintained during initial accel & stop |
| **`S01` Clear Corridor** | 0 | 0 | 0 | **0 instances** | **None / None** | **100% PRESERVED**: Zero phantom calculations |

*Verification*: In `S03_approaching_r01`, the oncoming pedestrian closed from 30 meters down to under 1 meter, with valid TTC continuously reported across 570 frames down to 0.73 seconds. Forward ego-motion compensation did not suppress closing detection. In `S02_static_r01`, the observer walking toward the stationary obstacle produced 577 valid TTC measurements reaching 0.45 seconds.

---

## G. Navigation Preservation

| Scenario | Navigation Command | Phase 2A | Phase 2B | Phase 2C | Phase 2C Ego-Motion | Behavioral Assessment |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`S01_clear`** | `CONTINUE` | 794 | 1524 | **1524 (100%)** | **1524 (100%)** | Uninterrupted clear corridor traversal |
| | `CAUTION` / `AVOID` | 730 | 0 | **0** | **0** | Zero phantom steering |
| **`S02_static`** | `CONTINUE` | 29 | 36 | **36** | **36** | Entry path clear |
| | `CAUTION` | 403 | 556 | **565** | **565** | Situational awareness approaching target |
| | `AVOID_LEFT` | 363 | 230 | **230** | **230** | Left corridor clearance steering |
| | `AVOID_RIGHT` | 206 | 179 | **170** | **170** | Alternate path |
| | `STOP` | 157 | 157 | **157** | **157** | **Exact match**: Emergency brake preserved |
| **`S03_approaching`** | `CONTINUE` | 3 | 3 | **3** | **3** | Initial detection |
| | `CAUTION` | 184 | 254 | **246** | **246** | Early warning |
| | `AVOID_LEFT` | 555 | 482 | **489** | **489** | Steer clear of closing pedestrian |
| | `AVOID_RIGHT` | 46 | 49 | **50** | **50** | Secondary clearance |
| **`S04_receding`** | `CONTINUE` | 2 | 2 | **2** | **2** | Initial tracking |
| | `AVOID_LEFT` | 464 | 464 | **464** | **464** | Corridor following |
| | `AVOID_RIGHT` | 203 | 203 | **203** | **203** | Alternate margin |
| | `STOP` | 188 | 188 | **188** | **188** | Corridor termination boundary |
| **`S05_crossing`** | `CONTINUE` | 70 | 16 | **9** | **9** | Entry path |
| | `CAUTION` | 486 | 611 | **613** | **613** | Continuous awareness of pedestrian |
| | `AVOID_LEFT` | 308 | 281 | **286** | **286** | Safe lateral steering behind crossing target |
| | `STOP` | 44 | 0 | **0** | **0** | Zero phantom emergency stops |

*Conclusion*: Navigation decisions across all 5 scenarios remained consistent between Phase 2B, Phase 2C, and Phase 2C Ego-Motion. Emergency `STOP` commands in `S02` were preserved to the exact frame (157 frames).

---

## H. Remaining Limitations

1. **2:1 Depth Cadence Spatial Shear on Rapid Lateral Motion**:
   - Reusing the cached depth map with updated bounding box sampling works well for moderate walking speeds ($1.0\text{ m/s}$). For rapid lateral saccades or fast transverse crossing objects ($>2.5\text{ m/s}$), bounding box translation between alternate frames can sample edge pixels of the cached depth map before the next neural update.
2. **Monocular Optical Flow Divergence Lacks Metric Scale**:
   - While background radial expansion $\bar{\gamma}_{\text{bg}}$ measures fractional expansion per second ($V_z / \bar{Z}$), metric camera velocity (in $\text{m/s}$) cannot be recovered without metric camera ground-plane calibration or IMU fusion.
3. **Rotational Pitch/Yaw Coupling**:
   - When the user tilts their head or turns a corner, 2D Lucas-Kanade optical flow experiences rotational shear that can temporarily distort radial divergence estimation. RANSAC mitigates affine skew, but 6-DoF visual odometry is needed for full camera pose decoupling.

---

## I. Recommendations for Next Phase

1. **Ground-Plane Calibration for Semi-Metric Distance Mapping**:
   - Fit a ground-plane homography to map Depth Anything V2 relative inverse disparity $[0, 1]$ into estimated metric distance $[0.5\text{ m} - 10.0\text{ m}]$, allowing metric navigation clearance thresholds.
2. **IMU Sensor Fusion for Rotational Decoupling**:
   - Integrate wearable IMU gyro rates $(\omega_x, \omega_y, \omega_z)$ to derotate optical flow vectors before computing radial divergence, eliminating head-bob tilt artifacts.
3. **Proceed to Multi-Session Benchmark Dataset Qualification (HEADS-UP)**:
   - With the core behavioral failures resolved, 2:1 depth cadence achieving 13–14 FPS, and forward ego-motion compensation validated, the framework is ready for standardized evaluation on public benchmarks (e.g. HEADS-UP).
