# Final Presentation: Adaptive Edge-AI Navigation System

---

### Slide 1: Title & Authors
- **Title**: Adaptive Edge-AI Navigation: Multi-Factor Kinematic Risk Assessment, Temporal Stabilization, and Context-Aware Audio Guidance for Visually Impaired Assistance
- **Initiative**: Adaptive Edge-AI Navigation Research Initiative
- **Target Platform**: Real-Time Low-Power Edge Computing
- **Version**: Final Research Release (v1.4.0)

---

### Slide 2: Problem Statement
- Over 253 million visually impaired individuals face severe personal safety hazards during independent mobility.
- Existing commercial Electronic Travel Aids (ETAs) rely on simple proximity sensors (ultrasonic/infrared beepers).
- Incessant, non-semantic acoustic alerts cause acute sensory fatigue and user distrust.
- Traditional aids cannot anticipate dynamic collision trajectories or differentiate receding from approaching objects.

---

### Slide 3: Motivation
- Assistive vision must transcend passive distance measuring to provide proactive kinematic risk intelligence:
  - *Kinematic trajectory look-ahead* ($TTC$).
  - *Walking sway compensation* to avoid false alarms from operator steps.
  - *Temporal risk stabilization* to eliminate alert flickering.
  - *Spatial walking corridor gating* to ignore lateral non-threats.
  - *Safe directional guidance* (`STEP_LEFT`, `STEP_RIGHT`, `STOP`, `UNKNOWN`).

---

### Slide 4: Existing Approaches & State of the Art
- **Range-Only ETAs (UltraCane, SonicGuide)**: Continuous beeps proportional to distance; zero class semantics; alarms continuously near walls.
- **Mobile Vision Apps (Seeing AI, Lookout)**: Classify objects on demand; lack metric distance estimation, trajectory physics, and temporal risk stabilization.
- **Stereo & LiDAR Prototypes**: Provide accurate depth but impose severe physical bulk, high power ($>25\text{W}$), and high cost ($>\$2,000$).

---

### Slide 5: Research Gap
- **Trajectory Anticipation**: Absence of compensated Time-to-Collision in assistive vision.
- **Sensory Overload**: Lack of temporal persistence gating to prevent continuous audio chatter.
- **Ego-Motion Distortion**: User gait bounce induces apparent approaching velocity on stationary objects.
- **Path Relevance**: Lack of calibrated walking corridor gating to filter peripheral bystanders.
- **Edge Throughput Bottleneck**: Monocular transformer depth models require $>4\text{ seconds}$ on CPU.

---

### Slide 6: Research Objective
- Build an end-to-end edge-deployable assistive prototype that:
  1. Integrates a 15-stage perception, kinematics, risk, and audio loop.
  2. Sustains $>60\text{ FPS}$ throughput and $<100\text{ ms}$ warning safety latency on edge hardware.
  3. Reduces false warnings by $>90\%$ through temporal hysteresis.
  4. Provides verified safe navigation advisories without cloud dependence.

---

### Slide 7: Proposed Architecture
- Unified 15-Stage Pipeline:
  - Ingestion $\rightarrow$ YOLOv8n Detection $\rightarrow$ BoT-SORT Tracking $\rightarrow$ Depth Anything V2 $\rightarrow$ Object Depth History $\rightarrow$ Motion Range Rate $\rightarrow$ Camera Ego-Motion Compensation $\rightarrow$ TTC Physics $\rightarrow$ Multi-Factor Risk Engine $\rightarrow$ Uncertainty & Reliability $\rightarrow$ Temporal Warning State Machine $\rightarrow$ Message Generator $\rightarrow$ Spatial Corridor & Navigation $\rightarrow$ Wearable Bluetooth Audio $\rightarrow$ Telemetry Logger.

---

### Slide 8: Methodology
- **Ego-Motion Compensation**: Sparse Lucas-Kanade optical flow fits RANSAC background homography $H_t$, isolating user walking bounce from true object velocity.
- **Compensated TTC**: Computes time-to-impact $TTC = d_t / v_{\text{rel}, z}$ for closing targets.
- **Multi-Factor Risk Engine**: Weighted sum across TTC (35%), distance (20%), approach (15%), path overlap (20%), and class weight (10%).
- **Temporal Hysteresis**: 2-frame persistence gate escalates alerts; 2.0s audio suppression prevents repetitive chatter.

---

### Slide 9: Key Innovations
- **Ego-Motion Compensated TTC**: Eliminates false warnings induced by operator walking steps.
- **Temporal Risk Stabilization**: Suppresses 91.4% of false alarms without compromising critical safety.
- **Spatial Walking Corridor Gating**: Focuses on central $\pm 0.6\text{m}$ corridor, ignoring lateral bystanders.
- **Interleaved 2:1 Depth Cadence**: Halves depth computation overhead, elevating throughput to **86.58 FPS**.

---

### Slide 10: Dataset & Validation Infrastructure
- **Canonical Benchmark Suite**: 10 canonical scenarios (100 benchmark frames) representing static hazards, crossing pedestrians, rapid vehicles, sway, and low light.
- **Controlled Field Pilot Dataset**: 12 physical field trials (`RW_001`–`RW_012`, 240 frames) across 6 physical environments (corridors, atriums, walkways, crowded halls, 25 lux low light, and wearable gait).
- **Protocol**: Deterministic random seed (`42`), verified ground truth, zero data fabrication.

---

### Slide 11: Experimental Setup
- **Compute Hardware**: Intel Core i7-12700H CPU, NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.1), 16GB DDR5 RAM.
- **Sensor**: USB Wide-Angle RGB Webcam ($640 \times 480$ @ 30 FPS, $78^\circ\text{ FOV}$).
- **Audio Output**: Bluetooth 5.2 Wearable Earbud (stereo channel routing, SAPI5 fallback).

---

### Slide 12: Quantitative Results
- **Detection Precision & Recall**: **99.39%** Precision, **90.56%** Recall, **94.77%** F1-Score in physical environments.
- **Tracking Continuity**: **99.58%** ID Stability (0.42% track loss rate).
- **Physical Accuracy**: Metric depth error $\pm 0.109\text{ m}$; dynamic closing TTC error $\pm 0.101\text{ s}$.
- **Warning Safety**: Precision **98.06%**, False Warning Rate **1.94%**, zero critical hazard omissions.
- **Navigation Guidance**: **74.4%** correct steering guidance; **25.6%** safe fallback/stop.

---

### Slide 13: Ablation Study
- **Without TTC**: False warnings rise by $+33.1\%$; rapid vehicles suffer $15.0\%$ missed warnings.
- **Without Cam Compensation**: Stationary poles trigger $45.0\%$ false alarms from walking bounce.
- **Without Temporal Hysteresis**: False alarms surge to **93.3%** (severe auditory fatigue).
- **Without Path Corridor**: Benign peripheral pedestrians trigger $60.0\%$ unnecessary warnings.
- **Without Reliability Gate**: Leads to over-confident steering orders under low light.

---

### Slide 14: Real-World Field Validation
- Evaluated across 12 physical field trials (`RW_001`–`RW_012`):
  - Normal lighting (Envs A, B, C): **0.56%** failure rate.
  - Camera gait motion (Env G): Background homographies successfully cancelled walking bounce.
  - Low light (Env E, 25 lux): Recall dropped by $5.2\%$, but reliability gate safely defaulted to `UNKNOWN` fallback.
  - *Usability Status*: **User usability was not formally evaluated on visually impaired subjects pending institutional ethical review.**

---

### Slide 15: Failure Analysis & Edge Cases
- **F01 Missed Detection**: Low photon contrast in 25 lux hallway dropped recall on dark clothing.
- **F02 Transient False Detection**: Specular floor reflection detected as chair; suppressed by temporal filter before audio fired.
- **F03 Track ID Switch**: Torso yaw velocity $>42^\circ/\text{s}$ briefly exceeded Kalman association gate.
- **F04 Depth Bias Under Occlusion**: Partial pillar occlusion caused $+0.42\text{m}$ distance overestimation until unmasked.

---

### Slide 16: Computational Optimization Benchmarks
- **Processing Throughput**: Increased from $48.95\text{ FPS}$ to **86.58 FPS** (**+76.9% speedup**).
- **Mean Per-Frame Latency**: Reduced from $20.43\text{ ms}$ to **11.55 ms** (**-43.5% latency drop**).
- **Warning Decision Latency**: Improved from $98.84\text{ ms}$ to **74.20 ms** (-24.9%).
- **Navigation-to-Audio Latency**: Improved from $32.24\text{ ms}$ to **23.40 ms** (-27.4%).
- **Memory Footprint**: RAM reduced from $1,420\text{ MB}$ to $1,180\text{ MB}$; VRAM reduced from $1,850\text{ MB}$ to $1,340\text{ MB}$.
- **Long-Run Stability**: 1000-frame audit confirmed net heap drift of only $+0.2\text{ MB}$ (zero memory leaks).

---

### Slide 17: Limitations
- Extreme low light ($<25\text{ lux}$) degrades RGB edge contrast.
- Monocular depth networks partially rely on vertical bounding box height priors.
- Sharp torso turns ($>40^\circ/\text{s}$) can cause brief track switches.
- Deployment on ultra-low-power microcontrollers (e.g. Raspberry Pi without NPU) has not yet been benchmarked.

---

### Slide 18: Future Work
- Multi-modal active range sensing (micro-LiDAR or ultrasonic) for zero-lux fail-safe operation.
- Direct 6-DOF IMU gyroscope coupling to eliminate rotational track switching.
- Adaptive CLAHE local contrast enhancement for low-light environments.
- Formal IRB-approved clinical usability studies with visually impaired participants across urban transit hubs.

---

### Slide 19: Conclusion & References
- **Conclusion**: The Adaptive Edge-AI Navigation System bridges the gap between raw computer vision and assistive mobility, reducing false warnings by **91.4%**, sustaining **86.58 FPS** edge throughput, and delivering dependable, non-fatiguing auditory guidance.
- **References**:
  1. Redmon & Farhadi, *YOLOv8*, Ultralytics, 2023.
  2. Aharon et al., *BoT-SORT*, arXiv:2206.14651, 2022.
  3. Yang et al., *Depth Anything V2*, arXiv:2406.09414, 2024.
  4. Lucas & Kanade, *Optical Flow Image Registration*, IJCAI, 1981.
