# Practical Ground-Truth Protocol for Monocular Edge-AI Navigation Validation

> **Document ID:** `docs/research/ground_truth_protocol.md`  
> **Status:** Active Protocol Standard  
> **Date:** October 5, 2026  
> **Target System:** Adaptive Monocular Edge-AI Navigation Framework (`v1.4.0-final`)

---

## 1. Executive Summary & Objective

To transition the monocular edge-AI navigation prototype from qualitative behavior verification to rigorous scientific evaluation, this protocol establishes a low-cost, repeatable **Ground-Truth (GT) measurement methodology**.

This protocol defines independent reference measurements for five critical system variables:
1. **Distance** ($d_{\text{GT}}$ in meters)
2. **Time-to-Collision** ($\text{TTC}_{\text{GT}}$ in seconds)
3. **Collision / No-Collision Outcome** ($\text{Outcome}_{\text{GT}}$ binary/ordinal)
4. **Navigation Advisory Decision** ($\text{Nav}_{\text{GT}} \in \{\text{CONTINUE}, \text{AVOID\_LEFT}, \text{AVOID\_RIGHT}, \text{STOP}\}$)
5. **Warning Advisory Timing** ($t_{\text{warning}}$ vs $t_{\text{GT\_hazard\_crossing}}$)

---

## 2. Low-Cost Experimental Setup & Equipment

Rather than requiring high-cost LiDAR arrays or optical motion-capture systems (e.g., Vicon), this protocol uses a controlled physical setup accessible in standard academic/research environments:

### 2.1 Hardware Requirements
- **Primary Sensing Rig**: Laptop webcam / USB camera mounted on a chest harness at nominal walking height ($1.35\text{ m}$).
- **Ground Alignment Grid**: High-visibility floor tape markers placed at $0.5\text{ m}$ intervals along a $5.0\text{ m} \times 2.0\text{ m}$ straight walking corridor.
- **Static Verification**: Digital handheld laser distance meter ($\pm 1.5\text{ mm}$ accuracy) for static target position verification.
- **Reference Recording Device**: Secondary smartphone or GoPro camera mounted on a side tripod ($60\text{ FPS}$) capturing a lateral view of the trajectory lane for frame-accurate trajectory verification.
- **Time Synchronization Tool**: Visual flash or acoustic sync clap at the start of each trial run.

---

## 3. Mathematical Ground-Truth Formulations

### 3.1 Ground-Truth Distance ($d_{\text{GT}}$)
For a target moving along a straight line towards the camera at marked position $x(t)$:
$$d_{\text{GT}}(t) = x_{\text{target}}(t) - x_{\text{camera}}$$

### 3.2 Ground-Truth Time-to-Collision ($\text{TTC}_{\text{GT}}$)
For an object moving with measured constant velocity $v_{\text{GT}} = \frac{\Delta x}{\Delta t}$:
$$\text{TTC}_{\text{GT}}(t) = \frac{d_{\text{GT}}(t) - d_{\text{safety\_margin}}}{v_{\text{GT}}}$$
where $d_{\text{safety\_margin}} = 0.5\text{ m}$ (nominal body thickness buffer).

### 3.3 Ground-Truth Hazard Crossing Timestamp ($t_{\text{GT\_hazard\_crossing}}$)
The exact frame timestamp when $d_{\text{GT}}(t) \le 1.5\text{ m}$ or $\text{TTC}_{\text{GT}}(t) \le 2.0\text{ s}$.

### 3.4 Ground-Truth Navigation Reference ($\text{Nav}_{\text{GT}}$)
Derived from physical spatial occupancy:
- **`CONTINUE`**: Lateral corridor ($\pm 0.6\text{ m}$) completely clear within $3.0\text{ m}$.
- **`AVOID_LEFT`**: Obstacle present in center/right corridor ($[0.0, +0.6]\text{ m}$), left corridor ($[-0.6, 0.0]\text{ m}$) clear.
- **`AVOID_RIGHT`**: Obstacle present in center/left corridor ($[-0.6, 0.0]\text{ m}$), right corridor ($[0.0, +0.6]\text{ m}$) clear.
- **`STOP`**: Both left and right corridors blocked, or obstacle $d_{\text{GT}} \le 1.0\text{ m}$ in central path.

---

## 4. Controlled Experimental Scenarios (S01–S06 Suite)

Each trial is executed 5 times to ensure statistical reliability (30 total initial evaluation runs):

| ID | Scenario Title | Initial Distance | Target Motion | Camera State | Repeat Count |
|:---|:---|:---:|:---|:---|:---:|
| **S01** | Clear Corridor | $>5.0\text{ m}$ | None (Clear path) | Straight walking ($1.0\text{ m/s}$) | 5 |
| **S02** | Static Obstacle | $3.5\text{ m}$ | Static mannequin / box | Straight walking ($1.0\text{ m/s}$) | 5 |
| **S03** | Person Approaching | $4.5\text{ m}$ | Approaching ($v \approx 1.2\text{ m/s}$) | Stationary camera | 5 |
| **S04** | Person Receding | $1.5\text{ m}$ | Walking away ($v \approx 1.0\text{ m/s}$) | Stationary camera | 5 |
| **S05** | Person Crossing | $3.0\text{ m}$ | Lateral cross ($v \approx 1.0\text{ m/s}$) | Stationary camera | 5 |
| **S06** | Camera Pitch/Sway | $3.0\text{ m}$ | Static obstacle | Head pan/tilt ($\pm 15^\circ$) | 5 |

---

## 5. Synchronization & Data Capture Protocol

1. **Pre-Trial Calibration**: Record 3 static laser distance checkpoints ($1.0\text{ m}$, $2.5\text{ m}$, $4.0\text{ m}$) to verify camera intrinsic/extrinsic alignment.
2. **Audio-Visual Sync**: Trigger visual LED flash in camera view at $t=0.0\text{ s}$.
3. **Telemetry Capture**: Demo Control Center records raw video frames and frame-by-frame JSON telemetry (`live_camera_metrics.json`).
4. **Frame Annotation**: Reference video is aligned in OpenCV frame-by-frame, recording ground-truth distance $d_{\text{GT}}(f)$ and state $\text{Nav}_{\text{GT}}(f)$ per frame $f$.

---

## 6. Scientific Metrics & Error Definitions

1. **Distance Error**:
   $$\text{MAE}_d = \frac{1}{N} \sum_{i=1}^N |d_{\text{est}, i} - d_{\text{GT}, i}|, \quad \text{RMSE}_d = \sqrt{\frac{1}{N} \sum_{i=1}^N (d_{\text{est}, i} - d_{\text{GT}, i})^2}$$

2. **Time-to-Collision Error**:
   $$\text{MAE}_{\text{TTC}} = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} |\text{TTC}_{\text{est}, i} - \text{TTC}_{\text{GT}, i}|$$

3. **Warning Lead Time ($\Delta t_{\text{lead}}$)**:
   $$\Delta t_{\text{lead}} = t_{\text{GT\_hazard\_crossing}} - t_{\text{system\_warning}}$$
   *(Positive value indicates warning was issued before hazard boundary was crossed; negative indicates late warning).*

4. **Navigation Decision Matrix**:
   - Accuracy (%) = $\frac{\text{Correct Nav Frames}}{\text{Total Nav Frames}} \times 100\%$
   - Unnecessary Stop Rate (%) = $\frac{\text{False STOP Frames}}{\text{Clear Corridor Frames}} \times 100\%$

---

## 7. Data Storage & Output Artifacts

All validation artifacts are stored under:
```
validation/results/ground_truth/
├── gt_logs_S01_to_S06.json
├── gt_distance_ttc_error_analysis.csv
└── ground_truth_validation_report.md
```
