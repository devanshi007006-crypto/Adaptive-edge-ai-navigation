# Dataset Card: Adaptive Edge-AI Navigation System

## 1. Dataset Summary
The validation data infrastructure consists of two distinct components designed to evaluate perception, risk estimation, and safety under strictly controlled conditions:
1. **Canonical Simulation Suite**: A 10-scenario canonical benchmark (100 sequential frames) representing canonical navigation challenges.
2. **Real-World Field Pilot Dataset**: A 12-scenario physical field evaluation suite (`RW_001` through `RW_012`, 240 sequential frames) collected across 6 verified campus and urban environments.

---

## 2. Canonical Simulation Suite
- **Dataset Name**: Adaptive Navigation Canonical Evaluation Suite
- **Source**: Synthetically modeled kinematic trajectories using photorealistic pedestrian and vehicle trajectories
- **Purpose**: Quantitative benchmarking, ablation studies, and ground-truth comparison under controlled repeatable kinematic conditions
- **Data Type**: Sequential video frames ($640 	imes 480 	imes 3$ RGB @ 30 FPS) with synchronized JSON metadata
- **Annotations Available**:
  - Precise 2D bounding boxes $[x_1, y_1, x_2, y_2]$
  - Categorical object classes (`person`, `car`, `chair`, `box`, `bollard`)
  - Ground-truth metric depth ($z	ext{ in meters}$)
  - Ground-truth closing velocity ($v_z	ext{ in m/s}$)
  - Exact Time-to-Collision ($TTC	ext{ in seconds}$)
  - 4-tier ground-truth risk level (`NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  - Ground-truth required warning flag (`True` / `False`)
  - Ground-truth evasive directional command (`STEP_LEFT`, `STEP_RIGHT`, `STOP`, `CLEAR`)
- **Split**: 10 canonical scenarios $\times 10	ext{ frames} = 100	ext{ benchmark frames}$ (100% evaluation partition; models are pretrained without test-set tuning)
- **Limitations**: Synthetic kinematics follow smooth constant-velocity models; does not simulate sudden erratic pedestrian direction switches.

---

## 3. Real-World Field Pilot Dataset
- **Dataset Name**: Controlled Real-World Field Validation Suite (`RW_001` to `RW_012`)
- **Source**: Controlled physical recordings collected by trained research operators using a chest-mounted USB wide-angle sensor ($78^\circ	ext{ FOV}$)
- **Purpose**: Evaluate domain transfer, lighting robustness, gait perturbations, and wearable audio usability outside the canonical dataset
- **Tested Physical Environments**:
  - **Environment A**: Indoor campus corridor (linoleum floor, lockers, standard fluorescent lighting $\sim 350	ext{ lux}$)
  - **Environment B**: Indoor open atrium (spacious lobby, high ceiling, diffuse natural lighting $\sim 450	ext{ lux}$)
  - **Environment C**: Outdoor walkway (paved concrete/asphalt, overcast daylight $\sim 1200	ext{ lux}$)
  - **Environment D**: Crowded hallway (supervised multi-person flow, 4 simultaneous pedestrians)
  - **Environment E**: Low-light environment (dim evening transition corridor $\sim 25	ext{ lux}$)
  - **Environment G**: Camera-motion environment (wearable walking gait: pitch $\pm 4^\circ$, roll $\pm 2.5^\circ$, $1.1	ext{ m/s}$)
- **Annotations Available**:
  - Frame-by-frame verified obstacle presence
  - Approximate physical distance markers (calibrated physical tape grid: $1.1	ext{m}$ to $5.5	ext{m}$)
  - Binary in-path corridor occupancy ($\pm 0.6	ext{m}$ lateral margin)
  - Safe evasive direction label (`STEP_LEFT`, `STEP_RIGHT`, `STOP`, `CLEAR`, or `UNKNOWN`)
- **Privacy & Ethical Safeguards**:
  - **Zero Facial Recognition**: No face identification, biometric storage, or identity logging
  - **Zero Audio Surveillance**: Operator microphone recording disabled; no bystander speech retained
  - **Controlled Transit**: Testing conducted exclusively in non-hazardous pedestrian spaces; traffic and live vehicular roadways strictly prohibited.
- **Limitations**: Limited to single-sensor monocular capture; does not span inclement weather conditions (heavy rain, dense fog, snow).
- **Licensing & Access**: Research prototype assets; maintained under open research license for academic reproducibility.
