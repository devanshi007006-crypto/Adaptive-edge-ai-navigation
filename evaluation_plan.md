# Scientific Validation Architecture & Evaluation Plan
**Repository**: `Adaptive-edge-ai-navigation`  
**Version**: 2.0-Validation-Architecture  
**Status**: Architecture Design & Specification  

---

## 1. Core Principles of the Validation Architecture

To eliminate the scientific integrity flaws identified in the audit, the new validation architecture enforces five non-negotiable architectural invariants:

1. **Unified Inference Pipeline**: The exact same inference pipeline (`AdaptiveNavigationPipeline`) must execute for all three validation modes (Synthetic, HEADS-UP External Dataset, and Real-World Video). No separate simulation paths or bypassed modules are permitted.
2. **Strict Epistemic Isolation**: The inference pipeline accepts strictly raw input frames (`FramePacket`) and sensory timestamps. Ground truth annotations are **never passed into the pipeline**; they are consumed strictly by the offline `MetricsEvaluator` at the end of the evaluation run.
3. **Zero Fabricated Benchmarks**: All metrics must be computed strictly from the mathematical comparison between actual model outputs and verified ground truth annotations. Missing data must be explicitly reported as `NOT_EVALUATED` rather than populated with synthetic fallbacks.
4. **Hardware-Accurate Profiling**: Latencies must be measured with high-resolution monotonic clocks (`time.perf_counter()`) on CPU and asynchronous synchronization events (`torch.cuda.Event`) on GPU. The first 10 frames of any run must be discarded as warm-up iterations to avoid JIT and model initialization skew.
5. **Clear Domain Demarcation**: Performance obtained in synthetic tests must be explicitly labeled as synthetic; real-world claims can only be made on verified physical sensor data.

---

## 2. Validation Triad Architecture

The validation architecture supports three distinct evaluation modes sharing a common inference core:

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Raw Input Providers                      │
                          └────────────────────────────────────────────────────────┘
                                    │                           │                 │
             ┌──────────────────────┴──────────┐                │                 │
             ▼                                 ▼                ▼                 ▼
   Mode A: Synthetic Generator        Mode B: HEADS-UP Adapter      Mode C: Real Video Loader
   - Rendered video frames            - Extracted RGB frames        - Locally recorded MP4s
   - Known kinematic trajectories     - 3D/2D bounding boxes        - Calibrated distance markers
   - Controlled noise/lighting        - Calibrated metric depth     - Physical obstacle tags
             │                                 │                                  │
             └─────────────────────────────────┼──────────────────────────────────┘
                                               │
                                               ▼
                             ┌───────────────────────────────────┐
                             │       Raw Frame Sequence          │
                             │  FramePacket(t, frame_idx, BGR)   │
                             └───────────────────────────────────┘
                                               │
                                               ▼
                      ═════════════════════════════════════════════════════
                      │         UNIFIED INFERENCE PIPELINE                │
                      │  - YOLO Detector (Real weights)                   │
                      │  - BoT-SORT Tracker (Real Kalman updates)         │
                      │  - Depth Anything V2 (Metric scale-calibrated)    │
                      │  - Temporal History & Motion Estimator            │
                      │  - Optical Flow Ego-Motion Compensator            │
                      │  - Kinematic TTC Engine                           │
                      │  - Multi-Factor Risk Assessment Engine            │
                      │  - Uncertainty & Reliability Layer                │
                      │  - Temporal Warning State Machine (Hysteresis)    │
                      │  - Spatial Corridor Navigation Engine             │
                      ═════════════════════════════════════════════════════
                                               │
                                               ▼
                             ┌───────────────────────────────────┐
                             │       PipelineOutputPacket        │
                             │  - Predictions (Detections, Depth,│
                             │    TTC, Risk, Warnings, Actions)  │
                             │  - Hardware latency timestamps    │
                             └───────────────────────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
          ┌───────────────────────┐                         ┌───────────────────────┐
          │   Per-Frame Logger    │                         │   MetricsEvaluator    │
          │  - evaluation_run.csv │                         │  - Reads GT data      │
          │  - evaluation_run.json│                         │  - Bipartite matching │
          └───────────────────────┘                         │  - Scientific metrics │
                                                            └───────────────────────┘
                                                                        │
                                                                        ▼
                                                            ┌───────────────────────┐
                                                            │   Validated Reports   │
                                                            │  - summary_table.csv  │
                                                            │  - metrics_report.json│
                                                            └───────────────────────┘
```

---

## 3. The Three Evaluation Modes

### Mode A: Controlled Synthetic Tests
* **Purpose**: Verify mathematical correctness, threshold sensitivity, and state machine transitions under mathematically exact ground-truth conditions.
* **Input Generation**: Synthesize actual visual video frames (e.g., using OpenCV drawing primitives, 3D textured primitives, or a lightweight simulation renderer) where objects approach, cross, or recede against varying backgrounds.
* **Ground Truth**: Exact mathematical 3D trajectories, known distances ($z$), closing velocities ($v_z$), and true Time-to-Collision ($TTC = z / v_z$).
* **Pipeline Execution**: The rendered synthetic frames are passed into the actual pipeline as standard BGR image matrices.
* **Integrity Guard**: Clearly labeled as `SYNTHETIC_VERIFICATION`. Never cited as real-world perception accuracy.

### Mode B: HEADS-UP External Dataset Evaluation
* **Purpose**: Benchmark object detection, multi-object tracking, and monocular depth estimation on established, peer-reviewed computer vision datasets.
* **Target Datasets**:
  1. **HEADS-UP** (Wearable / Egocentric Navigation Dataset for visually impaired assistance).
  2. **KITTI / MOT17 / MOT20** (for standardized multi-object tracking metrics: MOTA, IDF1, HOTA).
  3. **NYU-Depth V2 / KITTI Depth** (for standardized depth metrics: AbsRel, RMSE, $\delta < 1.25$).
* **Dataset Adapter**: A clean reader converting external annotations into the standardized `GroundTruthFrame` format.
* **Pipeline Execution**: Sequence frames are piped sequentially through the inference pipeline at their native resolution and frame intervals.

### Mode C: Locally Recorded Real-World Video Evaluation
* **Purpose**: Characterize system performance in target operational domains (corridors, walkways, atriums, low light) using body-worn or handheld camera rigs.
* **Data Collection Protocol**:
  - Record synchronized MP4 videos ($640 \times 480$ or $1280 \times 720$ @ 30 FPS).
  - Physical distance markers (calibrated tape on floor at $1.0\text{m}, 2.0\text{m}, 3.0\text{m}, 4.0\text{m}, 5.0\text{m}$).
  - Manual or semi-automated frame annotation using standard labeling tools (e.g., CVAT) to establish 2D bounding boxes and ground-truth distance checkpoints.
* **Pipeline Execution**: Real video files are streamed through `VideoFileStreamer` into the unified inference pipeline.
* **Safety Protocol**: Zero human hazard exposure; strictly non-traffic pedestrian zones.

---

## 4. Standardized Data Contracts

### 4.1 Input Contract: `FramePacket`
```python
@dataclass
class FramePacket:
    frame_index: int              # Sequential integer frame ID (0-indexed)
    timestamp: float              # Monotonic timestamp (seconds)
    frame: np.ndarray             # H x W x 3 uint8 BGR image
    camera_intrinsics: Optional[CameraIntrinsics] = None  # fx, fy, cx, cy
```

### 4.2 Output Contract: `PipelineOutputPacket`
```python
@dataclass
class PipelineOutputPacket:
    frame_index: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    depth_map: np.ndarray
    object_depths: List[TrackedObjectDepth]
    motion_estimates: Dict[int, MotionEstimate]
    camera_motion: CameraMotionEstimate
    compensated_motion: Dict[int, CompensatedMotionEstimate]
    ttc_results: Dict[int, TTCResult]
    risk_assessments: Dict[int, RiskAssessment]
    system_reliability: SystemReliability
    warning_decisions: Dict[int, WarningDecision]
    global_warning: GlobalWarningDecision
    navigation_decision: SceneNavigationState
    speech_message: Optional[WarningMessage]
    module_latency_ms: Dict[str, float]
    total_pipeline_latency_ms: float
```

### 4.3 Ground Truth Contract: `GroundTruthFrame`
```python
@dataclass
class GroundTruthFrame:
    frame_index: int
    timestamp: float
    objects: List[GroundTruthObject]   # BBox, class, metric depth, velocity, TTC
    hazard_present: bool
    safe_navigation_action: str        # 'CONTINUE', 'AVOID_LEFT', 'AVOID_RIGHT', 'STOP'
    environment_type: str
    lighting_lux: Optional[float]
```

---

## 5. Metric Calculation Specifications

All metrics must be evaluated through `validation/evaluators/`:

### 5.1 Object Detection Metrics
* **Matching**: Greedy bipartite matching with IoU threshold $\tau \in \{0.50, 0.75\}$ and COCO mAP@[.50:.95].
* **Outputs**: Precision, Recall, F1, Average Precision (AP) per class.

### 5.2 Multi-Object Tracking Metrics
* **Implementation**: Standard MOT evaluation (via `motmetrics` or native implementation).
* **Outputs**:
  - **MOTA** (Multi-Object Tracking Accuracy)
  - **IDF1** (Identification F1-score)
  - **HOTA** (Higher Order Tracking Accuracy)
  - **ID Switches** ($N_{\text{id-switches}}$)
  - **Fragmentation Count** ($N_{\text{frag}}$)

### 5.3 Monocular Depth Metrics
* **Alignment**: Median scaling against ground truth: $\hat{s} = \text{median}(z_{\text{gt}} / z_{\text{pred}})$ for relative depth, or unscaled for calibrated metric depth.
* **Standard Metrics**:
  - Absolute Relative Error: $\text{AbsRel} = \frac{1}{N} \sum \frac{|z - \hat{z}|}{z}$
  - Root Mean Squared Error: $\text{RMSE} = \sqrt{\frac{1}{N} \sum (z - \hat{z})^2}$
  - Threshold Accuracy: $\%$ of pixels/objects where $\max(\frac{z}{\hat{z}}, \frac{\hat{z}}{z}) < 1.25^k$ for $k \in \{1, 2, 3\}$.
  - Ordinal Error: Ranking accuracy across all pairs of simultaneous obstacles.

### 5.4 Kinematic TTC Metrics
* Evaluated only on confirmed closing targets ($v_{\text{close}} > 0.1\text{ m/s}$).
* **Outputs**:
  - TTC MAE: $\frac{1}{N} \sum |TTC_{\text{pred}} - TTC_{\text{gt}}|$
  - TTC Relative Error: $\frac{1}{N} \sum \frac{|TTC_{\text{pred}} - TTC_{\text{gt}}|}{TTC_{\text{gt}}}$
  - Early/Late Warning Bias: $\text{Bias} = \text{mean}(TTC_{\text{pred}} - TTC_{\text{gt}})$

### 5.5 Warning & Navigation Safety Metrics
* **Warning Lead Time**: Physical seconds elapsed between initial `CRITICAL` alert and time of collision / minimum proximity.
* **False Alert Frequency**: Number of false warning triggers per minute of operational video ($FWR / \text{min}$).
* **Navigation Decision Confusion Matrix**: Multi-class accuracy across `CONTINUE`, `CAUTION`, `AVOID_LEFT`, `AVOID_RIGHT`, `STOP`.

### 5.6 Hardware & Latency Profiling
* Measured using `time.perf_counter()` on synchronized streams.
* **Outputs**: Mean, P95, P99, and Peak latency per module and end-to-end.
* Effective FPS: $1000.0 / \text{Mean}(Latency_{\text{total}})$.

---

## 6. Execution Roadmap & Milestones

1. **Phase 1: Foundation Refactoring**
   - Implement `AdaptiveNavigationPipeline` wrapper class in `adaptive_navigation/pipeline.py`.
   - Implement `PipelineOutputPacket` serialization to CSV/JSON.
   - Fix Depth Anything V2 metric sign inversion and scaling in `ttc.py`.
2. **Phase 2: Validation Harness Implementation**
   - Create clean evaluation directory structure `validation/`.
   - Implement `SyntheticVideoGenerator` to generate real rendered video frames for Mode A.
   - Implement `HEADSUpDatasetAdapter` for Mode B.
   - Implement `RealWorldVideoEvaluator` for Mode C.
3. **Phase 3: Real Model Weight Provisioning & Pre-Flight Verification**
   - Download official YOLOv8n / YOLO11n `.pt` weights and Depth Anything V2 VITS `.pth`.
   - Verify execution on CPU and GPU (if CUDA available).
4. **Phase 4: Execution & Genuine Reporting**
   - Run Mode A (Synthetic Tests) on rendered video sequences.
   - Run Mode B (HEADS-UP / External Dataset) on real benchmark frames.
   - Run Mode C (Real-World Videos) on recorded pedestrian footage.
   - Output genuinely verified metric tables, deprecating all fabricated values.
