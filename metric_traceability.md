# Metric Traceability Matrix & Provenance Audit
**Repository**: `Adaptive-edge-ai-navigation`  
**Purpose**: Trace every reported benchmark and validation metric backward to its originating code line, data source, and generation mechanism.

---

## 1. Metric Traceability Assessment Framework

Every metric reported in repository deliverables (`FINAL_VALIDATION_SUMMARY.md`, `final_validation_matrix.csv`, `evaluation/results/metrics.json`, `evaluation/results/real_world_metrics.json`, and `presentation/paper_results.md`) is evaluated against six rigorous audit questions:
1. **Measured from actual model output?** (Yes / No)
2. **Derived from ground truth?** (Yes / No — i.e., constructed by jittering or echoing GT)
3. **Simulated?** (Yes / No — generated via kinematic equation or synthetic model)
4. **Randomly generated?** (Yes / No — generated via `np.random` distribution)
5. **Timing measured or manually constructed?** (Measured / Constructed / Random)
6. **Can it be reproduced from raw input?** (Yes / No — from genuine image/video inputs)

---

## 2. Comprehensive Traceability Table

| Metric Name | Reported Value(s) | Source File & Line | Actual Model Output? | Derived from GT? | Simulated? | Randomly Generated? | Timing Provenance | Reproducible from Raw Input? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Detection Precision** | `1.0000` / `99.39%` | `run_evaluation.py:161`<br>`pilot_testing.py:463` | **No** | **Yes** | **Yes** | **Yes** (`np.random.normal`, `rand()`) | N/A | **No** |
| **Detection Recall** | `1.0000` / `95.80%` / `90.56%` | `run_evaluation.py:161`<br>`pilot_testing.py:465` | **No** | **Yes** | **Yes** | **Yes** (`det_noise > 0.10`) | N/A | **No** |
| **Detection Mean IoU** | `0.9576` | `run_evaluation.py:162-169`<br>`metrics.py:75` | **No** | **Yes** | **Yes** | **Yes** (`jitter_x = np.random.normal(0, 1.5)`) | N/A | **No** |
| **Tracking Stability Rate** | `0.9200` / `100.0%` / `99.58%` | `run_evaluation.py:171`<br>`pilot_testing.py:475` | **No** | **Yes** | **Yes** | **Yes** (Arbitrary `if f_idx == 11`) | N/A | **No** |
| **Track Loss Rate** | `0.0042` (`0.42%`) | `pilot_testing.py:477`<br>`real_world_metrics.json:10` | **No** | **No** | **Yes** | **Yes** (1 drop forced in 240 frames) | N/A | **No** |
| **Depth MAE** | `0.0962 m` / `0.109 m` | `run_evaluation.py:174`<br>`pilot_testing.py:490` | **No** | **Yes** | **Yes** | **Yes** (`depth_err = np.random.normal(0, 0.12)`) | N/A | **No** |
| **Depth RMSE** | `0.1311 m` | `run_evaluation.py:174`<br>`metrics.py:126` | **No** | **Yes** | **Yes** | **Yes** (`np.random.normal(0, 0.12)`) | N/A | **No** |
| **Depth Relative Error** | `0.0265` (`2.65%`) | `run_evaluation.py:175`<br>`metrics.py:128` | **No** | **Yes** | **Yes** | **Yes** (`np.random.normal(0, 0.12)`) | N/A | **No** |
| **Velocity / Motion Error**| `< 0.04 m/s` / `100% dir` | `run_evaluation.py:223`<br>`final_validation_matrix.csv:5` | **No** | **Yes** | **Yes** | **Yes** (`np.random.normal(0, 0.05)`) | N/A | **No** |
| **TTC MAE** | `0.1285 s` / `0.101 s` / `0.117 s` | `run_evaluation.py:181`<br>`pilot_testing.py:496` | **No** | **Yes** | **Yes** | **Yes** (`ttc_err = np.random.normal(0, 0.15)`) | N/A | **No** |
| **TTC Tolerance Acc. (±0.5s)**| `1.0000` (`100%`) | `run_evaluation.py:182`<br>`metrics.py:178` | **No** | **Yes** | **Yes** | **Yes** (Bound by 3-sigma of 0.15) | N/A | **No** |
| **Risk Accuracy** | `95.38%` / `92.0%` | `run_evaluation.py:202` | **No** | **Yes** | **Yes** | **Yes** (`gt if rand() > 0.08 else MEDIUM`) | N/A | **No** |
| **Risk Macro F1** | `0.9550` / `0.9160` | `run_evaluation.py:202`<br>`metrics.py:214` | **No** | **Yes** | **Yes** | **Yes** (Derived from 8% mutation) | N/A | **No** |
| **Reliability ECE** | `0.1305` | `run_evaluation.py:209-214`<br>`metrics.py:254` | **No** | **No** | **Yes** | **Yes** (`uniform(0.75, 0.95)`, `rand() > 0.05`) | N/A | **No** |
| **Warning Precision** | `0.9583` / `0.9806` | `run_evaluation.py:246-250`<br>`metrics.py:277` | **No** | **Yes** | **Yes** | **Yes** (`is_warn or (raw and rand() < 0.20)`) | N/A | **No** |
| **Warning Recall** | `1.0000` / `73.72%` | `run_evaluation.py:248`<br>`real_world_metrics.json:21` | **No** | **Yes** | **Yes** | **Yes** (Forced `is_warn_gt or ...`) | N/A | **No** |
| **False Warning Rate** | `0.0370` / `0.0194` (`1.94%`)| `run_evaluation.py:248`<br>`pilot_testing.py:542` | **No** | **Yes** | **Yes** | **Yes** (`rand() < 0.20` filter) | N/A | **No** |
| **Critical Missed Warnings**| `0.00%` | `run_evaluation.py:248`<br>`FINAL_VALIDATION_SUMMARY.md:28`| **No** | **Yes** | **Yes** | **Yes** (Hardcoded preservation of GT) | N/A | **No** |
| **Navigation Accuracy** | `94.00%` / `74.4%` | `run_evaluation.py:259`<br>`pilot_testing.py:549` | **No** | **Yes** | **Yes** | **Yes** (`gt if rand() > 0.06 else CAUTION`) | N/A | **No** |
| **Navigation Macro F1** | `0.9255` | `run_evaluation.py:259`<br>`metrics.py:214` | **No** | **Yes** | **Yes** | **Yes** (Derived from 6% mutation) | N/A | **No** |
| **Processing Latency** | `11.55 ms` / `19.83 ms` | `run_evaluation.py:97-152`<br>`pilot_testing.py:377-388`| **No** | **No** | **Yes** | **Constructed / Random** (`sleep` + offset) | **No** |
| **Throughput (FPS)** | `86.58 FPS` / `50.44 FPS` | `benchmark.py:131`<br>`real_world_metrics.json:39` | **No** | **No** | **Yes** | **Constructed** ($1000 / \text{fake latency}$) | **No** |
| **Warning Decision Latency**| `74.20 ms` / `98.84 ms` | `pilot_testing.py:388`<br>`real_world_metrics.json:25` | **No** | **No** | **Yes** | **Constructed** (Sum of random floats) | **No** |
| **Navigation-to-Audio Lat.**| `23.40 ms` / `32.24 ms` | `FINAL_VALIDATION_SUMMARY.md:26`<br>`real_world_metrics.json:33` | **No** | **No** | **Yes** | **Constructed** (`aud_lat = 0.11 + uniform`) | **No** |
| **Memory Peak / Drift** | `1,180 MB` / `+0.2 MB` | `FINAL_VALIDATION_SUMMARY.md:14,33` | **No** | **No** | **Yes** | **Unmeasured / Manually Typed** | **No** |

---

## 3. Deep-Dive Metric Origin Investigations

### 3.1 Detection Metrics (IoU = 0.9576, Recall = 95.8% / 100%)
* **Mechanism**:
  ```python
  # evaluation/run_evaluation.py: Line 162
  jitter_x = np.random.normal(0, 1.5)
  jitter_y = np.random.normal(0, 1.5)
  pred_box = GroundTruthBBox(
      x1=gt_obj.bbox.x1 + jitter_x,
      y1=gt_obj.bbox.y1 + jitter_y,
      x2=gt_obj.bbox.x2 + jitter_x,
      y2=gt_obj.bbox.y2 + jitter_y,
  )
  ```
* **Findings**:
  - The detector (`YOLOObjectDetector`) was never called.
  - Adding a Gaussian offset with standard deviation $\sigma = 1.5$ pixels on a $640 \times 480$ image mathematically guarantees an IoU between $0.93$ and $0.98$ for boxes larger than $50 \times 50$ pixels.
  - The resulting IoU ($0.9576$) is solely a property of the injected Gaussian distribution $\mathcal{N}(0, 1.5^2)$, not YOLO11n.

### 3.2 Depth Metrics (MAE = 0.0962 m, RMSE = 0.1311 m)
* **Mechanism**:
  ```python
  # evaluation/run_evaluation.py: Line 174
  depth_err = np.random.normal(0, 0.12)
  pr_depth = max(0.5, gt_obj.depth + depth_err) if gt_obj.depth else None
  ```
* **Findings**:
  - For a zero-mean normal distribution with $\sigma = 0.12$, the theoretical expected value of the absolute error is $\mathbb{E}[|X|] = \sigma \sqrt{2/\pi} = 0.12 \times 0.7979 = 0.0957\text{ m}$.
  - The reported MAE of $0.0962\text{ m}$ matches the analytical expectation of $\mathcal{N}(0, 0.12^2)$ within sample variance ($N=130$).
  - Depth Anything V2 was never run, and no physical laser distance ground truths were ever recorded.

### 3.3 TTC Metrics (MAE = 0.1285 s, Tolerance Accuracy = 100.0%)
* **Mechanism**:
  ```python
  # evaluation/run_evaluation.py: Line 181
  ttc_err = np.random.normal(0, 0.15)
  pr_ttc = max(0.2, gt_obj.ttc_seconds + ttc_err)
  ```
* **Findings**:
  - The theoretical mean absolute error for $\sigma = 0.15$ is $0.15 \times \sqrt{2/\pi} \approx 0.1197\text{ s}$.
  - The maximum error under $3\sigma$ is $3 \times 0.15 = 0.45\text{ s}$.
  - Because $0.45\text{ s} < 0.50\text{ s}$, the metric `ttc_within_0.5s_accuracy` was guaranteed to evaluate to exactly $1.0000$ ($100.0\%$).
  - The underlying kinematic TTC model was not evaluated.

### 3.4 False Warning Reduction (93.3% raw -> 1.94% / 3.70% stabilized)
* **Mechanism**:
  ```python
  # evaluation/run_evaluation.py: Lines 246-248
  raw_warn = is_warn_gt or (np.random.rand() < 0.15) # 15% spurious raw spikes
  stab_warn = is_warn_gt or (raw_warn and np.random.rand() < 0.20)
  ```
* **Findings**:
  - The ablation study in `presentation/paper_results.md` claiming a "75.0% absolute reduction in nuisance alert frequency" was constructed by multiplying independent random Bernoulli trials ($0.15 \times 0.20 = 0.03 = 3\%$).
  - The temporal state machine in `adaptive_navigation/warning/state_machine.py` was never evaluated on noisy video streams.

### 3.5 Latency and Throughput (86.58 FPS, 11.55 ms latency)
* **Mechanism**:
  ```python
  # evaluation/run_evaluation.py: Lines 97-150
  t0 = time.perf_counter()
  time.sleep(0.001)
  det_ms = (time.perf_counter() - t0) * 1000.0 + 1.2
  time.sleep(0.0005)
  trk_ms = (time.perf_counter() - t0) * 1000.0 + 0.8
  time.sleep(0.002)
  dep_ms = (time.perf_counter() - t0) * 1000.0 + 15.0
  ```
* **Findings**:
  - These manual additions yield a total cycle time of $\sim 11.55\text{ ms}$ or $\sim 86.6\text{ FPS}$.
  - Depth Anything V2 VITS on CPU requires $> 150\text{ ms}$ per frame; on an RTX 3060 Laptop GPU in FP16, it typically requires $15\text{–}25\text{ ms}$ by itself.
  - The claimed 86.58 FPS throughput and sub-12 ms total pipeline latency are physically unattainable when running Depth Anything V2 synchronously on every frame without frame skipping, TensorRT INT8 optimization, or asynchronous threading.

---

## 4. Traceability Summary & Verdict

1. **Every reported metric in the paper results, validation summary, and evaluation matrix traces back to random number generation and hardcoded arithmetic offsets.**
2. **Zero metrics reflect real-world execution of YOLO11n, BoT-SORT, Depth Anything V2, or the navigation decision engine.**
3. **All existing evaluation tables and markdown summaries must be deprecated and replaced with genuine, reproducible evaluations.**
