# Phase 4B — External Depth Calibration Validation Report

**Benchmark Stage**: Phase 4B External Metric Depth Validation  
**Evaluation Date**: October 5, 2026  
**Hardware Setup**: Host CPU / PyTorch 2.10.0+cu130 (`cuda:0` RTX 4050)  
**Primary Objective**: Evaluate whether the fixed Phase 4A affine metric depth calibration formulation ($\hat{Z} = \frac{1}{a \cdot d + b}$ with fixed $a = 0.103, b = 0.000$) generalizes across external HEADS-UP sequences without per-sequence parameter refitting.

---

## 1. Executive Summary & Calibration Formulation

* **Fixed Calibration Formulation**:
  $$\hat{Z} = \frac{1}{a \cdot d_{\text{rel}} + b} = \frac{1}{0.103 \cdot d_{\text{rel}}} = \frac{9.7087}{d_{\text{rel}}} \text{ (meters)}$$
* **No-Refitting Enforcement**: Parameters $a=0.103, b=0.000$ fitted during Phase 4A were held **strictly fixed** across all 8 external HEADS-UP sequences.
* **Test Coverage**: $N_{\text{val}} = 3,919$ valid obstacle observations evaluated across 8 behaviorally stratified HEADS-UP sequences covering near range, mid range, far range, head motion, dense crowd, approach, receding, and lateral crossing.

---

## 2. Overall External Accuracy & Error Metrics

| Metric | Overall Value | Scientific Interpretation |
| :--- | :---: | :--- |
| **Valid Comparisons ($N_{\text{val}}$)** | **3,919** | Evaluated across 8 HEADS-UP validation sequences |
| **Mean Absolute Error (MAE)** | **0.688 m** | Overall mean error across 1.0m–10.0m range |
| **Median Absolute Error** | **0.200 m** | **20.0 cm median error** across typical operational distance |
| **Root Mean Square Error (RMSE)** | **5.098 m** | Outlier-sensitive metric driven by far-range background variance |
| **Mean Absolute Percentage Error (MAPE)** | **9.2 %** | High overall relative accuracy across distance spectrum |
| **Bias (Mean Signed Error)** | **+0.477 m** | Slight distance overestimation bias at far range |

---

## 3. Distance Zone Breakdown

Accuracy was evaluated across three operational proximity zones:
* **NEAR Zone**: $< 2.5\text{ m}$
* **MID Zone**: $2.5\text{ m} - 5.0\text{ m}$
* **FAR Zone**: $> 5.0\text{ m}$

| Proximity Zone | Range Bounds | Sample Count ($N_{\text{val}}$) | MAE (m) | Median AE (m) | RMSE (m) | MAPE (%) | Bias (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NEAR Zone** | $< 2.5\text{ m}$ | 1,143 | **0.123 m** | **0.099 m** | 0.160 m | 7.1 % | +0.059 m |
| **MID Zone** | $2.5\text{ m} - 5.0\text{ m}$ | 1,327 | **0.235 m** | **0.191 m** | 0.307 m | 6.8 % | +0.083 m |
| **FAR Zone** | $> 5.0\text{ m}$ | 1,449 | **1.549 m** | **0.483 m** | 8.378 m | 13.1 % | +1.169 m |

> [!NOTE]
> **Key Finding on Distance Scaling**:
> Metric distance accuracy is exceptionally high in the critical close-range navigation zone ($<2.5\text{m}$ MAE $= 12.3\text{ cm}$), maintaining sub-quarter-meter median accuracy up to $5.0\text{m}$. Beyond $5.0\text{m}$, monocular depth scale compression introduces higher relative variance ($1.549\text{m}$ MAE), confirming that metric depth should be used primarily for proximity zone classification rather than long-range ranging.

---

## 4. Per-Sequence Validation Results

| Sequence ID | Category | $N_{\text{val}}$ | MAE (m) | Median AE (m) | RMSE (m) | MAPE (%) | Bias (m) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `HU_U01_multiped` | Multi-Pedestrian | 304 | 0.332m | 0.243m | 0.477m | 6.7% | +0.139m |
| `HU_U02_approach` | Steady Approach | 178 | 0.354m | 0.245m | 0.503m | 6.2% | +0.092m |
| `HU_U03_headmotion` | Strong Head Motion | 326 | 1.672m | 0.362m | 3.635m | 14.2% | +1.563m |
| `HU_U04_dense_crowd` | Dense Plaza Crowd | 669 | 0.451m | 0.306m | 0.670m | 6.8% | +0.125m |
| `HU_U05_crossing` | Lateral Crossing | 603 | 1.969m | 0.170m | 12.671m | 18.2% | +1.799m |
| `HU_U06_close_following` | Close Following | 865 | 0.187m | 0.136m | 0.253m | 6.7% | +0.047m |
| `HU_U07_receding` | Receding Pedestrians | 621 | 0.344m | 0.218m | 0.509m | 7.3% | +0.123m |
| `HU_U08_disappearance` | Track Disappearance | 353 | 0.348m | 0.196m | 0.555m | 6.6% | +0.052m |

---

## 5. Head-Pitch / Camera-Pose Stability Analysis

The stability of the fixed Affine Inverse Model ($\hat{Z} = \frac{1}{a \cdot d + b}$) was compared against Ground-Plane Geometry Model C ($Z_{\text{ground}} = \frac{H}{\tan(\theta + \alpha)}$) under head tilt variations:

| Camera Pose State | Head Pitch Offset ($\Delta \theta$) | Affine Inverse Model MAE | Ground-Plane Model C MAE | Stability Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Normal Head Pose** | $|\Delta \theta| \le 5.0^\circ$ | **0.638 m** | 3.664 m | **$5.7\times$ lower error** |
| **High Head Motion** | $|\Delta \theta| > 5.0^\circ$ | **1.494 m** | 4.325 m | **$2.9\times$ lower error** |

### Analysis:
Under the evaluated head-motion conditions, the affine inverse depth model showed greater empirical stability than ground-plane geometry. In contrast, Ground-Plane Geometry Model C assumes a fixed flat ground contact line; when the user nods or tilts their head, the ground contact angle changes, causing catastrophic distance errors ($>3.6\text{m}$ MAE).

---

## 6. Proximity-Zone Categorization Stability

To test whether calibrated metric depth provides stable categorical spatial range assignments (`NEAR`, `MID`, `FAR`), predicted zones were compared against reference ground-truth zones:

* **Overall Zone Classification Agreement**: **$94.5\%$** ($3,705 / 3,919$ zone-classification agreement on the evaluated external sample)

### Proximity Zone Confusion Matrix

| Reference Zone \ Predicted Zone | Predicted NEAR ($<2.5\text{m}$) | Predicted MID ($2.5-5.0\text{m}$) | Predicted FAR ($>5.0\text{m}$) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Reference NEAR ($<2.5\text{m}$)** | **1,070** | 73 | 0 | **93.6%** |
| **Reference MID ($2.5-5.0\text{m}$)** | 48 | **1,228** | 51 | **92.5%** |
| **Reference FAR ($>5.0\text{m}$)** | 0 | 42 | **1,407** | **97.1%** |

> [!IMPORTANT]
> **Safety Boundary Performance & Ranging Limitation**:
> Zero observations ($0 / 1,143$) from the reference NEAR zone were misclassified as FAR, and zero observations from the FAR zone were misclassified as NEAR. Minor misclassifications occur strictly between adjacent zones (NEAR $\leftrightarrow$ MID or MID $\leftrightarrow$ FAR). Note that metric depth is NOT reliable for precise ranging beyond 5 meters due to high variance and scale compression. These zone assignments provide coarse spatial context for navigation decision making, but should **NOT** be interpreted as absolute safety guarantees or exact metric measurements.

---

## 7. Artifact Summary

All Phase 4B external depth validation artifacts are saved in `validation/results/phase4/`:

* Report: `phase4b_external_depth_validation.md`
* Table 1: `depth_external_validation.csv`
* Metrics: `depth_external_metrics.json`
* Plots (`validation/results/phase4/plots/phase4b/`):
  1. `predicted_vs_reference_depth.png` — Scatter plot of predicted vs reference depth
  2. `depth_error_vs_distance.png` — Residual error scatter vs distance
  3. `depth_error_by_sequence.png` — Bar chart of MAE and RMSE per sequence
  4. `depth_error_by_zone.png` — Residual error boxplots by proximity zone

---

## 8. Final Decision & Classification

**Classification**: **B) USABLE WITH LIMITATIONS**

### Final Report & Readiness Summary:
1. **External Accuracy**: External MAE $= 0.688\text{ m}$, Median AE $= 0.200\text{ m}$ ($20.0\text{ cm}$), MAPE $= 9.2\%$, RMSE $= 5.098\text{ m}$.
2. **Per-Zone Error**:
   - NEAR Zone ($<2.5\text{m}$): MAE $= 0.123\text{ m}$ ($12.3\text{ cm}$), Median AE $= 0.099\text{ m}$ ($9.9\text{ cm}$)
   - MID Zone ($2.5\text{m}-5.0\text{m}$): MAE $= 0.235\text{ m}$ ($23.5\text{ cm}$), Median AE $= 0.191\text{ m}$
   - FAR Zone ($>5.0\text{m}$): MAE $= 1.549\text{ m}$, Median AE $= 0.483\text{ m}$
3. **Strongest Failure Mode**: Monocular scale compression beyond $5.0\text{m}$ increases variance (+1.169m bias in FAR zone); metric depth is NOT reliable for precise $>5\text{m}$ ranging.
4. **Generalization Verdict**: Fixed affine calibration ($\hat{Z} = \frac{1}{0.103 \cdot d_{\text{rel}}}$) generalizes well without refitting across all 8 external sequences for coarse proximity range categorization ($94.5\%$ zone-classification agreement on the evaluated external sample).
5. **Final Recommendation**:
   * **The metric-depth component IS READY for use as a static proximity-zone feature** (`NEAR`, `MID`, `FAR` range categorization).
   * **Dynamic Time-to-Collision (TTC) remains strictly scale-invariant and disparity-based** ($\tau = d / \dot{d}$), as established in Phase 4A.
