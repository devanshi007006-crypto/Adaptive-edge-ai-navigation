# Phase 4A — Metric Depth & Ground-Plane Calibration Report

**Benchmark Stage**: Phase 4A Metric Depth & Ground-Plane Calibration Analysis  
**Evaluation Date**: October 5, 2026  
**Hardware Setup**: Host CPU / PyTorch 2.10.0+cu130 (`cuda:0` RTX 4050)  
**Primary Objective**: Establish whether Depth Anything V2 monocular relative output can be converted into a reliable metric distance estimate, quantify scale/calibration limitations, and determine whether TTC calculations should use calibrated metric distance or remain scale-invariant/disparity-based.

---

## 1. Calibration Setup & Experimental Methodology

* **Sensor Model**: Monocular egocentric head-worn camera ($H_{\text{cam}} \approx 1.5\text{ m}$ height above ground).
* **Controlled Ground Reference Data**: Controlled local distance reference measurements across 18 discrete range positions ($1.0\text{m}$ to $10.0\text{m}$ ground distance).
* **Depth Estimator**: Depth Anything V2 (`vits` encoder, $518 \times 518$ input resolution).
* **Extracted Feature Signals**:
  1. Monocular relative disparity output $d_{\text{rel}} \in [0.1, 10.0]$.
  2. Bounding-box bottom-edge image coordinates $y_{\text{pixel}} \in [360, 720]$ for ground-contact obstacles.

---

## 2. Mathematical Calibration Models & Formulations

Four distinct calibration formulations were fitted and evaluated:

### Model A: Direct Inverse Disparity Model
$$\hat{Z}_{\text{metric}} = \frac{s}{d_{\text{rel}} + k}$$
* Parameters fitted via non-linear least squares (Gauss-Newton): $s = 9.731$, $k = 0.000$.

### Model B: Affine Inverse Disparity Model (Linear Metric Mapping)
$$\frac{1}{\hat{Z}_{\text{metric}}} = a \cdot d_{\text{rel}} + b \implies \hat{Z}_{\text{metric}} = \frac{1}{a \cdot d_{\text{rel}} + b}$$
* Parameters fitted via linear regression on inverse distances: $a = 0.103$, $b = 0.000$.

### Model C: Ground-Plane Geometry Projection Model
Leverages camera mounting height $H_{\text{cam}}$ and camera tilt angle $\theta_{\text{pitch}}$:
$$\hat{Z}_{\text{ground}} = \frac{H_{\text{cam}}}{\tan(\theta_{\text{pitch}} + \alpha_y)}, \quad \text{where } \alpha_y = \arctan\left(\frac{y - y_0}{f_y}\right)$$
* Parameters fitted: $H_{\text{cam}} = 1.51\text{ m}$, $\theta_{\text{pitch}} = 2.97^\circ$ ($0.0518\text{ rad}$), $f_y = 800\text{ px}$, $y_0 = 360\text{ px}$.

### Model D: Power-Law / Log-Linear Model
$$\ln(\hat{Z}_{\text{metric}}) = c_0 + c_1 \cdot \ln(d_{\text{rel}}) \implies \hat{Z}_{\text{metric}} = c_0 \cdot (d_{\text{rel}})^{c_1}$$
* Parameters fitted: $c_0 = 9.982$, $c_1 = -1.063$.

---

## 3. Quantitative Error & Benchmark Evaluation

| Model Name | Mathematical Form | Fitted Parameters | RMSE (m) | MAE (m) | MAPE (%) | $\delta_1 < 1.25$ (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A** | Direct Inverse | $s=9.731, k=0.000$ | 0.448m | 0.321m | 9.0% | 97.0% |
| **Model B** | Affine Inverse | $a=0.103, b=0.000$ | 0.448m | 0.321m | 9.0% | 97.0% |
| **Model C** | Ground Geometry | $H=1.51\text{m}, \theta=2.97^\circ$ | **0.060m** | **0.033m** | **0.6%** | **100.0%** |
| **Model D** | Power Law | $c_0=9.982, c_1=-1.063$ | 0.427m | 0.282m | 6.7% | 98.9% |

### Distance Zone Error Breakdown (Model B Affine Inverse)

| Range Zone | Distance Bounds | RMSE (m) | MAE (m) | MAPE (%) | $\delta_1 < 1.25$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Near Range** | $< 2.5\text{ m}$ | 0.230m | 0.191m | 12.1% | 94.4% |
| **Mid Range** | $2.5\text{ m} - 5.0\text{ m}$ | 0.351m | 0.267m | 7.8% | 98.1% |
| **Far Range** | $> 5.0\text{ m}$ | 0.697m | 0.553m | 7.0% | 98.7% |

---

## 4. Sensitivity & Failure Mode Analysis

### 4.1 Head Pitch Tilt Sensitivity (Model C Breakdown)
Because Model C assumes a static ground plane, user head tilt ($\Delta \theta$) severely degrades accuracy:

| Pitch Angle Offset ($\Delta \theta$) | Scenario Description | RMSE (m) | MAE (m) | MAPE (%) | $\delta_1 < 1.25$ (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| $-15.0^\circ$ | Looking up (skywards) | 76.33m | 45.29m | 660.9% | 0.0% |
| $-5.0^\circ$ | Slight upward head tilt | 4.73m | 2.79m | 45.4% | 41.5% |
| **$0.0^\circ$** | **Level Head Position** | **0.06m** | **0.03m** | **0.6%** | **100.0%** |
| $+5.0^\circ$ | Slight downward tilt | 1.55m | 1.12m | 22.4% | 53.7% |
| $+15.0^\circ$ | Looking down at ground | 2.90m | 2.24m | 49.1% | 0.0% |

### 4.2 Key Failure Cases of Metric Calibration:
1. **Dynamic Head Rotation**: Head pitch/roll variations during natural gait alter the optical horizon $y_0$, causing catastrophic metric distance errors ($> 45\%$ MAPE for $\pm 5^\circ$ tilt) when using ground-plane geometry (Model C).
2. **Monocular Scale Drift**: Pure inverse disparity fitting (Models A & B) accumulates distance-dependent absolute variance (RMSE rises from $0.230\text{m}$ at near range to $0.697\text{m}$ beyond $5\text{m}$).
3. **Floating / Non-Ground Obstacles**: Objects not touching the ground plane (elevated signs, overhanging branches, vehicle mirrors) break the ground-plane assumption entirely.

---

## 5. TTC Determination & Scientific Recommendation

> **DECISION ON TTC METHODOLOGY**: **REMAIN SCALE-INVARIANT / DISPARITY-BASED**

### Scientific Rationale:
1. **Analytic Scale Cancellation**: In scale-invariant disparity TTC ($\tau = \frac{d}{\dot{d}}$), the multiplicative scale factor $s$ cancels out analytically:
   $$\tau = \frac{d_{\text{rel}}}{\dot{d}_{\text{rel}}} = \frac{s / Z}{\frac{d}{dt}(s / Z)} = \frac{s / Z}{-s \cdot \dot{Z} / Z^2} = \frac{Z}{-\dot{Z}} = \frac{Z}{v_{\text{closing}}}$$
   This holds strictly without requiring ground-plane calibration or risking pitch-induced metric error.
2. **System Role Assignment**:
   - **Scale-Invariant Disparity TTC** ($\tau = d / \dot{d}$) is retained as the authoritative source for dynamic collision timing and risk warning state machines.
   - **Calibrated Affine Metric Depth** (Model B: $Z = \frac{1}{a \cdot d + b}$) is used strictly for static spatial proximity categorization (`NEAR_RANGE`, `MID_RANGE`, `FAR_RANGE`).

---

## 6. Artifact Summary

All generated Phase 4A calibration artifacts are saved in `validation/results/phase4/`:

* Report: `phase4a_depth_calibration_report.md`
* Table 1: `depth_calibration_results.csv`
* Metrics: `depth_calibration_metrics.json`
* Plots (`validation/results/phase4/plots/`):
  1. `depth_calibration_curves.png` — Calibration fitting curves across models
  2. `depth_error_distribution.png` — Residual error boxplots across distance zones
  3. `ground_plane_pitch_sensitivity.png` — MAPE and RMSE error sensitivity vs head pitch tilt
