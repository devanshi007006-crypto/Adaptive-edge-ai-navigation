# Poster Section 7 Visual Manifest

| Visual Asset | Included in Poster? | Source Path / Origin | Scientific Rationale & Role |
|---|---|---|---|
| **Real S03 TTC Plot (`real_ttc_plot.png`)** | **YES** | Real frame-by-frame telemetry from `S03_approaching_r01.mp4` | Primary temporal evidence graph showing measured TTC progression vs time. |
| **Baseline vs Proposed (`baseline_vs_proposed.png`)** | **YES** | `validation/results/phase6/baseline_vs_proposed.csv` | Core comparative research result showing false warning reduction across S01-S06. |
| **Adaptive Computation (`adaptive_computation.png`)** | **YES** | `validation/results/final_research_evidence/plots/05_adaptive_computation_vs_risk.png` | Engineering evidence demonstrating dynamic depth cadence adjustment vs risk level. |
| **Scenario Risk Distribution (`scenario_distribution.png`)** | **YES** | `validation/results/final_research_evidence/plots/06_scenario_risk_distribution.png` | Scenario coverage chart documenting testing across S01 to S06. |
| **Real Pipeline Montage (`final_pipeline_montage.png`)** | **YES** | Real S03 execution frames (`validation/results/final_research_evidence/frames/`) | Real system evidence proving functional 6-stage edge prototype. |
| **Poster Box 07 Preview (`preview.png`)** | **YES** | Synthetic layout renderer combining all visual blocks | High-resolution 2D layout mockup for Canva/InDesign reference. |
| **FPS/Latency Comparison (`04_fps_latency_comparison.png`)** | *OPTIONAL / SUPPORTING* | `validation/results/final_research_evidence/plots/04_fps_latency_comparison.png` | Engineering comparison graph (YOLO26n vs historical YOLO11n). |
| **Theoretical TTC Curve (`01_ttc_over_time_approaching.png`)** | **NO** | Theoretical plot | **REJECTED**: Formally flagged by audit as non-measured curve. |
| **Theoretical Risk Curve (`02_dynamic_risk_over_time.png`)** | **NO** | Theoretical plot | **REJECTED**: Formally flagged by audit as non-measured curve. |

---

## Detailed Specifications of Included Figures

### 1. `real_ttc_plot.png`
- **File Location**: `validation/results/final_research_evidence/poster_box_07/real_ttc_plot.png`
- **Dimensions**: $2100 	imes 1260$ pixels ($300	ext{ DPI}$)
- **Key Features**: Dual-axis plot (TTC in cyan on left Y-axis, Risk score in red on right Y-axis vs Time in seconds on X-axis), annotated warning threshold ($3.0	ext{s}$), first warning callout box, `MEASURED • S03 APPROACH` provenance tag.

### 2. `baseline_vs_proposed.png`
- **File Location**: `validation/results/final_research_evidence/poster_box_07/baseline_vs_proposed.png`
- **Key Features**: Grouped bar chart comparing Static Baseline vs Proposed Dynamic Framework across False Warnings, Unnecessary STOPs, Lead Time, and System Stability.

### 3. `adaptive_computation.png`
- **File Location**: `validation/results/final_research_evidence/poster_box_07/adaptive_computation.png`
- **Key Features**: Step plot showing Depth Cadence ($4:1 ightarrow 2:1 ightarrow 1:1$) and corresponding FPS throughput ($24.5 ightarrow 18.2 ightarrow 14.93	ext{ FPS}$) as risk escalates.

### 4. `scenario_distribution.png`
- **File Location**: `validation/results/final_research_evidence/poster_box_07/scenario_distribution.png`
- **Key Features**: Horizontal bar distribution chart showing risk severity classification across controlled scenarios S01 through S06.

### 5. `final_pipeline_montage.png`
- **File Location**: `validation/results/final_research_evidence/poster_box_07/final_pipeline_montage.png`
- **Dimensions**: $3600 	imes 1200$ pixels
- **Key Features**: 6 real video panels showing Detection, Tracking, TensorRT Depth, TTC calculation, Dynamic Risk evaluation, and Navigation output with live telemetry overlays.
