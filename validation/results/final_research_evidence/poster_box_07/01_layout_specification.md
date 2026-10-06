# Section 7 Layout Specification — "PROTOTYPE & RESULTS"
**Poster Dimensions**: 1 m × 1 m Research Poster  
**Section Target Area**: Bottom 35% of total poster canvas (1000 mm × 350 mm equivalent grid)  
**Primary Question Answered**: *"What did we actually build, test, and measure?"*

---

## 1. Visual Hierarchy Architecture

| Level | Component | Focus / Visual Priority | Grid Position |
|---|---|---|---|
| **LEVEL 1** | **Research Results** | Baseline vs Proposed Comparison, Measured S03 TTC Telemetry Curve | Top-Center & Top-Right |
| **LEVEL 2** | **System Performance** | KPI Performance Strip (Latency, FPS, Test Suite Pass Rate) | Top Horizontal Banner |
| **LEVEL 3** | **Engineering Credibility** | Real 6-Panel Prototype Montage, Adaptive Computation Modes | Bottom Horizontal Strip |

---

## 2. Grid & Spatial Blueprint (1000 mm × 350 mm Canvas)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SECTION 7: PROTOTYPE & RESULTS                                         │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BLOCK A: PERFORMANCE KPI STRIP (Width: 1000mm, Height: 45mm)                                           │
│  [YOLO26n Latency] [Depth TRT Latency] [p50 E2E Latency] [Preliminary Throughput] [Unit Test Pass]     │
│   9.02 ms           7.82 ms            50.13 ms            17.68 FPS*          17 / 17             │
├──────────────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ BLOCK B: TEMPORAL RISK / TTC TELEMETRY   │ BLOCK C: BASELINE VS PROPOSED FRAMEWORK                     │
│ (Width: 480mm, Height: 160mm)            │ (Width: 480mm, Height: 160mm)                               │
│ • Real measured S03 TTC trajectory plot  │ • False warning comparison (0 vs 6)                         │
│ • Warning threshold overlay (3.0s)       │ • Unnecessary STOP comparison (0 vs 4)                      │
│ • Provenance: MEASURED • S03 APPROACH    │ • Provenance: CONTROLLED PROTOCOL • S01-S06                 │
├──────────────────────────────────────────┴─────────────────────────────────────────────────────────────┤
│ BLOCK D & E: ADAPTIVE COMPUTATION & REAL WORKING PROTOTYPE PIPELINE (Width: 1000mm, Height: 125mm)   │
│  [Adaptive Modes: 4:1 / 2:1 / 1:1]  │  [Real 6-Panel S03 Pipeline Montage: Detect -> Track -> Depth   │
│  [S01-S06 Scenario Distribution]    │   -> TTC -> Dynamic Risk -> Navigation Guidance]               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Typography & Styling System

- **Section Header**: 36 pt Bold Sans-Serif (Inter / Roboto), `#F8FAFC` on `#1E293B` container.
- **Block Titles**: 20 pt Bold, `#F8FAFC`.
- **KPI Large Numerals**: 34 pt ExtraBold, color-coded per category (`#38BDF8` Latency, `#34D399` Depth, `#FBBF24` E2E, `#A78BFA` FPS, `#F472B6` Tests).
- **KPI Subtext & Provenance**: 12 pt Medium, uppercase, `#94A3B8`.
- **Body & Annotations**: 14 pt Regular, line-height 1.4, `#E2E8F0`.
- **Provenance Badges**: 11 pt Bold, padding 4px, rounded corners, dark contrast fill.
