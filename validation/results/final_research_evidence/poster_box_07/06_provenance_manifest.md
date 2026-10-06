# Poster Section 7 Provenance & Scientific Integrity Manifest

Every metric, figure, and claim presented in Section 7 is explicitly mapped to its origin, hardware environment, evaluation scope, and scientific classification tag.

---

## 1. Metric Provenance Audit Matrix

| Metric / Metric Name | Claimed Value | Provenance Classification | Telemetry / Source Document | Test & Hardware Environment |
|---|---|---|---|---|
| **YOLO26n Detection Latency** | **9.02 ms** | `MEASURED` | `reports/poster_evidence_report.md` | NVIDIA RTX 4050 Laptop GPU, FP16 CUDA PyTorch |
| **Depth TRT FP16 Latency** | **7.82 ms** | `MEASURED` | `reports/poster_evidence_report.md` | TensorRT FP16 Engine, $518	imes518$ input resolution |
| **p50 End-to-End Latency** | **50.13 ms** | `MEASURED` | `reports/FINAL_EVIDENCE_AUDIT.md` | Real S03 16-frame preliminary execution run |
| **System Throughput** | **17.68 FPS** | `MEASURED` *(Preliminary)* | `reports/poster_claims.md` | Preliminary measured run on 16-frame S03 sequence |
| **Unit Test Pass Rate** | **17 / 17** | `MEASURED` | `reports/FINAL_RESEARCH_PROTOTYPE_STATUS.md` | Automated pytest suite covering core edge modules |
| **False Warning Reduction** | **100%** | `CONTROLLED PROTOCOL` | `validation/results/phase6/baseline_vs_proposed.csv` | Controlled S01–S06 scenario evaluation suite |
| **Adaptive Cadence Throughput** | **24.5 / 18.2 / 14.93 FPS** | `MEASURED` | `tables/adaptive_computation_results.csv` | Synthetic/Controlled risk cadence switching evaluation |
| **Historical YOLO11n Latency** | **11.45 ms** | `HISTORICAL` | `tables/performance_results.csv` | Phase 4C historical benchmark run |
| **Theoretical TTC Plot** | *Excluded* | `ILLUSTRATIVE` | `plots/01_ttc_over_time_approaching.png` | **DO NOT USE AS EXPERIMENTAL RESULT** |

---

## 2. Standardized Provenance Tags Applied

1. **`MEASURED`**: Directly measured runtime performance recorded on real hardware (NVIDIA RTX 4050 Laptop GPU).
2. **`CONTROLLED PROTOCOL`**: Result produced during standardized controlled scenario testing (S01–S06 suite).
3. **`HISTORICAL`**: Legacy baseline measurement recorded during earlier development phases (e.g., Phase 4C YOLO11n).
4. **`ILLUSTRATIVE`**: Conceptual mathematical model or synthetic curve (excluded from measured results).

---

## 3. Mandatory Poster Labeling Rules Applied

- **Throughput Qualification**: The metric **17.68 FPS** is strictly labeled as `S03 preliminary measured run • RTX 4050` and is not represented as universal hardware throughput.
- **False Warning Qualification**: The phrase **100% false-warning reduction** is strictly appended with `across the controlled S01–S06 scenario suite`.
- **Image Integrity Verification**: All 6 panels in `final_pipeline_montage.png` originate directly from actual YOLO26n / TensorRT FP16 execution on real video `S03_approaching_r01.mp4`. Zero synthetic, placeholder, or black images are present.

---

## 4. Unsupported Claims Blacklist Compliance

The following phrases have been audited and **strictly excluded** from all Section 7 documentation:
- ❌ *"100% safe"*
- ❌ *"100% real-world accuracy"*
- ❌ *"clinical grade"*
- ❌ *"fully validated"*
- ❌ *"guaranteed navigation"*
- ❌ *"universally safer"*
- ❌ *"real-world false-warning reduction"*
- ❌ *"wearable deployment proven"*
- ❌ *"full 3D VIO"*
- ❌ *"closed-loop cloud intelligence"*
