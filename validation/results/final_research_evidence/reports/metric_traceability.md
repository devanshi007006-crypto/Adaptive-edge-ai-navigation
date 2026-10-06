# Metric Traceability Matrix

> **Document ID:** `validation/results/final_research_evidence/reports/metric_traceability.md`

| Metric | Source File | Source Run / Telemetry | Calculation Method |
|:---|:---|:---|:---|
| **YOLO26n Latency (9.02ms)** | `scripts/run/run_final_prototype.py` | Real inference run on `S03_approaching_r01.mp4` | Mean PyTorch CUDA event timer |
| **TRT FP16 Depth (7.82ms)** | `models/deployment/depth_anything_v2_vits_fp16.engine` | TRT FP16 engine benchmark | TensorRT C++ Execution Context timer |
| **Pipeline Throughput (17.68 FPS)** | `validation/results/final_research_evidence/telemetry/` | 16-frame test video run | Total frames / sum(loop_time) |
| **False Warning Rate (0.00)** | `validation/results/phase6/baseline_vs_proposed.csv` | Phase 6 30-trial controlled evaluation | False warnings / total trial duration |
