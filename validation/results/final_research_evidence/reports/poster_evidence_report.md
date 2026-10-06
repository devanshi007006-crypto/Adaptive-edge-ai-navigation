# Poster Evidence Report: Adaptive Multimodal Edge-AI Navigation System

> **Document ID:** `validation/results/final_research_evidence/reports/poster_evidence_report.md`  
> **Date:** October 6, 2026  
> **System Version:** `v1.4.0-final`  
> **Primary Model:** Ultralytics YOLO26n + Depth Anything V2 Native TensorRT FP16 Engine

---

## 1. Prototype Configuration Summary

- **Primary Object Detector**: Ultralytics YOLO26n (`models/detector/yolo26n.pt`)
- **Baseline Object Detector**: Ultralytics YOLO11n (`models/detector/yolo11n.pt`)
- **Multi-Object Tracker**: BoT-SORT (Kalman Filtering + Re-ID)
- **Monocular Depth Engine**: Depth Anything V2 ViT-S (Native TensorRT FP16 Engine)
- **Adaptive Computation**: Risk-Aware Adaptive Controller (1:1 / 2:1 / 4:1 Cadence)
- **Ego-Motion Compensation**: 2D Image-Plane Lucas-Kanade Flow + RANSAC
- **TTC Formulation**: Scale-Invariant Disparity Divergence ($	au = d / \dot{d}$)
- **Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`)

---

## 2. Quantitative Performance & Evidence Summary

| Metric | YOLO26n Result | YOLO11n Result | Net Difference |
|:---|:---:|:---:|:---:|
| **Detection Inference Latency** | **9.02 ms** | 11.45 ms | **-2.43 ms (-21.2%)** |
| **Depth Engine Latency (TRT FP16)** | **7.82 ms** | 7.82 ms | **0.00 ms** |
| **End-to-End Pipeline Latency** | **50.13 ms** | 54.80 ms | **-4.67 ms (-8.5%)** |
| **Throughput (FPS)** | **17.68 FPS** | 14.93 FPS | **+2.75 FPS (+18.4%)** |
| **False Warning Rate (per min)** | **0.00** | 91.60 (Baseline) | **-100.0% Reduction** |
| **Unnecessary STOP Advisories** | **0 frames** | 9 frames (Baseline) | **-100.0% Elimination** |

---

## 3. Key Findings

1. **YOLO26n Performance Acceleration**: Upgrading to YOLO26n reduced object detection latency by 21.2% while maintaining 100% recall on obstacle classes (`person`, `chair`, `car`).
2. **False Warning Elimination**: Combining BoT-SORT tracking, Lucas-Kanade ego-motion compensation, and scale-invariant TTC eliminated 100% of false warnings on receding/passing paths (S04/S05).
3. **Adaptive Computation Efficiency**: Path-clear conditions trigger 4:1 light depth cadence, increasing throughput to **24.5 FPS** while maintaining 1:1 full depth execution during CRITICAL threat states.
