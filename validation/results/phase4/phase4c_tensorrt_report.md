# Phase 4C — ONNX Runtime CUDA FP16 Optimization Benchmark Report

**Benchmark Unit**: Real-World Multimodal Inference Pipeline (YOLO11n + BoT-SORT + Depth Anything V2 + Risk Engine + Navigation Engine)  
**Evaluation Date**: October 5, 2026  
**Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6144 MB VRAM, `cuda:0`)  
**Host System**: Windows x86_64, PyTorch 2.10.0+cu130, CUDA 13.0  
**Active Execution Provider**: `CUDAExecutionProvider` (ONNX Runtime CUDA FP16 Acceleration)  
**TensorRT EP Status**: `Unavailable in this run` (native `nvinfer_10.dll` missing on system PATH; runtime executed graceful fallback to ONNX Runtime CUDA FP16 execution provider)

> [!NOTE]
> **Audit Correction Notice**:
> The 2.20x speedup ($5.57 \to 12.24\text{ FPS}$) reported in this Phase 4C benchmark represents **ONNX Runtime CUDA FP16 acceleration** (`CUDAExecutionProvider`), NOT a native TensorRT FP16 engine speedup.
> 
> Native TensorRT 11.3 FP16 engine verification is conducted in **[Phase 4C.1 Native TensorRT Audit Report](phase4c1_native_tensorrt_audit.md)**, where Native TensorRT FP16 engines (`.engine`) achieved **15.74 FPS** (**2.60x speedup**, 33.81 ms depth latency) on the RTX 4050 GPU and were formally classified as **A) NATIVE TENSORRT ACCEPTED**.

---

## 1. Reference PyTorch FP32/AMP Baseline Specifications

Before optimization, a clean PyTorch reference baseline benchmark was executed on the target RTX 4050 GPU using a reproducible 120-frame validation stream (with 1:1 depth inference cadence across every frame):

* **Object Detector**: YOLO11n (`models/detector/yolo11n.pt`, $640 \times 640$ resolution, batch size 1)
* **Depth Estimator**: Depth Anything V2 (`vits`, `models/depth/depth_anything_v2_vits.pth`, $518 \times 518$ resolution, batch size 1)
* **Precision**: PyTorch FP32 / Automatic Mixed Precision (AMP)
* **Baseline Performance**:
  - Detector Mean Latency: $34.44\text{ ms}$
  - Depth Mean Latency: $139.29\text{ ms}$
  - End-to-End p50 Latency: **$158.38\text{ ms}$**
  - End-to-End p95 Latency: **$184.14\text{ ms}$**
  - Aggregate Throughput: **$5.57\text{ FPS}$**
  - Peak VRAM Allocation: **$449.28\text{ MB}$**

---

## 2. Model Export & Provider Audit

Export compatibility was investigated independently for each neural network model:

### 2.1 YOLO11n Object Detector
* **ONNX Export**: Successfully exported via `ultralytics` to ONNX opset 18 (`models/deployment/yolo11n.onnx`, 10.7 MB) with dynamic batching.
* **Execution Provider**: Fused PyTorch CUDA FP16 engine.

### 2.2 Depth Anything V2 (ViT-S)
* **ONNX Export**: Successfully exported via PyTorch to ONNX opset 14 (`models/deployment/depth_anything_v2_vits.onnx`, 97.7 MB) with constant folding.
* **Execution Provider Audit**:
  - `TensorrtExecutionProvider` attempted initialization but logged missing native DLL dependency (`nvinfer_10.dll`).
  - ONNXRuntime executed automatic fallback to `CUDAExecutionProvider` FP16 engine, compiling CUDA FP16 kernels directly on the RTX 4050 GPU.
  - No architecture alteration or layer replacement was performed; the original Depth Anything V2 ViT-S architecture was preserved intact.

---

## 3. Performance & Resource Comparison

| Pipeline Benchmark Metric | PyTorch FP32/AMP Baseline | ONNX Runtime CUDA FP16 | Acceleration / Speedup |
| :--- | :---: | :---: | :---: |
| **Object Detector Latency** | 34.44 ms | 24.75 ms | **1.39x Speedup** (28.1% faster) |
| **Depth Estimator Latency** | 139.29 ms | 50.44 ms | **2.76x Speedup** (63.8% faster) |
| **End-to-End p50 Latency** | **158.38 ms** | **63.60 ms** | **2.49x Speedup** (59.8% reduction) |
| **End-to-End p95 Latency** | **184.14 ms** | **90.71 ms** | **2.03x Speedup** (50.7% reduction) |
| **Aggregate Throughput (FPS)** | **5.57 FPS** | **12.24 FPS** | **2.20x Speedup** (+119.7% higher) |
| **Peak VRAM Allocation** | **449.28 MB** | **67.32 MB** | **85.0% VRAM Savings** (-381.96 MB) |

---

## 4. Numerical & Behavioral Equivalence Assessment

Outputs from PyTorch FP32 and ONNX Runtime CUDA FP16 pipelines were evaluated frame-by-frame on identical inputs:

| Evaluated Output Dimension | Quantitative Measured Value | Acceptance Tolerance | Status |
| :--- | :---: | :---: | :---: |
| **Depth Map Mean Absolute Error (MAE)** | 0.3520 | $< 0.50$ | **PASSED** |
| **Depth Map Max Absolute Error (MaxAE)** | 2.5855 | $< 5.00$ | **PASSED** |
| **Detector Bounding Box / Count Agreement** | **100.0%** | $> 95.0\%$ | **PASSED** |
| **Detector Confidence Mean Delta** | 0.0000 | $< 0.01$ | **PASSED** |
| **Track Assignment Stability** | **100.0%** | $> 95.0\%$ | **PASSED** |
| **Downstream Risk State Agreement** | **100.0%** | $> 98.0\%$ | **PASSED** |
| **Downstream Navigation State Agreement** | **100.0%** | $> 98.0\%$ | **PASSED** |

---

## 5. Formal Research Claim & Engineering Distinction

> [!IMPORTANT]
> **Engineering Optimization Rule**:
> ONNX Runtime CUDA FP16 optimization is an **ENGINEERING optimization**, NOT a scientific improvement in navigation accuracy.

### Explicit Research Claim:
> *"ONNX Runtime CUDA FP16 reduced measured latency and increased throughput on the RTX 4050 while maintaining measured behavioral/output agreement within defined tolerances."*

---

## 6. Artifact Summary

All generated Phase 4C optimization artifacts are saved in `validation/results/phase4/`:

* Report: `phase4c_tensorrt_report.md`
* Table 1: `phase4c_performance.csv`
* Table 2: `phase4c_equivalence.csv`
* Metrics: `phase4c_metrics.json`
* Exported ONNX Models (`models/deployment/`):
  1. `yolo11n.onnx` (10.7 MB)
  2. `depth_anything_v2_vits.onnx` (97.7 MB)
* Plots (`validation/results/phase4/plots/phase4c/`):
  1. `latency_comparison.png` — Pipeline stage latency comparison bar chart
  2. `fps_comparison.png` — Aggregate FPS throughput comparison
  3. `memory_comparison.png` — Peak GPU VRAM allocation comparison
  4. `output_equivalence.png` — Depth map spatial error residual visualization
  5. `ttc_comparison.png` — Downstream TTC timeline equivalence

---

## 7. Final Decision & Classification

**Classification**: **A) ACCEPTED — ONNX Runtime CUDA FP16 Accelerated**

### Summary:
1. **Models Exported**: YOLO11n ONNX (`models/deployment/yolo11n.onnx`) and Depth Anything V2 ViT-S ONNX (`models/deployment/depth_anything_v2_vits.onnx`).
2. **Execution Provider**: `CUDAExecutionProvider` (ONNX Runtime CUDA FP16).
3. **Measured Acceleration**: **2.20x FPS speedup** ($5.57 \to 12.24\text{ FPS}$), **59.8% reduction in p50 latency** ($158.38\text{ms} \to 63.60\text{ms}$), and **85.0% reduction in peak VRAM** ($449.28\text{MB} \to 67.32\text{MB}$).
4. **Equivalence Verdict**: 100.0% agreement in detector counts, track persistence, downstream risk states, and navigation decisions.
