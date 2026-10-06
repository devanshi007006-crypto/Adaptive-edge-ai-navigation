# Phase 6 Experiment Log: Ground-Truth & Baseline Comparison

> **Document ID:** `docs/research/PHASE6_EXPERIMENT_LOG.md`  
> **Date:** October 5, 2026  
> **Experiment Status:** Completed & Verified

---

## Experiment Summary

- **Total Trials Executed:** 60 (30 System A Static Baseline + 30 System B Proposed Framework)
- **Scenarios Evaluated:** S01–S06 (5 repetitions per scenario)
- **Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`)
- **Execution Script:** `scripts/benchmark/run_phase6_study.py`

## Output Summary Matrix

| System | False Warning Rate (per min) | Median Lead Time Δt (s) | Nav Accuracy (%) | Unnecessary STOPs |
|:---|:---:|:---:|:---:|:---:|
| **System A (Static Baseline)** | 91.60 | 1.43 s | 45.6% | 9 |
| **System B (Proposed Framework)** | 0.00 | 1.50 s | 53.5% | 0 |
| **Net Scientific Gain** | **-100.0%** | **+0.07s** | **+7.9%** | **-100.0%** |

All primary outputs saved in `validation/results/phase6/`.
