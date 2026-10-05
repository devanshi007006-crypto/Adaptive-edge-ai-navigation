"""
Hardware Resource and Per-Module Latency Benchmark for Step 16 Research Evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import time
import os
import numpy as np
import psutil
import torch


@dataclass
class LatencyProfile:
    module_name: str
    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.samples)) if self.samples else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.samples)) if self.samples else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "p95_ms": round(self.p95, 2),
        }


@dataclass
class ResourceProfile:
    cpu_percent_mean: float
    cpu_percent_peak: float
    ram_used_mb_mean: float
    ram_used_mb_peak: float
    gpu_available: bool
    gpu_device_name: str
    gpu_vram_mb_peak: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkRunner:
    """
    Benchmarks system resources, per-module latency, and end-to-end throughput without fabricating results.
    """

    def __init__(self):
        self.profiles: Dict[str, LatencyProfile] = {
            "Detection (YOLO11n)": LatencyProfile("Detection (YOLO11n)"),
            "Tracking (BoT-SORT)": LatencyProfile("Tracking (BoT-SORT)"),
            "Depth (Depth Anything V2)": LatencyProfile("Depth (Depth Anything V2)"),
            "Temporal Buffer": LatencyProfile("Temporal Buffer"),
            "Motion Estimation": LatencyProfile("Motion Estimation"),
            "Camera Compensation": LatencyProfile("Camera Compensation"),
            "TTC Calculation": LatencyProfile("TTC Calculation"),
            "Risk Engine": LatencyProfile("Risk Engine"),
            "Reliability Layer": LatencyProfile("Reliability Layer"),
            "Temporal Warning Machine": LatencyProfile("Temporal Warning Machine"),
            "Spatial & Navigation": LatencyProfile("Spatial & Navigation"),
            "Message & Audio Dispatch": LatencyProfile("Message & Audio Dispatch"),
            "End-to-End Pipeline": LatencyProfile("End-to-End Pipeline"),
        }
        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []

    def record_module_time(self, module_name: str, duration_ms: float) -> None:
        if module_name in self.profiles:
            self.profiles[module_name].samples.append(duration_ms)

    def sample_system_resources(self) -> None:
        try:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self.ram_samples.append(mem.used / (1024 * 1024))
        except Exception:
            pass

    def get_resource_profile(self) -> ResourceProfile:
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "None (CPU Execution)"
        vram_peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if gpu_avail else None

        cpu_mean = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        cpu_peak = float(np.max(self.cpu_samples)) if self.cpu_samples else 0.0
        ram_mean = float(np.mean(self.ram_samples)) if self.ram_samples else 0.0
        ram_peak = float(np.max(self.ram_samples)) if self.ram_samples else 0.0

        return ResourceProfile(
            cpu_percent_mean=round(cpu_mean, 1),
            cpu_percent_peak=round(cpu_peak, 1),
            ram_used_mb_mean=round(ram_mean, 1),
            ram_used_mb_peak=round(ram_peak, 1),
            gpu_available=gpu_avail,
            gpu_device_name=gpu_name,
            gpu_vram_mb_peak=round(vram_peak, 1) if vram_peak is not None else None,
        )

    def get_latency_summary(self) -> List[dict]:
        return [p.to_dict() for p in self.profiles.values()]

    def get_effective_fps(self) -> float:
        e2e = self.profiles["End-to-End Pipeline"]
        if e2e.count > 0 and e2e.mean > 0:
            return round(1000.0 / e2e.mean, 2)
        return 0.0
