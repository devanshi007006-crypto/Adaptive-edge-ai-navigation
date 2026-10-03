from dataclasses import dataclass
from typing import List, Dict
from .logger import FrameLogRecord

@dataclass
class SystemMetrics:
    """Benchmark performance metrics for prototype evaluation."""
    total_frames: int
    average_fps: float
    average_latency_ms: float
    total_warnings_generated: int
    total_warnings_suppressed: int
    decision_distribution: Dict[str, int]

class MetricsCalculator:
    """Calculates operational metrics across evaluation logs."""
    def compute(self, records: List[FrameLogRecord]) -> SystemMetrics:
        if not records:
            return SystemMetrics(0, 0.0, 0.0, 0, 0, {})
        avg_fps = sum(r.fps for r in records) / len(records)
        avg_latency = sum(r.latency_ms for r in records) / len(records)
        dist: Dict[str, int] = {}
        gen = 0
        supp = 0
        for r in records:
            dist[r.navigation_decision] = dist.get(r.navigation_decision, 0) + 1
            if r.warning_generated:
                gen += 1
            if r.warning_suppressed:
                supp += 1
        return SystemMetrics(
            total_frames=len(records),
            average_fps=avg_fps,
            average_latency_ms=avg_latency,
            total_warnings_generated=gen,
            total_warnings_suppressed=supp,
            decision_distribution=dist
        )
