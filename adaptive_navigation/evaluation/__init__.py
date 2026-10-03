"""Evaluation module for structured logging, metric calculations, and report generation."""
from .logger import FrameLogRecord, NavigationLogger
from .metrics import SystemMetrics, MetricsCalculator
from .report import ReportGenerator

__all__ = [
    "FrameLogRecord",
    "NavigationLogger",
    "SystemMetrics",
    "MetricsCalculator",
    "ReportGenerator",
]
