"""Uncertainty module re-exporting ReliabilityEstimator and ReliabilityAssessment."""
from uncertainty.reliability import (
    ReliabilityAssessment,
    SystemReliability,
    ReliabilityEstimator,
)
from dataclasses import dataclass

@dataclass
class UncertaintyMetric:
    """Legacy compatibility structure."""
    detection_uncertainty: float
    tracking_uncertainty: float
    depth_uncertainty: float
    composite_uncertainty: float

UncertaintyEstimator = ReliabilityEstimator

__all__ = [
    "ReliabilityAssessment",
    "SystemReliability",
    "ReliabilityEstimator",
    "UncertaintyMetric",
    "UncertaintyEstimator",
]
