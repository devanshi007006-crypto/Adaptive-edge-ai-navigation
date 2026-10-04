"""Uncertainty module for legacy risk compatibility."""
from dataclasses import dataclass

@dataclass
class UncertaintyMetric:
    """Legacy compatibility structure."""
    detection_uncertainty: float = 0.0
    tracking_uncertainty: float = 0.0
    depth_uncertainty: float = 0.0
    composite_uncertainty: float = 0.0

class UncertaintyEstimator:
    """Legacy compatibility estimator."""
    def estimate(self, *args, **kwargs) -> UncertaintyMetric:
        return UncertaintyMetric()

__all__ = [
    "UncertaintyMetric",
    "UncertaintyEstimator",
]
