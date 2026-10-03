from dataclasses import dataclass

@dataclass
class UncertaintyMetric:
    """Quantifies uncertainty in tracking, depth, and detection confidence."""
    detection_uncertainty: float
    tracking_uncertainty: float
    depth_uncertainty: float
    composite_uncertainty: float

class UncertaintyEstimator:
    """Computes uncertainty metrics to penalize unreliable risk scores."""
    def estimate(self, confidence: float, track_length: int, depth_std: float) -> UncertaintyMetric:
        # Concrete implementation in STEP 11
        return UncertaintyMetric(
            detection_uncertainty=1.0 - confidence,
            tracking_uncertainty=max(0.0, 1.0 - (track_length / 10.0)),
            depth_uncertainty=0.0,
            composite_uncertainty=0.0
        )
