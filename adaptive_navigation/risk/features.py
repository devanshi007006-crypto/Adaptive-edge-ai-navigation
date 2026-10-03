from dataclasses import dataclass
from typing import Optional
from .ttc import TTCResult
from temporal.velocity import VelocityEstimate

@dataclass
class RiskFeatures:
    """Normalized multi-dimensional risk features for an individual obstacle."""
    track_id: int
    normalized_distance: float
    approach_intensity: float
    ttc_factor: float
    path_overlap_ratio: float
    detection_confidence: float
    temporal_stability: float

class RiskFeatureExtractor:
    """Extracts normalized risk features from perception and temporal outputs."""
    def extract(self, track_id: int, depth: float, velocity: VelocityEstimate, ttc: TTCResult, path_overlap: float, confidence: float, stability: float) -> RiskFeatures:
        # Concrete normalization in STEP 10
        return RiskFeatures(
            track_id=track_id,
            normalized_distance=0.0,
            approach_intensity=0.0,
            ttc_factor=0.0,
            path_overlap_ratio=path_overlap,
            detection_confidence=confidence,
            temporal_stability=stability
        )
