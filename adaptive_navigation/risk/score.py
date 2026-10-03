from dataclasses import dataclass
from enum import Enum
from typing import Dict
from .features import RiskFeatures
from .uncertainty import UncertaintyMetric

class RiskLevel(Enum):
    """Categorical risk severity levels."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskScoreResult:
    """Comprehensive risk score and associated severity level."""
    track_id: int
    numerical_score: float
    level: RiskLevel
    feature_contributions: Dict[str, float]
    is_discounted_by_uncertainty: bool

class RiskScorer:
    """Calculates weighted risk score and maps to prototype severity levels."""
    def __init__(self, weights: Dict[str, float], thresholds: Dict[str, float]):
        self.weights = weights
        self.thresholds = thresholds

    def compute_score(self, features: RiskFeatures, uncertainty: UncertaintyMetric) -> RiskScoreResult:
        # Concrete calculation in STEP 12
        return RiskScoreResult(
            track_id=features.track_id,
            numerical_score=0.0,
            level=RiskLevel.NONE,
            feature_contributions={},
            is_discounted_by_uncertainty=False
        )
