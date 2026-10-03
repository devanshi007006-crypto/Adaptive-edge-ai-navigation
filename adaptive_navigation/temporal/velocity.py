from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
from .history import TrackHistoryRecord

class ApproachState(Enum):
    """Categorical approach state of an obstacle."""
    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    STATIONARY = "STATIONARY"
    UNCERTAIN = "UNCERTAIN"

@dataclass
class VelocityEstimate:
    """Represents estimated object velocity and approach status."""
    track_id: int
    pixel_velocity: Tuple[float, float]  # (vx, vy) in pixels/sec
    depth_rate_of_change: float          # delta_depth / delta_t
    approach_state: ApproachState
    is_metric: bool                      # Whether depth_rate_of_change is in meters/sec or relative units/sec
    confidence: float

class VelocityEstimatorInterface(ABC):
    """Abstract interface for estimating velocities and approach states."""
    @abstractmethod
    def estimate(self, record: TrackHistoryRecord) -> VelocityEstimate:
        pass

class NumericalVelocityEstimator(VelocityEstimatorInterface):
    """Estimates velocity via finite differences over smoothed history."""
    def __init__(self, approaching_threshold: float = 0.05, receding_threshold: float = -0.05):
        self.approaching_threshold = approaching_threshold
        self.receding_threshold = receding_threshold

    def estimate(self, record: TrackHistoryRecord) -> VelocityEstimate:
        # Concrete implementation in STEP 7
        return VelocityEstimate(
            track_id=record.track_id,
            pixel_velocity=(0.0, 0.0),
            depth_rate_of_change=0.0,
            approach_state=ApproachState.UNCERTAIN,
            is_metric=False,
            confidence=0.0
        )
