from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class CameraMotionEstimate:
    """Estimated ego-motion / camera translation and rotation."""
    dx: float
    dy: float
    rotation_rad: float
    confidence: float
    is_compensated: bool

class CameraMotionCompensatorInterface(ABC):
    """Abstract interface for camera ego-motion compensation."""
    @abstractmethod
    def estimate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> CameraMotionEstimate:
        pass

class OpticalFlowMotionCompensator(CameraMotionCompensatorInterface):
    """Optical flow based camera motion estimator."""
    def __init__(self, method: str = "farneback"):
        self.method = method

    def estimate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> CameraMotionEstimate:
        # Concrete implementation in STEP 8
        return CameraMotionEstimate(dx=0.0, dy=0.0, rotation_rad=0.0, confidence=0.0, is_compensated=False)
