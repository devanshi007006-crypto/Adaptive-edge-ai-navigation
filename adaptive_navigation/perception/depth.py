from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class DepthResult:
    """Container for estimated depth data."""
    depth_map: np.ndarray                  # 2D float depth map
    is_metric: bool                        # True if metric depth (meters), False if relative depth
    min_depth: float
    max_depth: float
    timestamp: float

class DepthEstimatorInterface(ABC):
    """Abstract interface for monocular depth estimation."""
    @abstractmethod
    def load_model(self, model_path: str, device: str) -> None:
        """Load depth model weights."""
        pass

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray, timestamp: float) -> DepthResult:
        """Compute full-frame depth map."""
        pass

    @abstractmethod
    def get_object_depth(self, depth_result: DepthResult, bbox: Tuple[float, float, float, float]) -> float:
        """Estimate representative depth value for an object's bounding box."""
        pass

class DepthAnythingV2Estimator(DepthEstimatorInterface):
    """Depth Anything V2 model interface."""
    def __init__(self, model_path: str = "", is_metric: bool = False, device: str = "cpu"):
        self.model_path = model_path
        self.is_metric = is_metric
        self.device = device
        self.model = None

    def load_model(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        # Model loading in STEP 5

    def estimate_depth(self, frame: np.ndarray, timestamp: float) -> DepthResult:
        # Concrete implementation in STEP 5
        h, w = frame.shape[:2]
        dummy_map = np.zeros((h, w), dtype=np.float32)
        return DepthResult(depth_map=dummy_map, is_metric=self.is_metric, min_depth=0.0, max_depth=0.0, timestamp=timestamp)

    def get_object_depth(self, depth_result: DepthResult, bbox: Tuple[float, float, float, float]) -> float:
        # Concrete object depth extraction in STEP 5
        return 0.0
