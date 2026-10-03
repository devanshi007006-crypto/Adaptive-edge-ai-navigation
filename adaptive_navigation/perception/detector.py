from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Detection:
    """Represents a single detected bounding box and class attribution."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[float, float]              # (cx, cy)
    timestamp: float

class DetectorInterface(ABC):
    """Abstract interface for 2D Object Detection models."""
    @abstractmethod
    def load_model(self, model_path: str, device: str) -> None:
        """Load detector checkpoint onto specified device."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        """Perform object detection on the input frame."""
        pass

class YOLOObjectDetector(DetectorInterface):
    """Ultralytics YOLO implementation of the detector interface."""
    def __init__(self, model_path: str = "yolo11n.pt", confidence_thresh: float = 0.35, device: str = "cpu"):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh
        self.device = device
        self.model = None

    def load_model(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        # Model loading will be executed in STEP 3

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        # Concrete implementation in STEP 3
        return []
