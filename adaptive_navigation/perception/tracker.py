from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .detector import Detection

@dataclass
class TrackedObject:
    """Represents a temporally tracked object with persistent track ID."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float] # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[float, float]
    timestamp: float

class TrackerInterface(ABC):
    """Abstract interface for Multi-Object Trackers."""
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray, timestamp: float) -> List[TrackedObject]:
        """Update tracker state with new frame detections."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state."""
        pass

class BoTSORTTracker(TrackerInterface):
    """BoT-SORT tracker interface via Ultralytics."""
    def __init__(self, tracker_config: str = "botsort.yaml"):
        self.tracker_config = tracker_config
        self.tracker = None

    def update(self, detections: List[Detection], frame: np.ndarray, timestamp: float) -> List[TrackedObject]:
        # Concrete implementation in STEP 4
        return []

    def reset(self) -> None:
        self.tracker = None
