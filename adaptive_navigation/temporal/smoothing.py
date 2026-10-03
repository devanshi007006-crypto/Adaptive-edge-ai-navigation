from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class TemporalSmootherInterface(ABC):
    """Abstract interface for temporal filters."""
    @abstractmethod
    def smooth_value(self, current: float, previous: float) -> float:
        pass

    @abstractmethod
    def smooth_bbox(self, current: Tuple[float, float, float, float], previous: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        pass

class ExponentialMovingAverageSmoother(TemporalSmootherInterface):
    """Exponential Moving Average (EMA) smoother."""
    def __init__(self, alpha_value: float = 0.7, alpha_bbox: float = 0.7):
        self.alpha_value = alpha_value
        self.alpha_bbox = alpha_bbox

    def smooth_value(self, current: float, previous: float) -> float:
        return self.alpha_value * current + (1.0 - self.alpha_value) * previous

    def smooth_bbox(self, current: Tuple[float, float, float, float], previous: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        return tuple(
            self.alpha_bbox * c + (1.0 - self.alpha_bbox) * p
            for c, p in zip(current, previous)
        ) # type: ignore
