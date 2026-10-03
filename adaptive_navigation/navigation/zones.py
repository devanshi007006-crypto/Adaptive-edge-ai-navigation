from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from risk.score import RiskScoreResult

class ZoneType(Enum):
    """Horizontal spatial partition."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"

@dataclass
class ZoneRisk:
    """Aggregated risk profile for a single spatial zone."""
    zone: ZoneType
    max_risk: float
    average_risk: float
    obstacle_count: int
    most_critical_track_id: Optional[int] = None

class ZoneDivider:
    """Partitions camera scene into LEFT, CENTER, and RIGHT zones."""
    def __init__(self, left_boundary: float = 0.33, right_boundary: float = 0.67):
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def classify_position(self, normalized_x: float) -> ZoneType:
        if normalized_x < self.left_boundary:
            return ZoneType.LEFT
        elif normalized_x > self.right_boundary:
            return ZoneType.RIGHT
        return ZoneType.CENTER

    def aggregate_zone_risks(self, obstacles_with_risks: List[Tuple[float, RiskScoreResult]]) -> Dict[ZoneType, ZoneRisk]:
        # Concrete implementation in STEP 13
        return {
            ZoneType.LEFT: ZoneRisk(ZoneType.LEFT, 0.0, 0.0, 0),
            ZoneType.CENTER: ZoneRisk(ZoneType.CENTER, 0.0, 0.0, 0),
            ZoneType.RIGHT: ZoneRisk(ZoneType.RIGHT, 0.0, 0.0, 0),
        }
