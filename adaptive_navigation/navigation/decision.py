from enum import Enum
from dataclasses import dataclass
from typing import Dict
from .zones import ZoneType, ZoneRisk
from .free_space import FreeSpaceSegment

class NavigationAction(Enum):
    """Discrete navigation instruction for visually impaired user."""
    CONTINUE_CENTER = "CONTINUE_CENTER"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    STOP = "STOP"

@dataclass
class NavigationDecision:
    """Decision output with confidence and reasoning."""
    action: NavigationAction
    confidence: float
    reason: str
    target_zone: ZoneType

class NavigationEngine:
    """Synthesizes zone risks and free space clearance into a safe navigation decision."""
    def __init__(self, critical_proximity_thresh: float = 0.8):
        self.critical_proximity_thresh = critical_proximity_thresh

    def decide(self, zone_risks: Dict[ZoneType, ZoneRisk], free_spaces: Dict[ZoneType, FreeSpaceSegment]) -> NavigationDecision:
        # Concrete decision logic in STEP 13
        return NavigationDecision(
            action=NavigationAction.CONTINUE_CENTER,
            confidence=1.0,
            reason="Path clear",
            target_zone=ZoneType.CENTER
        )
