from enum import Enum
from dataclasses import dataclass
from navigation.decision import NavigationAction, NavigationDecision

class WarningPriority(Enum):
    """D4 Audio-Warning Priority Levels."""
    P1_LOW = 1
    P2_MEDIUM = 2
    P3_HIGH = 3
    P4_CRITICAL = 4

@dataclass
class WarningMessage:
    """Structured warning message with priority and timestamp."""
    text: str
    priority: WarningPriority
    timestamp: float

class MessageGenerator:
    """Generates concise, action-oriented warning phrases conforming to speech policy."""
    def generate(self, decision: NavigationDecision, timestamp: float) -> WarningMessage:
        if decision.action == NavigationAction.STOP:
            return WarningMessage(text="STOP. Obstacle ahead.", priority=WarningPriority.P4_CRITICAL, timestamp=timestamp)
        elif decision.action == NavigationAction.MOVE_LEFT:
            return WarningMessage(text="Move left.", priority=WarningPriority.P3_HIGH, timestamp=timestamp)
        elif decision.action == NavigationAction.MOVE_RIGHT:
            return WarningMessage(text="Move right.", priority=WarningPriority.P3_HIGH, timestamp=timestamp)
        return WarningMessage(text="Path clear.", priority=WarningPriority.P1_LOW, timestamp=timestamp)
