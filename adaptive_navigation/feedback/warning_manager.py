from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .message_generator import WarningMessage, WarningPriority
from .speech import OfflineTTSInterface

class WarningManagerState(Enum):
    """D4 Warning State Machine States."""
    NO_ACTIVE_WARNING = "NO_ACTIVE_WARNING"
    CANDIDATE_WARNING = "CANDIDATE_WARNING"
    SPEAK = "SPEAK"
    COOLDOWN = "COOLDOWN"
    INTERRUPT = "INTERRUPT"

class WarningManager:
    """Implements D4 Audio-Warning Priority State Machine preventing warning spam."""
    def __init__(self, tts: OfflineTTSInterface, cooldown_seconds: float = 3.0, candidate_persistence_frames: int = 3):
        self.tts = tts
        self.cooldown_seconds = cooldown_seconds
        self.candidate_persistence_frames = candidate_persistence_frames
        self.current_state = WarningManagerState.NO_ACTIVE_WARNING
        self.last_spoken_time = 0.0
        self.last_spoken_priority: Optional[WarningPriority] = None

    def process_warning(self, candidate: WarningMessage, current_time: float) -> bool:
        # Concrete state transitions in STEP 14
        return False
