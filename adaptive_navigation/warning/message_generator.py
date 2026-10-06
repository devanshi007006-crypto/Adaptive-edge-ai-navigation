"""
User-Facing Warning Message Generator (Step 13 & Step 14 Navigation).

Transforms stabilized GlobalWarningDecision, individual track decisions,
and Step 14 SceneNavigationState into concise, directional, natural language alert messages.

Features:
- Never fabricates absolute metrics in relative depth mode.
- Synchronizes physical audio repetition with real-world wall-clock pacing.
- Supports continuous guidance so active threats receive continuous audible assistance.
- Integrates Step 14 lateral avoidance guidance ("Move left", "Move right", "Stop").
- Immediately announces state escalations and direction shifts.
"""

from dataclasses import dataclass
import time
from typing import Dict, Optional

from warning.state_machine import GlobalWarningDecision, WarningDecision


@dataclass
class WarningMessage:
    """Structured user-facing warning message."""
    text: str
    priority: str                       # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    state: str                          # 'NO_WARNING', 'CAUTION', 'WARNING', 'CRITICAL', 'UNKNOWN'
    track_id: Optional[int]
    reason: str
    should_speak: bool
    timestamp: float


class WarningMessageGenerator:
    """
    Transforms stabilized GlobalWarningDecision, individual track decisions,
    and SceneNavigationState into prioritized, continuous natural language alerts.
    """

    STATE_PRIORITY_MAP = {
        "NO_WARNING": "NONE",
        "CAUTION": "LOW",
        "WARNING": "HIGH",
        "CRITICAL": "CRITICAL",
        "UNKNOWN": "NONE",
    }

    PRIORITY_ORDER = {
        "NONE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    # Configurable risk-based repeat intervals (seconds)
    RISK_INTERVALS = {
        "NO_WARNING": 6.0,
        "CAUTION": 3.0,
        "WARNING": 1.8,
        "CRITICAL": 0.9,
        "UNKNOWN": 6.0,
    }

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        audio_cfg = self.config.get("audio", {})
        templates_cfg = self.config.get("warning_messages", {})

        self.audio_mode = audio_cfg.get("mode", "CONTINUOUS_RISK")  # CONTINUOUS_RISK or TRANSITIONS_ONLY
        self.caution_template = templates_cfg.get("caution", "Caution. Obstacle nearby.")
        self.warning_template = templates_cfg.get("warning", "Warning. Obstacle ahead.")
        self.critical_template = templates_cfg.get("critical", "Critical. Immediate obstacle ahead. Stop.")

        # Tracking state for deduplication and pacing
        self.last_spoken_text: str = ""
        self.last_spoken_time: Optional[float] = None
        self.last_spoken_wall_time: Optional[float] = None
        self.last_spoken_state: str = "NO_WARNING"
        self.last_spoken_nav: str = "CONTINUE"
        self.last_spoken_track_id: Optional[int] = None
        self.last_spoken_priority: str = "NONE"
        self.last_spoken_ttc: Optional[float] = None

    def set_audio_mode(self, mode: str) -> None:
        """Dynamically set audio guidance mode ('CONTINUOUS_RISK' vs 'TRANSITIONS_ONLY')."""
        self.audio_mode = mode.upper()

    def get_seconds_to_next_update(self) -> float:
        """Returns seconds remaining until next periodic audio update."""
        if self.last_spoken_wall_time is None:
            return 0.0
        interval = self.RISK_INTERVALS.get(self.last_spoken_state, 5.0)
        elapsed = time.time() - self.last_spoken_wall_time
        return max(0.0, interval - elapsed)

    def generate(
        self,
        global_warning: GlobalWarningDecision,
        track_decisions: Optional[Dict[int, WarningDecision]] = None,
        current_time: Optional[float] = None,
        scene_nav=None,
    ) -> WarningMessage:
        """
        Generate a concise, explainable warning message based on global decision and scene navigation.
        """
        current_wall_time = time.time()
        if current_time is None:
            current_time = current_wall_time

        track_decisions = track_decisions or {}
        state = global_warning.state
        tid = global_warning.selected_track_id
        target_dec = track_decisions.get(tid) if (tid is not None) else None

        priority = self.STATE_PRIORITY_MAP.get(state, "NONE")
        nav_state = getattr(scene_nav, "navigation_state", "CONTINUE") if scene_nav else "CONTINUE"

        # 1. Formulate natural wording based on risk escalation, de-escalation, or current state
        raw_text, is_deescalation = self._build_natural_text(state, target_dec, scene_nav=scene_nav)

        # 2. Evaluate pacing, state-change triggers, TTC shifts, and deduplication
        should_speak = self._evaluate_should_speak(
            text=raw_text,
            state=state,
            priority=priority,
            track_id=tid,
            nav_state=nav_state,
            is_deescalation=is_deescalation,
            target_dec=target_dec,
            current_time=current_time,
            current_wall_time=current_wall_time,
        )

        if should_speak:
            self.last_spoken_text = raw_text
            self.last_spoken_time = current_time
            self.last_spoken_wall_time = current_wall_time
            self.last_spoken_state = state
            self.last_spoken_nav = nav_state
            self.last_spoken_track_id = tid
            self.last_spoken_priority = priority
            self.last_spoken_ttc = target_dec.ttc_seconds if (target_dec and target_dec.ttc_seconds is not None) else None

        return WarningMessage(
            text=raw_text if should_speak else "",
            priority=priority,
            state=state,
            track_id=tid,
            reason=global_warning.reason,
            should_speak=should_speak,
            timestamp=current_time,
        )

    def _build_natural_text(
        self,
        state: str,
        target: Optional[WarningDecision],
        scene_nav=None,
    ) -> Tuple[str, bool]:
        """
        Builds concise natural language warning text.
        Returns Tuple[text, is_deescalation_flag].
        """
        nav_state = getattr(scene_nav, "navigation_state", "CONTINUE") if scene_nav else "CONTINUE"
        curr_p_order = self.PRIORITY_ORDER.get(self.STATE_PRIORITY_MAP.get(state, "NONE"), 0)
        last_p_order = self.PRIORITY_ORDER.get(self.last_spoken_priority, 0)

        is_deescalation = (curr_p_order < last_p_order)

        # De-escalation Phrasing
        if is_deescalation:
            if state == "WARNING":
                return "Risk decreasing. Still approaching.", True
            elif state == "CAUTION":
                return "Risk decreasing.", True
            elif state in ("NO_WARNING", "UNKNOWN"):
                return "Risk cleared. Path clear.", True

        # Emergency STOP takes top precedence
        if nav_state == "STOP" or state == "CRITICAL":
            ttc_phrase = ""
            if target and target.ttc_seconds is not None and target.ttc_seconds > 0.0:
                ttc_phrase = f" TTC {target.ttc_seconds:.1f} seconds."
            if target and target.class_name:
                cls_str = target.class_name.capitalize()
                return f"Critical. {cls_str} approaching.{ttc_phrase} Stop.", False
            return "Critical. Stop.", False

        # Directional Lateral Avoidance Guidance
        if nav_state == "AVOID_LEFT":
            return "Obstacle ahead. Move left.", False

        if nav_state == "AVOID_RIGHT":
            return "Obstacle ahead. Move right.", False

        # Clear Path
        if state in ("NO_WARNING", "UNKNOWN") or target is None:
            return "Path clear.", False

        # Identify target object class label
        cls_name = target.class_name.lower().strip()
        if cls_name in self.VEHICLE_CLASSES:
            obj_label = "vehicle"
        elif cls_name in ("person", "bicycle", "chair", "table", "bed"):
            obj_label = cls_name
        else:
            obj_label = "obstacle"

        is_approaching = (target.approach_state == "APPROACHING")

        # TTC phrase formatting
        ttc_phrase = ""
        if is_approaching and target.ttc_seconds is not None and target.ttc_seconds > 0.0:
            ttc_phrase = f" TTC {target.ttc_seconds:.1f} seconds."

        # Structured Warning Format: [RISK] + [OBJECT] + [MOTION] + [TTC/ACTION]
        if state == "WARNING":
            if is_approaching:
                return f"Warning. {obj_label.capitalize()} approaching.{ttc_phrase}", False
            return f"Warning. {obj_label.capitalize()} ahead.", False

        if state == "CAUTION":
            if obj_label != "obstacle":
                return f"Caution. {obj_label.capitalize()} nearby.", False
            return "Caution. Obstacle nearby.", False

        return "Path clear.", False

    def _evaluate_should_speak(
        self,
        text: str,
        state: str,
        priority: str,
        track_id: Optional[int],
        nav_state: str,
        is_deescalation: bool,
        target_dec: Optional[WarningDecision],
        current_time: float,
        current_wall_time: float,
    ) -> bool:
        """
        Determines whether speech output should be dispatched based on risk state,
        transitions, TTC changes, deduplication, and audio mode.
        """
        if not text:
            return False

        # First message ever
        if not self.last_spoken_text:
            return True

        curr_p_order = self.PRIORITY_ORDER.get(priority, 0)
        last_p_order = self.PRIORITY_ORDER.get(self.last_spoken_priority, 0)

        # 1. Immediate Escalation: state increase triggers immediate speech
        if curr_p_order > last_p_order:
            return True

        # 2. Immediate De-escalation: state decrease triggers immediate speech
        if is_deescalation:
            return True

        # 3. Navigation Direction Shift (e.g. from CONTINUE to AVOID_LEFT or STOP)
        if nav_state != self.last_spoken_nav and nav_state != "CONTINUE":
            return True

        # 4. Significant TTC Change Trigger (e.g., TTC closed by >0.5s)
        if target_dec and target_dec.ttc_seconds is not None and self.last_spoken_ttc is not None:
            ttc_diff = abs(target_dec.ttc_seconds - self.last_spoken_ttc)
            if ttc_diff >= 0.5 and target_dec.approach_state == "APPROACHING":
                return True

        # If mode is TRANSITIONS_ONLY, do not send periodic updates
        if self.audio_mode == "TRANSITIONS_ONLY":
            return False

        # 5. Continuous Risk Pacing (Periodic updates based on risk state)
        wall_elapsed = (current_wall_time - self.last_spoken_wall_time) if (self.last_spoken_wall_time is not None) else 999.0
        stream_elapsed = (current_time - self.last_spoken_time) if (self.last_spoken_time is not None) else 999.0
        elapsed = max(wall_elapsed, stream_elapsed)

        interval = self.RISK_INTERVALS.get(state, 5.0)

        # Dispatch periodic update if required interval has elapsed
        if elapsed >= interval:
            return True

        return False
