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

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        audio_cfg = self.config.get("audio", {})
        templates_cfg = self.config.get("warning_messages", {})

        self.repeat_interval = float(audio_cfg.get("repeat_interval_seconds", 2.0))
        self.continuous_guidance = bool(audio_cfg.get("continuous_guidance", True))
        self.caution_template = templates_cfg.get("caution", "Please be cautious.")
        self.warning_template = templates_cfg.get("warning", "Obstacle ahead.")
        self.critical_template = templates_cfg.get("critical", "Immediate obstacle ahead.")

        # Deduplication and pacing state
        self.last_spoken_text: str = ""
        self.last_spoken_time: Optional[float] = None
        self.last_spoken_wall_time: Optional[float] = None
        self.last_spoken_state: str = "NO_WARNING"
        self.last_spoken_track_id: Optional[int] = None
        self.last_spoken_priority: str = "NONE"

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

        # 1. If NO_WARNING or UNKNOWN: silent
        if state in ("NO_WARNING", "UNKNOWN") or not state:
            return WarningMessage(
                text="",
                priority=priority,
                state=state,
                track_id=tid,
                reason=global_warning.reason,
                should_speak=False,
                timestamp=current_time,
            )

        # 2. Formulate natural wording based on obstacle context & navigation guidance
        raw_text = self._build_natural_text(state, target_dec, scene_nav=scene_nav)

        # 3. Deduplication & Escalation Check (wall-clock aware for physical audio output)
        should_speak = self._evaluate_should_speak(
            text=raw_text,
            state=state,
            priority=priority,
            track_id=tid,
            current_time=current_time,
            current_wall_time=current_wall_time,
        )

        if should_speak:
            self.last_spoken_text = raw_text
            self.last_spoken_time = current_time
            self.last_spoken_wall_time = current_wall_time
            self.last_spoken_state = state
            self.last_spoken_track_id = tid
            self.last_spoken_priority = priority

        return WarningMessage(
            text=raw_text,
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
    ) -> str:
        """
        Builds concise natural language warning text incorporating directional navigation.
        """
        nav_state = getattr(scene_nav, "navigation_state", "UNKNOWN") if scene_nav else "UNKNOWN"

        # Emergency STOP takes top precedence
        if nav_state == "STOP":
            if state == "CRITICAL":
                return "Immediate obstacle ahead. Stop."
            return "Stop."

        # Identify target obstacle label
        if target is not None:
            cls_name = target.class_name.lower().strip()
            if cls_name in self.VEHICLE_CLASSES:
                obj_label = "vehicle"
            elif cls_name in ("person", "bicycle"):
                obj_label = cls_name
            else:
                obj_label = "obstacle"
            is_approaching = (target.approach_state == "APPROACHING")
        else:
            obj_label = "obstacle"
            is_approaching = False

        # Step 14 Lateral Avoidance Guidance
        if nav_state == "AVOID_LEFT":
            if is_approaching:
                return f"{obj_label.capitalize()} approaching ahead. Move left."
            return f"Obstacle ahead. Move left."

        if nav_state == "AVOID_RIGHT":
            if is_approaching:
                return f"{obj_label.capitalize()} approaching ahead. Move right."
            return f"Obstacle ahead. Move right."

        if target is None:
            if state == "CRITICAL":
                return self.critical_template
            elif state == "WARNING":
                return self.warning_template
            elif state == "CAUTION":
                return self.caution_template
            return ""

        # TTC phrase (ONLY if metric, valid, and reliable)
        ttc_phrase = ""
        if target.ttc_state == "VALID" and target.ttc_seconds is not None:
            rounded_sec = max(1, int(round(target.ttc_seconds)))
            sec_word = "one second" if rounded_sec == 1 else f"{rounded_sec} seconds"
            ttc_phrase = f", about {sec_word}"

        # Standard Obstacle Alert Phrasing
        if state == "CRITICAL":
            if is_approaching:
                return f"Immediate {obj_label} approaching{ttc_phrase}."
            return f"Immediate {obj_label} ahead."

        if state == "WARNING":
            if is_approaching:
                return f"{obj_label.capitalize()} approaching ahead{ttc_phrase}."
            return f"{obj_label.capitalize()} ahead."

        if state == "CAUTION":
            if obj_label != "obstacle":
                return f"Caution, {obj_label} nearby."
            return self.caution_template

        return ""

    def _evaluate_should_speak(
        self,
        text: str,
        state: str,
        priority: str,
        track_id: Optional[int],
        current_time: float,
        current_wall_time: float,
    ) -> bool:
        """
        Determines whether speech output should be dispatched.
        Synchronizes with physical audio hardware using wall-clock time
        so users receive continuous, properly paced audible guidance.
        """
        if not text:
            return False

        # First spoken message
        if not self.last_spoken_text:
            return True

        curr_p_order = self.PRIORITY_ORDER.get(priority, 0)
        last_p_order = self.PRIORITY_ORDER.get(self.last_spoken_priority, 0)

        # De-escalation rule: do not announce de-escalation chatter
        if curr_p_order < last_p_order:
            return False

        # Immediate Escalation rule: higher priority always speaks immediately
        if curr_p_order > last_p_order:
            return True

        # New threat with same high priority
        if track_id is not None and self.last_spoken_track_id != track_id and curr_p_order >= 3:
            return True

        # Timing elapsed:
        # In real-world operation, speech delivery is experienced by the user in wall-clock time.
        # Check both wall-clock elapsed and stream elapsed to support slow inference (CPU 0.2 FPS)
        # as well as fast real-time playback.
        wall_elapsed = (current_wall_time - self.last_spoken_wall_time) if (self.last_spoken_wall_time is not None) else 999.0
        stream_elapsed = (current_time - self.last_spoken_time) if (self.last_spoken_time is not None) else 999.0
        elapsed = max(wall_elapsed, stream_elapsed)

        # If guidance instruction changed (e.g., from "Move left" to "Stop"):
        # Allow immediate update after a short conversational pause (1.0s)
        if text != self.last_spoken_text and elapsed >= 1.0:
            return True

        # Continuous Guidance: if the threat persists (WARNING, CRITICAL, or persistent CAUTION),
        # provide continuous periodic voice guidance once repeat_interval has elapsed.
        if elapsed >= self.repeat_interval:
            return True

        return False
