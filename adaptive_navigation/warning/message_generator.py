"""
User-Facing Warning Message Generator.

Converts internal GlobalWarningDecision into natural, concise, and explainable
speech messages with temporal deduplication and priority gating.

CRITICAL RULES:
- Never say "left" or "right" unless a validated navigation direction module exists.
  Use "ahead", "nearby", or "approaching".
- Never convert relative depth into meters.
- Never speak uncalibrated TTC seconds unless is_metric is explicitly True.
- Suppress repetitive identical warnings using configurable repeat_interval_seconds.
- Immediately speak when state escalates (e.g. CAUTION -> WARNING -> CRITICAL).
- Do not speak repetitive de-escalation chatter.
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
    Transforms stabilized GlobalWarningDecision and individual track decisions
    into prioritized, deduplicated natural language alert messages.
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
        self.caution_template = templates_cfg.get("caution", "Please be cautious.")
        self.warning_template = templates_cfg.get("warning", "Obstacle ahead.")
        self.critical_template = templates_cfg.get("critical", "Immediate obstacle ahead.")

        # Deduplication state
        self.last_spoken_text: str = ""
        self.last_spoken_time: float = 0.0
        self.last_spoken_state: str = "NO_WARNING"
        self.last_spoken_track_id: Optional[int] = None
        self.last_spoken_priority: str = "NONE"

    def generate(
        self,
        global_warning: GlobalWarningDecision,
        track_decisions: Optional[Dict[int, WarningDecision]] = None,
        current_time: Optional[float] = None,
    ) -> WarningMessage:
        """
        Generate a concise, explainable warning message based on global decision.
        """
        if current_time is None:
            current_time = time.time()

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

        # 2. Formulate natural wording based on obstacle context
        raw_text = self._build_natural_text(state, target_dec)

        # 3. Deduplication & Escalation Check
        should_speak = self._evaluate_should_speak(
            text=raw_text,
            state=state,
            priority=priority,
            track_id=tid,
            current_time=current_time,
        )

        if should_speak:
            self.last_spoken_text = raw_text
            self.last_spoken_time = current_time
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

    def _build_natural_text(self, state: str, target: Optional[WarningDecision]) -> str:
        """
        Builds concise natural language warning text without fabricating directions or metrics.
        """
        if target is None:
            if state == "CRITICAL":
                return self.critical_template
            elif state == "WARNING":
                return self.warning_template
            elif state == "CAUTION":
                return self.caution_template
            return ""

        cls_name = target.class_name.lower().strip()
        if cls_name in self.VEHICLE_CLASSES:
            obj_label = "vehicle"
        elif cls_name in ("person", "bicycle"):
            obj_label = cls_name
        else:
            obj_label = "obstacle"

        is_approaching = (target.approach_state == "APPROACHING")

        # TTC phrase (ONLY if metric, valid, and reliable)
        ttc_phrase = ""
        # Check if ttc is metric and valid (never use relative depth for seconds)
        if target.ttc_state == "VALID" and target.ttc_seconds is not None:
            rounded_sec = max(1, int(round(target.ttc_seconds)))
            sec_word = "one second" if rounded_sec == 1 else f"{rounded_sec} seconds"
            ttc_phrase = f", about {sec_word}"

        # Construct concise phrasing
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
    ) -> bool:
        """
        Determines whether speech output should be dispatched.
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

        # If identical text or same state: apply repeat interval timer
        elapsed = current_time - self.last_spoken_time
        if elapsed >= self.repeat_interval:
            return True

        return False
