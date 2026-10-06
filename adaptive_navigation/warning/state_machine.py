"""
Temporal Risk Stabilization & Warning Decision State Machine.

Maintains temporal risk history, persistence counters, hysteresis thresholds,
and reliability gating to prevent spurious alerts and abrupt alert drops.

Warning States:
- NO_WARNING: No immediate collision concern.
- CAUTION: Possible concern, monitor situation.
- WARNING: Persistent collision concern with verified reliability.
- CRITICAL: Immediate, high-confidence threat requiring immediate avoidance.
- UNKNOWN: Insufficient reliable evidence (distinct from NO_WARNING).
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence, Tuple

from risk.risk_engine import RiskAssessment
from uncertainty.reliability import ReliabilityAssessment
from risk.ttc import TTCResult
from temporal.camera_motion import CompensatedMotionEstimate


@dataclass
class RiskHistoryEntry:
    """Historical observation snapshot for a tracked obstacle."""
    timestamp: float
    frame_index: int
    risk_score: float
    risk_level: str
    reliability_score: float
    reliability_level: str
    ttc_seconds: Optional[float]
    ttc_state: str
    approach_state: str
    path_relevance: float


@dataclass
class WarningDecision:
    """Stabilized, explainable warning decision for an individual obstacle."""
    track_id: int
    class_name: str
    state: str                          # NO_WARNING, CAUTION, WARNING, CRITICAL, UNKNOWN
    risk_score: float
    risk_level: str
    reliability_score: float
    reliability_level: str
    persistence_count: int
    state_duration: float               # Duration in seconds spent in current state
    ttc_seconds: Optional[float]
    ttc_state: str
    path_relevance: float
    approach_state: str
    primary_reason: str
    decision_valid: bool = True


@dataclass
class GlobalWarningDecision:
    """Highest-priority warning decision selected across all tracked obstacles."""
    selected_track_id: Optional[int]
    state: str                          # NO_WARNING, CAUTION, WARNING, CRITICAL, UNKNOWN
    priority_score: float
    reason: str
    competing_tracks: List[int] = field(default_factory=list)
    active_warning_count: int = 0


class TrackWarningTracker:
    """
    Manages temporal risk history, persistence counters, and hysteresis
    for an individual tracked obstacle.
    """

    STATE_ORDER = {"NO_WARNING": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3, "UNKNOWN": -1}

    def __init__(self, track_id: int, config: dict) -> None:
        self.track_id = track_id
        self.config = config

        self.history_length = int(config.get("history_length", 10))
        self.history: deque[RiskHistoryEntry] = deque(maxlen=self.history_length)

        # State tracking
        self.current_state = "NO_WARNING"
        self.state_start_time: float = 0.0
        self.last_update_time: float = 0.0
        self.last_frame_index: int = 0

        # Persistence counters
        self.candidate_state = "NO_WARNING"
        self.candidate_persistence: int = 0
        self.deescalation_counter: int = 0

        # Grace period for missing track observations
        self.is_missing: bool = False
        self.missing_since: Optional[float] = None

    def update(
        self,
        risk: RiskAssessment,
        reliability: ReliabilityAssessment,
        ttc: Optional[TTCResult] = None,
        motion: Optional[CompensatedMotionEstimate] = None,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> WarningDecision:
        """
        Ingest current frame risk and reliability metrics and return stabilized warning decision.
        """
        self.is_missing = False
        self.missing_since = None
        self.last_update_time = timestamp
        self.last_frame_index = frame_index

        if self.state_start_time <= 0.0:
            self.state_start_time = timestamp

        ttc_sec = ttc.ttc_seconds if (ttc and ttc.ttc_valid) else None
        ttc_st = ttc.ttc_state if ttc else "UNKNOWN"
        app_st = motion.approach_state if (motion and motion.motion_valid) else "UNKNOWN"
        path_rel = risk.path_relevance

        # Record rolling history
        entry = RiskHistoryEntry(
            timestamp=timestamp,
            frame_index=frame_index,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            reliability_score=reliability.reliability_score,
            reliability_level=reliability.reliability_level,
            ttc_seconds=ttc_sec,
            ttc_state=ttc_st,
            approach_state=app_st,
            path_relevance=path_rel,
        )
        self.history.append(entry)

        # 1. Determine raw target candidate state from current evidence
        target_candidate = self._determine_candidate_state(risk, reliability, ttc, motion)

        # 2. Check for fast escalation condition (e.g. short TTC + high reliability + in path)
        allow_fast_escalation = self._check_fast_escalation(risk, reliability, ttc, motion)

        # 3. Apply persistence logic
        self._apply_persistence(target_candidate, timestamp, allow_fast_escalation)

        # 4. Generate explainable reason
        reason = self._generate_reason(risk, reliability, ttc, motion)

        state_duration = max(0.0, timestamp - self.state_start_time)

        return WarningDecision(
            track_id=self.track_id,
            class_name=risk.class_name,
            state=self.current_state,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            reliability_score=reliability.reliability_score,
            reliability_level=reliability.reliability_level,
            persistence_count=self.candidate_persistence,
            state_duration=state_duration,
            ttc_seconds=ttc_sec,
            ttc_state=ttc_st,
            path_relevance=path_rel,
            approach_state=app_st,
            primary_reason=reason,
            decision_valid=True,
        )

    def handle_missing_frame(self, current_time: float) -> Optional[WarningDecision]:
        """
        Handle a frame where this track was not observed.
        Returns WarningDecision during grace period, or None if expired.
        """
        grace_sec = float(self.config.get("lost_track_grace_seconds", 0.5))
        if self.missing_since is None:
            self.missing_since = self.last_update_time if self.last_update_time > 0 else current_time
            self.is_missing = True

        elapsed_missing = current_time - self.missing_since
        if elapsed_missing <= grace_sec:
            # Retain current warning state during grace period
            last_entry = self.history[-1] if self.history else None
            return WarningDecision(
                track_id=self.track_id,
                class_name="tracked_object",
                state=self.current_state,
                risk_score=last_entry.risk_score if last_entry else 0.0,
                risk_level=last_entry.risk_level if last_entry else "UNKNOWN",
                reliability_score=last_entry.reliability_score if last_entry else 0.0,
                reliability_level=last_entry.reliability_level if last_entry else "UNKNOWN",
                persistence_count=0,
                state_duration=max(0.0, current_time - self.state_start_time),
                ttc_seconds=last_entry.ttc_seconds if last_entry else None,
                ttc_state="LOST_GRACE",
                path_relevance=last_entry.path_relevance if last_entry else 0.0,
                approach_state="LOST_GRACE",
                primary_reason=f"Track temporarily lost (grace {elapsed_missing:.2f}s/{grace_sec}s)",
                decision_valid=True,
            )
        return None  # Grace period expired

    def _determine_candidate_state(
        self,
        risk: RiskAssessment,
        reliability: ReliabilityAssessment,
        ttc: Optional[TTCResult],
        motion: Optional[CompensatedMotionEstimate],
    ) -> str:
        """
        Calculates raw candidate state taking into account hysteresis thresholds and reliability gates.
        """
        # A. Reliability Gating
        min_rel_warning = float(self.config.get("minimum_reliability_for_warning", 0.50))
        min_rel_critical = float(self.config.get("minimum_reliability_for_critical", 0.75))

        # Check for fundamentally unknown evidence
        if (
            risk.evidence_coverage < 0.20
            or reliability.reliability_level == "UNKNOWN"
            or reliability.reliability_score < 0.20
        ):
            return "UNKNOWN"

        score = risk.risk_score
        rel_score = reliability.reliability_score
        hyst = self.config.get("hysteresis", {})
        w_enter = float(hyst.get("warning_enter", 0.70))
        w_exit = float(hyst.get("warning_exit", 0.55))
        c_enter = float(hyst.get("critical_enter", 0.85))
        c_exit = float(hyst.get("critical_exit", 0.70))

        # B. Hysteresis-aware state targeting
        if self.current_state == "CRITICAL":
            if score >= c_exit and rel_score >= min_rel_critical:
                return "CRITICAL"
            elif score >= w_exit and rel_score >= min_rel_warning:
                return "WARNING"
            elif score >= 0.30:
                return "CAUTION"
            else:
                return "NO_WARNING"

        elif self.current_state == "WARNING":
            if score >= c_enter and rel_score >= min_rel_critical:
                return "CRITICAL"
            elif score >= w_exit and rel_score >= min_rel_warning:
                return "WARNING"
            elif score >= 0.30:
                return "CAUTION"
            else:
                return "NO_WARNING"

        elif self.current_state == "CAUTION":
            if score >= c_enter and rel_score >= min_rel_critical:
                return "CRITICAL"
            elif score >= w_enter and rel_score >= min_rel_warning:
                return "WARNING"
            elif score >= 0.45:
                return "CAUTION"
            else:
                return "NO_WARNING"

        else:  # NO_WARNING or UNKNOWN
            if score >= c_enter and rel_score >= min_rel_critical:
                return "CRITICAL"
            elif score >= w_enter and rel_score >= min_rel_warning:
                return "WARNING"
            elif score >= 0.50:
                return "CAUTION"
            else:
                return "NO_WARNING"

    def _check_fast_escalation(
        self,
        risk: RiskAssessment,
        reliability: ReliabilityAssessment,
        ttc: Optional[TTCResult],
        motion: Optional[CompensatedMotionEstimate],
    ) -> bool:
        """
        Determines if imminent threat conditions justify reduced persistence for escalation.
        """
        if (
            ttc
            and ttc.ttc_valid
            and ttc.ttc_seconds is not None
            and ttc.ttc_seconds < 2.5
            and reliability.reliability_score >= 0.75
            and risk.path_relevance >= 0.60
            and motion
            and motion.approach_state == "APPROACHING"
        ):
            return True
        return False

    def _apply_persistence(self, target: str, timestamp: float, fast_escalation: bool = False) -> None:
        """
        Updates persistence counters and executes state transition only when required frames are met.
        """
        curr_order = self.STATE_ORDER.get(self.current_state, 0)
        target_order = self.STATE_ORDER.get(target, 0)

        esc_cfg = self.config.get("escalation_frames", {})
        deesc_cfg = self.config.get("deescalation_frames", {})

        req_esc_caution = int(esc_cfg.get("caution", 2))
        req_esc_warning = int(esc_cfg.get("warning", 3))
        req_esc_critical = int(esc_cfg.get("critical", 3))

        req_deesc_crit_to_warn = int(deesc_cfg.get("critical_to_warning", 2))
        req_deesc_warn_to_caut = int(deesc_cfg.get("warning_to_caution", 2))
        req_deesc_caut_to_none = int(deesc_cfg.get("caution_to_no_warning", 3))

        if fast_escalation:
            req_esc_warning = max(1, req_esc_warning - 1)
            req_esc_critical = max(1, req_esc_critical - 1)

        # Handling UNKNOWN transitions
        if target == "UNKNOWN":
            if self.candidate_state == "UNKNOWN":
                self.candidate_persistence += 1
            else:
                self.candidate_state = "UNKNOWN"
                self.candidate_persistence = 1

            if self.candidate_persistence >= 3:
                self._transition_to("UNKNOWN", timestamp)
            return

        # ESCALATION (target higher than current)
        if target_order > curr_order:
            self.deescalation_counter = 0
            if self.candidate_state == target:
                self.candidate_persistence += 1
            else:
                self.candidate_state = target
                self.candidate_persistence = 1

            # Check threshold for target
            if target == "CRITICAL" and self.candidate_persistence >= req_esc_critical:
                self._transition_to("CRITICAL", timestamp)
            elif target == "WARNING" and self.candidate_persistence >= req_esc_warning:
                self._transition_to("WARNING", timestamp)
            elif target == "CAUTION" and self.candidate_persistence >= req_esc_caution:
                self._transition_to("CAUTION", timestamp)

        # DE-ESCALATION (target lower than current)
        elif target_order < curr_order:
            self.candidate_persistence = 0
            if self.candidate_state == target:
                self.deescalation_counter += 1
            else:
                self.candidate_state = target
                self.deescalation_counter = 1

            # Gradual step-down logic
            if self.current_state == "CRITICAL":
                if self.deescalation_counter >= req_deesc_crit_to_warn:
                    self._transition_to("WARNING", timestamp)
                    self.deescalation_counter = 0
            elif self.current_state == "WARNING":
                if self.deescalation_counter >= req_deesc_warn_to_caut:
                    self._transition_to("CAUTION", timestamp)
                    self.deescalation_counter = 0
            elif self.current_state == "CAUTION":
                if self.deescalation_counter >= req_deesc_caut_to_none:
                    self._transition_to("NO_WARNING", timestamp)
                    self.deescalation_counter = 0
            elif self.current_state == "UNKNOWN":
                if self.deescalation_counter >= 2:
                    self._transition_to(target, timestamp)
                    self.deescalation_counter = 0

        else:
            # target_order == curr_order: in equilibrium
            self.candidate_state = self.current_state
            self.candidate_persistence = max(self.candidate_persistence, 1)
            self.deescalation_counter = 0

    def _transition_to(self, new_state: str, timestamp: float) -> None:
        """Executes state transition and resets timers."""
        if self.current_state != new_state:
            self.current_state = new_state
            self.state_start_time = timestamp
            self.candidate_state = new_state
            self.candidate_persistence = 1
            self.deescalation_counter = 0

    def _generate_reason(
        self,
        risk: RiskAssessment,
        reliability: ReliabilityAssessment,
        ttc: Optional[TTCResult],
        motion: Optional[CompensatedMotionEstimate],
    ) -> str:
        """Constructs an explainable explanation for the warning decision."""
        if self.current_state == "CRITICAL":
            if ttc and ttc.ttc_valid and ttc.ttc_seconds is not None:
                return f"Imminent collision threat (TTC {ttc.ttc_seconds:.1f}s) in path"
            return "Persistent critical risk with high reliability"

        if self.current_state == "WARNING":
            if motion and motion.approach_state == "APPROACHING":
                return "Persistent approaching obstacle in trajectory"
            return "Persistent elevated risk verified across observations"

        if self.current_state == "CAUTION":
            if reliability.reliability_score < float(self.config.get("minimum_reliability_for_warning", 0.50)):
                return "Elevated risk detected but evidence uncertain (gated to CAUTION)"
            if risk.path_relevance < 0.40:
                return "Moderate risk outside central corridor"
            return "Moderate hazard under temporal observation"

        if self.current_state == "UNKNOWN":
            return "Insufficient reliable evidence to determine warning"

        return "No immediate collision concern"


class WarningStateMachine:
    """
    Coordinates per-track warning state machines and evaluates
    global threat priority across all active obstacles.
    """

    STATE_PRIORITY_BASE = {
        "CRITICAL": 4000.0,
        "WARNING": 2000.0,
        "CAUTION": 1000.0,
        "UNKNOWN": 100.0,
        "NO_WARNING": 0.0,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.trackers: Dict[int, TrackWarningTracker] = {}

    def reset(self) -> None:
        """Reset all per-track warning state trackers."""
        self.trackers.clear()

    def update(
        self,
        risk_assessments: Dict[int, RiskAssessment],
        reliability_assessments: Dict[int, ReliabilityAssessment],
        ttc_results: Optional[Dict[int, TTCResult]] = None,
        compensated_motion: Optional[Dict[int, CompensatedMotionEstimate]] = None,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> Tuple[Dict[int, WarningDecision], GlobalWarningDecision]:
        """
        Process current frame risk assessments, update all per-track warning states,
        and select the global highest-priority warning.
        """
        ttc_results = ttc_results or {}
        compensated_motion = compensated_motion or {}

        decisions: Dict[int, WarningDecision] = {}
        active_ids = set(risk_assessments.keys())
        all_tracker_ids = list(self.trackers.keys())

        # 1. Update active observed tracks
        for tid in active_ids:
            if tid not in self.trackers:
                self.trackers[tid] = TrackWarningTracker(tid, self.config)

            r_ass = risk_assessments[tid]
            rel_ass = reliability_assessments.get(
                tid,
                ReliabilityAssessment(
                    track_id=tid, class_name=r_ass.class_name,
                    detection_reliability=0.0, tracking_reliability=0.0,
                    depth_reliability=0.0, temporal_reliability=0.0,
                    motion_reliability=0.0, camera_motion_reliability=0.0,
                    ttc_reliability=0.0, evidence_reliability=0.0,
                    reliability_score=0.0, uncertainty_score=1.0,
                    reliability_level="UNKNOWN", tracking_state="UNKNOWN"
                ),
            )
            ttc_res = ttc_results.get(tid)
            m_res = compensated_motion.get(tid)

            decisions[tid] = self.trackers[tid].update(
                risk=r_ass,
                reliability=rel_ass,
                ttc=ttc_res,
                motion=m_res,
                timestamp=timestamp,
                frame_index=frame_index,
            )

        # 2. Check missing tracks for grace period retention
        stale_ids = []
        for tid in all_tracker_ids:
            if tid not in active_ids:
                dec = self.trackers[tid].handle_missing_frame(timestamp)
                if dec is not None:
                    decisions[tid] = dec
                else:
                    stale_ids.append(tid)

        # Clean up expired stale tracks
        for tid in stale_ids:
            del self.trackers[tid]

        # 3. Global Warning Priority Selection
        global_decision = self._select_global_warning(decisions)

        return decisions, global_decision

    def _select_global_warning(self, decisions: Dict[int, WarningDecision]) -> GlobalWarningDecision:
        """
        Selects the single highest-priority warning threat across all active obstacles.
        """
        if not decisions:
            return GlobalWarningDecision(
                selected_track_id=None,
                state="NO_WARNING",
                priority_score=0.0,
                reason="No active obstacles in field of view",
                competing_tracks=[],
                active_warning_count=0,
            )

        # Check if all decisions are UNKNOWN
        all_unknown = all(d.state == "UNKNOWN" for d in decisions.values())
        if all_unknown:
            return GlobalWarningDecision(
                selected_track_id=next(iter(decisions.keys())),
                state="UNKNOWN",
                priority_score=50.0,
                reason="Insufficient reliable evidence across all obstacles",
                competing_tracks=list(decisions.keys()),
                active_warning_count=0,
            )

        best_tid: Optional[int] = None
        best_score = -1.0
        best_decision: Optional[WarningDecision] = None
        warning_count = sum(1 for d in decisions.values() if d.state in ("CAUTION", "WARNING", "CRITICAL"))

        for tid, dec in decisions.items():
            base_p = self.STATE_PRIORITY_BASE.get(dec.state, 0.0)

            # TTC bonus: shorter TTC gives higher priority
            ttc_bonus = 0.0
            if dec.ttc_seconds is not None and dec.ttc_seconds > 0.0:
                ttc_bonus = max(0.0, (6.0 - dec.ttc_seconds) / 6.0) * 300.0

            # Composite tie-breaker score
            threat_score = (
                base_p
                + (dec.risk_score * 300.0)
                + (dec.reliability_score * 200.0)
                + (dec.path_relevance * 200.0)
                + ttc_bonus
            )

            if threat_score > best_score:
                best_score = threat_score
                best_tid = tid
                best_decision = dec

        if best_decision is None or best_decision.state == "NO_WARNING":
            return GlobalWarningDecision(
                selected_track_id=best_tid,
                state="NO_WARNING",
                priority_score=best_score,
                reason="No immediate hazard requiring avoidance",
                competing_tracks=list(decisions.keys()),
                active_warning_count=warning_count,
            )

        competing = [tid for tid, d in decisions.items() if tid != best_tid and d.state in ("WARNING", "CRITICAL", "CAUTION")]

        return GlobalWarningDecision(
            selected_track_id=best_tid,
            state=best_decision.state,
            priority_score=best_score,
            reason=best_decision.primary_reason,
            competing_tracks=competing,
            active_warning_count=warning_count,
        )
