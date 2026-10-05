"""
Safe Navigation Decision & Scene-Level Analysis Engine.

Combines spatial obstacle locations, path corridor overlap, free-space clearance,
and multi-factor warning states to generate safe, explainable navigation actions.

Navigation States:
- CONTINUE: Path is clear, safe to proceed forward.
- CAUTION: Moderate hazard or nearby obstacle under observation.
- STOP: Path blocked with high confidence and no validated safe alternative.
- AVOID_LEFT: Center path blocked, left side validated as clear and safe.
- AVOID_RIGHT: Center path blocked, right side validated as clear and safe.
- UNKNOWN: Insufficient spatial or reliability evidence (safety-first fallback).

CRITICAL CONSTRAINTS:
- Never generate AVOID_LEFT or AVOID_RIGHT without validating BOTH sides.
- Never guess a direction when evidence is ambiguous or low reliability (< 0.70).
- Temporal persistence prevents rapid oscillation between directions.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from navigation.spatial import SpatialObjectRepresentation
from navigation.path_geometry import PathOverlapAssessment
from warning.state_machine import WarningDecision, GlobalWarningDecision


@dataclass
class NavigationDecision:
    """Per-track navigation relevance evaluation."""
    track_id: int
    spatial_zone: str                   # 'LEFT', 'CENTER', 'RIGHT'
    path_overlap_state: str             # 'OUTSIDE_PATH', 'PARTIAL_PATH_OVERLAP', 'INSIDE_PATH', 'UNKNOWN'
    path_relevance: float
    is_path_blocking: bool
    warning_state: str
    confidence: float
    reason: str


@dataclass
class SceneNavigationState:
    """Global scene-level navigation action decision."""
    navigation_state: str               # 'CONTINUE', 'CAUTION', 'STOP', 'AVOID_LEFT', 'AVOID_RIGHT', 'UNKNOWN'
    blocking_tracks: List[int]
    critical_tracks: List[int]
    path_blocked: bool
    left_available: bool
    center_available: bool
    right_available: bool
    left_free_space: float              # [0.0, 1.0]
    center_free_space: float            # [0.0, 1.0]
    right_free_space: float             # [0.0, 1.0]
    selected_threat: Optional[int]
    safe_direction: str                 # 'NONE', 'LEFT', 'RIGHT', 'UNKNOWN'
    reliability: float
    reason: str


class NavigationEngine:
    """
    Evaluates scene-level obstacle occupancy, analyzes lateral free-space,
    and applies temporal persistence and hysteresis to produce safe navigation actions.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        nav_cfg = self.config.get("navigation", {})

        self.enabled = bool(nav_cfg.get("enabled", True))
        self.min_reliability = float(nav_cfg.get("minimum_reliability", 0.70))

        pers_cfg = nav_cfg.get("persistence", {})
        self.min_persistence_frames = int(pers_cfg.get("minimum_frames", 3))
        self.dir_change_frames = int(pers_cfg.get("direction_change_frames", 3))

        # Temporal stabilization state
        self.current_state: str = "CONTINUE"
        self.candidate_state: str = "CONTINUE"
        self.candidate_counter: int = 0
        self.dir_switch_counter: int = 0

    def evaluate(
        self,
        spatial_objects: Dict[int, SpatialObjectRepresentation],
        path_assessments: Dict[int, PathOverlapAssessment],
        warning_decisions: Dict[int, WarningDecision],
        global_warning: GlobalWarningDecision,
        system_reliability_score: float = 1.0,
    ) -> Tuple[Dict[int, NavigationDecision], SceneNavigationState]:
        """
        Execute scene-level navigation evaluation and return per-track decisions
        alongside stabilized global SceneNavigationState.
        """
        # 1. Per-track navigation analysis
        per_track_decisions: Dict[int, NavigationDecision] = {}
        blocking_tracks: List[int] = []
        critical_tracks: List[int] = []

        # Free space clearance initialization [0.0 to 1.0, higher = more clear]
        left_occ = 0.0
        center_occ = 0.0
        right_occ = 0.0

        for tid, spatial in spatial_objects.items():
            p_ass = path_assessments.get(tid)
            w_dec = warning_decisions.get(tid)

            p_state = p_ass.overlap_state if p_ass else "UNKNOWN"
            p_rel = p_ass.path_relevance if p_ass else 0.0
            w_state = w_dec.state if w_dec else "NO_WARNING"
            r_score = w_dec.risk_score if w_dec else 0.0

            # Determine if this obstacle blocks the forward path
            is_blocking = (
                p_state in ("INSIDE_PATH", "PARTIAL_PATH_OVERLAP")
                and w_state in ("WARNING", "CRITICAL")
            )
            if is_blocking:
                blocking_tracks.append(tid)

            if w_state == "CRITICAL":
                critical_tracks.append(tid)

            # Accumulate lateral occupancy only for objects representing physical hazards or path obstructions
            is_spatial_hazard = (
                is_blocking
                or w_state in ("WARNING", "CRITICAL")
                or (p_state in ("INSIDE_PATH", "PARTIAL_PATH_OVERLAP") and r_score >= 0.50 and w_state != "NO_WARNING")
            )
            if is_spatial_hazard:
                occ_weight = min(1.0, max(0.2, (spatial.size_norm[0] * 2.0) + (spatial.center_norm[1] * 0.5)))
                if spatial.spatial_zone == "LEFT":
                    left_occ = min(1.0, left_occ + occ_weight)
                elif spatial.spatial_zone == "CENTER":
                    center_occ = min(1.0, center_occ + occ_weight)
                elif spatial.spatial_zone == "RIGHT":
                    right_occ = min(1.0, right_occ + occ_weight)

            reason_str = "Clear" if not is_blocking else f"Blocks path ({w_state})"
            per_track_decisions[tid] = NavigationDecision(
                track_id=tid,
                spatial_zone=spatial.spatial_zone,
                path_overlap_state=p_state,
                path_relevance=p_rel,
                is_path_blocking=is_blocking,
                warning_state=w_state,
                confidence=r_score,
                reason=reason_str,
            )

        left_free = float(max(0.0, 1.0 - left_occ))
        center_free = float(max(0.0, 1.0 - center_occ))
        right_free = float(max(0.0, 1.0 - right_occ))

        left_avail = (left_free >= 0.45)
        center_avail = (center_free >= 0.40 and len(blocking_tracks) == 0)
        right_avail = (right_free >= 0.45)

        path_blocked = (len(blocking_tracks) > 0 or (center_free < 0.30 and (len(critical_tracks) > 0 or any(w.state in ("WARNING", "CRITICAL") for w in warning_decisions.values()))))

        # 2. Determine raw candidate navigation state
        raw_state, safe_dir, raw_reason = self._determine_raw_state(
            path_blocked=path_blocked,
            blocking_tracks=blocking_tracks,
            critical_tracks=critical_tracks,
            left_avail=left_avail,
            right_avail=right_avail,
            left_free=left_free,
            right_free=right_free,
            global_warning_state=global_warning.state,
            reliability=system_reliability_score,
        )

        # 3. Apply temporal stabilization & direction change hysteresis
        stabilized_state = self._apply_persistence(raw_state)

        # Re-derive safe direction and reason aligned with stabilized state
        final_safe_dir, final_reason = self._derive_final_guidance(
            stabilized_state, left_avail, right_avail, raw_reason
        )

        scene_state = SceneNavigationState(
            navigation_state=stabilized_state,
            blocking_tracks=blocking_tracks,
            critical_tracks=critical_tracks,
            path_blocked=path_blocked,
            left_available=left_avail,
            center_available=center_avail,
            right_available=right_avail,
            left_free_space=left_free,
            center_free_space=center_free,
            right_free_space=right_free,
            selected_threat=global_warning.selected_track_id,
            safe_direction=final_safe_dir,
            reliability=system_reliability_score,
            reason=final_reason,
        )

        return per_track_decisions, scene_state

    def _determine_raw_state(
        self,
        path_blocked: bool,
        blocking_tracks: List[int],
        critical_tracks: List[int],
        left_avail: bool,
        right_avail: bool,
        left_free: float,
        right_free: float,
        global_warning_state: str,
        reliability: float,
    ) -> Tuple[str, str, str]:
        """
        Calculates raw target navigation state before temporal filtering.
        """
        # Safety-First Gating: low reliability forces UNKNOWN
        if reliability < self.min_reliability:
            return "UNKNOWN", "UNKNOWN", "INSUFFICIENT_SPATIAL_EVIDENCE"

        if global_warning_state == "UNKNOWN":
            return "UNKNOWN", "UNKNOWN", "INSUFFICIENT_RELIABLE_EVIDENCE"

        # Case 1: Forward path is clear
        if not path_blocked and len(critical_tracks) == 0:
            if global_warning_state == "CAUTION":
                return "CAUTION", "NONE", "CENTER_PATH_CLEAR_CAUTION_NEARBY"
            return "CONTINUE", "NONE", "CENTER_PATH_CLEAR"

        # Case 2: Forward path is blocked
        # Safe-side analysis: Evaluate both lateral sides
        if left_avail and not right_avail:
            return "AVOID_LEFT", "LEFT", "CENTER_PATH_BLOCKED_LEFT_AVAILABLE"

        if right_avail and not left_avail:
            return "AVOID_RIGHT", "RIGHT", "CENTER_PATH_BLOCKED_RIGHT_AVAILABLE"

        if left_avail and right_avail:
            # Both clear: select side with strictly greater free space
            if left_free >= right_free:
                return "AVOID_LEFT", "LEFT", "CENTER_PATH_BLOCKED_BOTH_SIDES_AVAILABLE"
            else:
                return "AVOID_RIGHT", "RIGHT", "CENTER_PATH_BLOCKED_BOTH_SIDES_AVAILABLE"

        # Both sides blocked: cannot steer safely
        if len(critical_tracks) > 0 or global_warning_state == "CRITICAL":
            return "STOP", "NONE", "CENTER_PATH_BLOCKED_BOTH_SIDES_BLOCKED"

        return "STOP", "NONE", "CENTER_PATH_BLOCKED_NO_SAFE_ALTERNATIVE"

    def _apply_persistence(self, raw_state: str) -> str:
        """
        Prevents rapid state jumping using persistence counters and direction hysteresis.
        """
        # Safety-First Fallback: low reliability or missing evidence forces immediate UNKNOWN
        if raw_state == "UNKNOWN":
            self.current_state = "UNKNOWN"
            self.candidate_state = "UNKNOWN"
            self.candidate_counter = 0
            self.dir_switch_counter = 0
            return self.current_state

        # 1. Immediate emergency STOP escalation
        if raw_state == "STOP" and self.current_state != "STOP":
            self.candidate_state = "STOP"
            self.candidate_counter += 1
            if self.candidate_counter >= 2:  # Faster escalation for STOP
                self.current_state = "STOP"
                self.candidate_counter = 0
            return self.current_state

        # 2. Direction switching hysteresis (AVOID_LEFT <-> AVOID_RIGHT)
        is_dir_switch = (
            (self.current_state == "AVOID_LEFT" and raw_state == "AVOID_RIGHT")
            or (self.current_state == "AVOID_RIGHT" and raw_state == "AVOID_LEFT")
        )
        if is_dir_switch:
            if self.candidate_state == raw_state:
                self.dir_switch_counter += 1
            else:
                self.candidate_state = raw_state
                self.dir_switch_counter = 1

            if self.dir_switch_counter >= self.dir_change_frames:
                self.current_state = raw_state
                self.dir_switch_counter = 0
            return self.current_state

        # 3. Standard state persistence
        if raw_state == self.current_state:
            self.candidate_state = raw_state
            self.candidate_counter = 0
            self.dir_switch_counter = 0
            return self.current_state

        if raw_state == self.candidate_state:
            self.candidate_counter += 1
        else:
            self.candidate_state = raw_state
            self.candidate_counter = 1

        if self.candidate_counter >= self.min_persistence_frames:
            self.current_state = raw_state
            self.candidate_counter = 0
            self.dir_switch_counter = 0

        return self.current_state

    def _derive_final_guidance(
        self,
        state: str,
        left_avail: bool,
        right_avail: bool,
        fallback_reason: str,
    ) -> Tuple[str, str]:
        """Aligns direction and explainable reason with stabilized state."""
        if state == "CONTINUE":
            return "NONE", "CENTER_PATH_CLEAR"
        elif state == "CAUTION":
            return "NONE", "CAUTION_MONITOR_SURROUNDINGS"
        elif state == "STOP":
            return "NONE", "BOTH_SIDES_BLOCKED" if (not left_avail and not right_avail) else "PATH_BLOCKED_STOP"
        elif state == "AVOID_LEFT":
            return "LEFT", "LEFT_SIDE_AVAILABLE"
        elif state == "AVOID_RIGHT":
            return "RIGHT", "RIGHT_SIDE_AVAILABLE"
        else:
            return "UNKNOWN", fallback_reason
