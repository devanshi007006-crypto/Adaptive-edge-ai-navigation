"""
Risk-Aware Adaptive Computation Controller.

Dynamically adapts depth estimation inference frequency based on live risk level,
temporal approach motion, time-to-collision validity, track stability, and warning state.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class AdaptiveCadenceDecision:
    """Decision contract output by AdaptiveComputationController per frame."""
    should_run_depth: bool
    cadence_interval: int
    computation_mode: str
    reason: str
    depth_reused: bool
    risk_level: str


class AdaptiveComputationController:
    """
    Dynamic risk-aware computation controller for monocular edge-AI navigation.
    
    Modes:
    - HIGH_RISK_FULL_CADENCE (Cadence 1): Executes depth every frame for CRITICAL/WARNING threats.
    - MEDIUM_RISK_INTERLEAVED (Cadence 2): Executes depth every 2nd frame for CAUTION threats.
    - LOW_RISK_LIGHT_CADENCE (Cadence 4): Executes depth every 4th frame when path is CLEAR.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        adaptive_cfg = self.config.get("adaptive_computation", {})

        self.enabled = bool(adaptive_cfg.get("enabled", True))
        self.low_risk_interval = int(adaptive_cfg.get("low_risk_interval", 4))
        self.medium_risk_interval = int(adaptive_cfg.get("medium_risk_interval", 2))
        self.high_risk_interval = int(adaptive_cfg.get("high_risk_interval", 1))

        self.last_depth_frame_idx: int = -999
        self.last_mode: str = "INITIALIZING"

    def evaluate(
        self,
        frame_idx: int,
        global_warning_state: str = "NO_WARNING",
        max_risk_score: float = 0.0,
        has_approaching_objects: bool = False,
        object_count: int = 0,
    ) -> AdaptiveCadenceDecision:
        """Evaluates whether to execute full depth estimation or reuse/propagate depth on current frame."""
        if not self.enabled:
            # Fallback to standard 2:1 cadence
            should_run = (frame_idx % 2 == 0)
            return AdaptiveCadenceDecision(
                should_run_depth=should_run,
                cadence_interval=2,
                computation_mode="FIXED_2_1_CADENCE",
                reason="Adaptive computation disabled in configuration.",
                depth_reused=not should_run,
                risk_level=global_warning_state,
            )

        # 1. High Risk / Critical State -> Every Frame (Cadence 1)
        if global_warning_state in ("CRITICAL", "WARNING") or max_risk_score >= 0.75 or has_approaching_objects:
            mode = "HIGH_RISK_FULL_CADENCE"
            interval = self.high_risk_interval
            reason = f"Active threat detected (Risk: {global_warning_state}, MaxScore: {max_risk_score:.2f}). Running full depth every frame."

        # 2. Medium Risk / Caution State -> Interleaved (Cadence 2)
        elif global_warning_state == "CAUTION" or max_risk_score >= 0.35:
            mode = "MEDIUM_RISK_INTERLEAVED"
            interval = self.medium_risk_interval
            reason = f"Moderate risk detected (Risk: {global_warning_state}). Running 2:1 depth cadence."

        # 3. Low Risk / Path Clear -> Light Cadence (Cadence 4)
        else:
            mode = "LOW_RISK_LIGHT_CADENCE"
            interval = self.low_risk_interval
            reason = f"Path clear / low risk (Risk: {global_warning_state}). Running light 4:1 depth cadence."

        frames_since_last_depth = frame_idx - self.last_depth_frame_idx
        should_run = (frames_since_last_depth >= interval) or (self.last_depth_frame_idx < 0)

        if should_run:
            self.last_depth_frame_idx = frame_idx
            depth_reused = False
        else:
            depth_reused = True

        self.last_mode = mode

        return AdaptiveCadenceDecision(
            should_run_depth=should_run,
            cadence_interval=interval,
            computation_mode=mode,
            reason=reason,
            depth_reused=depth_reused,
            risk_level=global_warning_state,
        )
