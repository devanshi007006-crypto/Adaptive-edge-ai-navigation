"""
Time-to-Collision (TTC) Estimation Module.

Calculates estimated Time-to-Collision (TTC) for tracked obstacles based on:
- Distance / relative depth
- Camera-compensated closing motion
- Verified depth conventions

CRITICAL SCIENTIFIC & SAFETY PRINCIPLES:
- TTC is an estimate of remaining time under constant closing velocity, NOT a guaranteed collision prediction.
- METRIC vs RELATIVE DISTINCTION:
  - If metric depth is configured and calibrated: TTC (seconds) = Distance (m) / Closing Speed (m/s).
  - If relative depth is used (uncalibrated Depth Anything V2): physical seconds CANNOT be legitimately
    claimed without calibration. In this case, ttc_valid is set to False, ttc_seconds to None, and
    ttc_state to 'RELATIVE_DEPTH_ONLY'.
- Receding or static objects are classified as 'NOT_CLOSING' (no collision countdown).
- Division-by-zero is guarded by minimum_closing_speed threshold.
- Unreliable camera motion is flagged as 'raw_fallback' motion_source.
- No warnings or risk scoring are generated here (belongs to Step 10 & 12).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from temporal.history import ObjectObservation, TemporalHistory
from temporal.camera_motion import CompensatedMotionEstimate, CameraMotionEstimate
from temporal.motion import MotionEstimate


@dataclass
class TTCResult:
    """
    Standardized result of Time-to-Collision estimation for an individual tracked obstacle.
    """
    track_id: int
    ttc_seconds: Optional[float]         # Physical seconds remaining, or None
    ttc_valid: bool                      # True only if metric depth and closing motion are valid
    ttc_state: str                       # 'VALID', 'NOT_CLOSING', 'RELATIVE_DEPTH_ONLY',
                                         # 'INSUFFICIENT_HISTORY', 'INVALID_DEPTH',
                                         # 'INVALID_MOTION', 'INVALID_TIME',
                                         # 'CAMERA_MOTION_UNRELIABLE', 'OUT_OF_RANGE'
    distance_value: Optional[float]      # Measured distance or relative depth value
    distance_type: str                   # 'metric' (meters) or 'relative' (dimensionless)
    closing_speed: Optional[float]       # Rate of closure (m/s or relative units/s)
    depth_type: str                      # 'metric' or 'relative'
    motion_source: str                   # 'compensated' or 'raw_fallback'
    reliability: str                     # 'HIGH', 'MEDIUM', 'LOW', 'INVALID'
    reason: str                          # Human-readable diagnostic explanation
    timestamp: float = 0.0
    frame_index: int = 0
    class_name: str = ""


class TTCEstimator:
    """
    Estimates Time-to-Collision (TTC) for tracked obstacles.
    """

    def __init__(
        self,
        enabled: bool = True,
        minimum_history_observations: int = 3,
        minimum_closing_speed: float = 0.05,
        maximum_time_gap_seconds: float = 0.5,
        max_ttc_seconds: float = 30.0,
        depth_convention: str = "higher_is_closer",
    ) -> None:
        """
        Initialize the TTCEstimator.
        
        Args:
            enabled: Whether TTC estimation is active.
            minimum_history_observations: Required observation count.
            minimum_closing_speed: Minimum closing rate to consider an obstacle closing.
            maximum_time_gap_seconds: Max acceptable dt between observations.
            max_ttc_seconds: Clamping limit for maximum valid TTC.
            depth_convention: 'higher_is_closer' (Depth Anything V2 disparity) or
                              'lower_is_closer' (metric distance in meters).
        """
        self.enabled = bool(enabled)
        self.minimum_history_observations = max(2, int(minimum_history_observations))
        self.minimum_closing_speed = float(minimum_closing_speed)
        self.maximum_time_gap_seconds = float(maximum_time_gap_seconds)
        self.max_ttc_seconds = float(max_ttc_seconds)
        self.depth_convention = str(depth_convention).lower()

    def estimate_track(
        self,
        track_id: int,
        observations: Sequence[ObjectObservation],
        compensated_motion: Optional[CompensatedMotionEstimate] = None,
    ) -> TTCResult:
        """
        Estimate TTC for an individual tracked obstacle.
        """
        if not self.enabled:
            return self._build_result(
                track_id=track_id,
                ttc_seconds=None,
                ttc_valid=False,
                ttc_state="DISABLED",
                distance_value=None,
                distance_type="unknown",
                closing_speed=None,
                depth_type="unknown",
                motion_source="none",
                reliability="INVALID",
                reason="TTC estimation disabled in configuration",
            )

        # 1. Validate sufficient history
        if not observations or len(observations) < self.minimum_history_observations:
            latest = observations[-1] if observations else None
            return self._build_result(
                track_id=track_id,
                ttc_seconds=None,
                ttc_valid=False,
                ttc_state="INSUFFICIENT_HISTORY",
                distance_value=latest.depth_value if latest else None,
                distance_type="metric" if (latest and latest.is_metric) else "relative",
                closing_speed=None,
                depth_type="metric" if (latest and latest.is_metric) else "relative",
                motion_source="none",
                reliability="INVALID",
                reason=f"Insufficient history observations ({len(observations)} < {self.minimum_history_observations})",
                latest_obs=latest,
            )

        curr = observations[-1]
        prev = observations[-2]

        # 2. Validate timestamp interval
        dt = curr.timestamp - prev.timestamp
        if dt <= 0.0 or dt > self.maximum_time_gap_seconds:
            return self._build_result(
                track_id=track_id,
                ttc_seconds=None,
                ttc_valid=False,
                ttc_state="INVALID_TIME",
                distance_value=curr.depth_value,
                distance_type="metric" if curr.is_metric else "relative",
                closing_speed=None,
                depth_type="metric" if curr.is_metric else "relative",
                motion_source="none",
                reliability="INVALID",
                reason=f"Invalid time delta dt={dt:.3f}s (max allowed={self.maximum_time_gap_seconds}s)",
                latest_obs=curr,
            )

        # 3. Validate depth availability
        if not curr.depth_valid or curr.depth_value is None or not prev.depth_valid or prev.depth_value is None:
            return self._build_result(
                track_id=track_id,
                ttc_seconds=None,
                ttc_valid=False,
                ttc_state="INVALID_DEPTH",
                distance_value=curr.depth_value,
                distance_type="metric" if curr.is_metric else "relative",
                closing_speed=None,
                depth_type="metric" if curr.is_metric else "relative",
                motion_source="none",
                reliability="INVALID",
                reason="Object depth measurement is unavailable or invalid",
                latest_obs=curr,
            )

        is_metric = curr.is_metric
        dist_type = "metric" if is_metric else "relative"
        current_distance = float(curr.depth_value)

        # Determine motion source & reliability
        if compensated_motion is not None and compensated_motion.camera_motion_valid:
            motion_source = "compensated"
            comp_rel = compensated_motion.reliability
        else:
            motion_source = "raw_fallback"
            comp_rel = "LOW"

        # 4. Determine closing rate according to depth convention
        # Depth Convention Physics:
        # A) 'higher_is_closer' (Depth Anything V2 / monocular disparity / inverse depth):
        #    Depth value d(t) is proportional to inverse depth: d(t) = s / Z(t), where s > 0 is unknown scale.
        #    When approaching (Z decreasing), d increases: d_curr > d_prev.
        #    Time derivative: d_dot = dd/dt = s * (-1/Z^2) * dZ/dt = s * v_close / Z^2 = (d/Z) * v_close = d / TTC.
        #    Therefore: TTC = Z / v_close = d(t) / d_dot(t).
        #    The unknown scale factor s cancels out completely, yielding valid physical seconds!
        #
        # B) 'lower_is_closer' (Metric distance in meters / LiDAR / calibrated range):
        #    Distance Z(t) decreases as object approaches: Z_curr < Z_prev.
        #    Closing velocity: v_close = (Z_prev - Z_curr) / dt = -dZ/dt.
        #    Therefore: TTC = Z_curr / v_close in seconds.

        # Use smoothed depth rate from compensated motion if available, otherwise raw delta
        if self.depth_convention == "higher_is_closer":
            if compensated_motion is not None and compensated_motion.depth_rate is not None:
                closing_speed = compensated_motion.depth_rate
            else:
                closing_speed = (curr.depth_value - prev.depth_value) / dt
        else:
            # lower_is_closer
            if compensated_motion is not None and compensated_motion.depth_rate is not None:
                closing_speed = -compensated_motion.depth_rate
            else:
                closing_speed = (prev.depth_value - curr.depth_value) / dt

        # Check closing state
        is_closing = closing_speed > self.minimum_closing_speed
        if compensated_motion is not None:
            if compensated_motion.approach_state == "RECEDING":
                is_closing = False
            elif compensated_motion.approach_state == "APPROACHING" and closing_speed > 0:
                is_closing = True

        if not is_closing:
            return self._build_result(
                track_id=track_id,
                ttc_seconds=None,
                ttc_valid=False,
                ttc_state="NOT_CLOSING",
                distance_value=current_distance,
                distance_type=dist_type,
                closing_speed=closing_speed,
                depth_type=dist_type,
                motion_source=motion_source,
                reliability="MEDIUM" if comp_rel in ("HIGH", "MEDIUM") else "LOW",
                reason=f"Object not closing (closing rate {closing_speed:.3f} <= threshold {self.minimum_closing_speed:.3f})",
                latest_obs=curr,
            )

        # Object is closing: compute TTC in physical seconds
        # Division by zero is guarded by minimum_closing_speed threshold
        ttc_raw = current_distance / closing_speed

        # Range clamping & state
        if ttc_raw > self.max_ttc_seconds:
            return self._build_result(
                track_id=track_id,
                ttc_seconds=self.max_ttc_seconds,
                ttc_valid=True,
                ttc_state="OUT_OF_RANGE",
                distance_value=current_distance,
                distance_type=dist_type,
                closing_speed=closing_speed,
                depth_type=dist_type,
                motion_source=motion_source,
                reliability="LOW",
                reason=f"Calculated TTC ({ttc_raw:.1f}s) exceeds maximum range ({self.max_ttc_seconds:.1f}s)",
                latest_obs=curr,
            )

        # Reliability scoring
        if (
            curr.depth_reliability == "HIGH"
            and motion_source == "compensated"
            and len(observations) >= 5
        ):
            ttc_rel = "HIGH"
        elif curr.depth_reliability in ("HIGH", "MEDIUM") and len(observations) >= 3:
            ttc_rel = "MEDIUM"
        else:
            ttc_rel = "LOW"

        return self._build_result(
            track_id=track_id,
            ttc_seconds=ttc_raw,
            ttc_valid=True,
            ttc_state="VALID",
            distance_value=current_distance,
            distance_type=dist_type,
            closing_speed=closing_speed,
            depth_type=dist_type,
            motion_source=motion_source,
            reliability=ttc_rel,
            reason=f"TTC calculated: {ttc_raw:.2f}s (depth={current_distance:.2f}, rate={closing_speed:.2f}, convention={self.depth_convention})",
            latest_obs=curr,
        )

    def estimate_all(
        self,
        temporal_history: TemporalHistory,
        compensated_estimates: Dict[int, CompensatedMotionEstimate],
    ) -> Dict[int, TTCResult]:
        """
        Estimate TTC for all active tracks in temporal history.
        """
        results: Dict[int, TTCResult] = {}
        for track_id in temporal_history.active_track_ids:
            obs_list = temporal_history.get(track_id)
            comp_m = compensated_estimates.get(track_id)
            results[track_id] = self.estimate_track(track_id, obs_list, comp_m)
        return results

    def _build_result(
        self,
        track_id: int,
        ttc_seconds: Optional[float],
        ttc_valid: bool,
        ttc_state: str,
        distance_value: Optional[float],
        distance_type: str,
        closing_speed: Optional[float],
        depth_type: str,
        motion_source: str,
        reliability: str,
        reason: str,
        latest_obs: Optional[ObjectObservation] = None,
    ) -> TTCResult:
        """Helper to construct a typed TTCResult."""
        return TTCResult(
            track_id=track_id,
            ttc_seconds=ttc_seconds,
            ttc_valid=ttc_valid,
            ttc_state=ttc_state,
            distance_value=distance_value,
            distance_type=distance_type,
            closing_speed=closing_speed,
            depth_type=depth_type,
            motion_source=motion_source,
            reliability=reliability,
            reason=reason,
            timestamp=latest_obs.timestamp if latest_obs else 0.0,
            frame_index=latest_obs.frame_index if latest_obs else -1,
            class_name=latest_obs.class_name if latest_obs else "unknown",
        )


# Compatibility aliases for legacy skeleton references
TTCEstimatorInterface = TTCEstimator
AnalyticalTTCEstimator = TTCEstimator
