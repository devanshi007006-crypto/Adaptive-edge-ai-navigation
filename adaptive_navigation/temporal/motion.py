"""
Motion & Approach Estimation Module with Temporal Smoothing.

Calculates:
- Image-plane velocity (vx, vy in pixels/second)
- Apparent image speed (pixels/second)
- Bounding-box area rate of change
- Depth rate of change (relative depth units / second)
- Approaching / Receding / Stable classification
- Per-track temporal smoothing (EMA / Moving Average)

CRITICAL SCIENTIFIC & ENGINEERING CONSTRAINTS:
- Do NOT claim physical-world velocity in m/s unless calibrated metric depth is available.
- Actual timestamps (dt = t_curr - t_prev) used; no assumption that dt = 1 / FPS.
- Depth Anything V2 produces relative inverse depth where HIGHER = CLOSER.
  Therefore: positive depth_rate (d_curr > d_prev) -> APPROACHING.
- Strict per-track smoothing state isolation: track IDs never share smoothing state.
- Graceful handling of missing frames, occlusion gaps, and invalid/missing depth.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .history import ObjectObservation, TemporalHistory


@dataclass
class MotionEstimate:
    """
    Structured motion estimate for an individual tracked object.
    
    Contains both raw frame-to-frame derivatives and smoothed temporal values.
    """
    track_id: int
    timestamp: float
    frame_index: int
    class_name: str

    # Observation interval & displacements
    dx: float                           # Center X displacement (pixels)
    dy: float                           # Center Y displacement (pixels)
    dt: float                           # Time interval (seconds)

    # Raw image-plane velocities (pixels / second)
    raw_vx: Optional[float]
    raw_vy: Optional[float]
    raw_speed: Optional[float]

    # Smoothed image-plane velocities (pixels / second)
    smoothed_vx: Optional[float]
    smoothed_vy: Optional[float]
    smoothed_speed: Optional[float]

    # Bounding-box area rate of change (pixels^2 / second)
    raw_area_rate: Optional[float]
    smoothed_area_rate: Optional[float]

    # Depth rate of change (relative depth units / second)
    raw_depth_rate: Optional[float]
    smoothed_depth_rate: Optional[float]

    # Classification & reliability
    approach_state: str                 # 'APPROACHING', 'RECEDING', 'STABLE', 'UNKNOWN'
    motion_valid: bool                  # True if image motion was successfully estimated
    motion_reliability: str             # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'
    depth_convention: str = "higher_is_closer"
    latest_depth_value: Optional[float] = None


class MotionEstimator:
    """
    Estimates 2D image-plane velocity, relative depth rate, and approaching state
    for tracked obstacles from their rolling observation history.
    """

    def __init__(
        self,
        minimum_dt_seconds: float = 0.01,
        max_valid_time_gap_seconds: float = 0.5,
        minimum_history_observations: int = 3,
        smoothing_method: str = "ema",
        smoothing_window: int = 5,
        stable_threshold: float = 0.05,
        minimum_depth_reliability: str = "MEDIUM",
        depth_convention: str = "higher_is_closer",
        temporal_stabilization_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the MotionEstimator.
        
        Args:
            minimum_dt_seconds: Minimum dt below which velocity calculation is skipped.
            max_valid_time_gap_seconds: Max acceptable dt between observations.
            minimum_history_observations: Required observation count for valid motion.
            smoothing_method: 'ema' (exponential moving average) or 'sma' (simple moving average).
            smoothing_window: Effective window length for smoothing filter.
            stable_threshold: Magnitude threshold for depth_rate below which object is STABLE.
            minimum_depth_reliability: Minimum depth reliability for approach classification.
            depth_convention: 'higher_is_closer' for Depth Anything V2.
            temporal_stabilization_config: Configuration dict for temporal hysteresis & multi-frame stabilization.
        """
        self.minimum_dt_seconds = float(minimum_dt_seconds)
        self.max_valid_time_gap_seconds = float(max_valid_time_gap_seconds)
        self.minimum_history_observations = max(2, int(minimum_history_observations))
        self.smoothing_method = str(smoothing_method).lower()
        self.smoothing_window = max(2, int(smoothing_window))
        self.stable_threshold = float(stable_threshold)
        self.minimum_depth_reliability = str(minimum_depth_reliability).upper()
        self.depth_convention = str(depth_convention)

        # Temporal stabilization parameters (Phase 2B)
        self.temporal_stabilization_config = temporal_stabilization_config or {}
        self.stabilization_enabled = bool(self.temporal_stabilization_config.get("enabled", True))
        self.min_consecutive_approaching = int(self.temporal_stabilization_config.get("min_consecutive_approaching", 5))
        self.min_consecutive_receding = int(self.temporal_stabilization_config.get("min_consecutive_receding", 5))
        self.window_observations = max(3, int(self.temporal_stabilization_config.get("window_observations", 8)))
        self.optical_crossval_enabled = bool(self.temporal_stabilization_config.get("optical_crossval_enabled", True))
        self.area_shrink_threshold = float(self.temporal_stabilization_config.get("area_shrink_threshold", -0.15))

        # Exponential smoothing factor alpha
        self.ema_alpha = 2.0 / (self.smoothing_window + 1.0)

        # Internal state per track: track_id -> dict of smoothed values or window buffers
        self._track_states: Dict[int, dict] = {}

    def estimate_track(
        self,
        track_id: int,
        observations: Sequence[ObjectObservation],
    ) -> MotionEstimate:
        """
        Estimate motion and approach state for a single track given its observation history.
        """
        # 1. Check for sufficient history
        if not observations or len(observations) < self.minimum_history_observations:
            latest = observations[-1] if observations else None
            return MotionEstimate(
                track_id=track_id,
                timestamp=latest.timestamp if latest else 0.0,
                frame_index=latest.frame_index if latest else -1,
                class_name=latest.class_name if latest else "unknown",
                dx=0.0,
                dy=0.0,
                dt=0.0,
                raw_vx=None,
                raw_vy=None,
                raw_speed=None,
                smoothed_vx=None,
                smoothed_vy=None,
                smoothed_speed=None,
                raw_area_rate=None,
                smoothed_area_rate=None,
                raw_depth_rate=None,
                smoothed_depth_rate=None,
                approach_state="UNKNOWN",
                motion_valid=False,
                motion_reliability="UNKNOWN",
                depth_convention=self.depth_convention,
            )

        curr = observations[-1]
        prev = observations[-2]

        dt = curr.timestamp - prev.timestamp
        dx = curr.center[0] - prev.center[0]
        dy = curr.center[1] - prev.center[1]

        # 2. Check for abnormal dt or occlusion time gaps
        if dt <= self.minimum_dt_seconds or dt > self.max_valid_time_gap_seconds:
            return MotionEstimate(
                track_id=track_id,
                timestamp=curr.timestamp,
                frame_index=curr.frame_index,
                class_name=curr.class_name,
                dx=dx,
                dy=dy,
                dt=dt,
                raw_vx=None,
                raw_vy=None,
                raw_speed=None,
                smoothed_vx=None,
                smoothed_vy=None,
                smoothed_speed=None,
                raw_area_rate=None,
                smoothed_area_rate=None,
                raw_depth_rate=None,
                smoothed_depth_rate=None,
                approach_state="UNKNOWN",
                motion_valid=False,
                motion_reliability="UNKNOWN",
                depth_convention=self.depth_convention,
            )

        # 3. Calculate raw image-plane velocity & speed (pixels / second)
        raw_vx = dx / dt
        raw_vy = dy / dt
        raw_speed = math.sqrt(raw_vx * raw_vx + raw_vy * raw_vy)

        # 4. Calculate raw bounding-box area rate (pixels^2 / second)
        area_curr = curr.width * curr.height
        area_prev = prev.width * prev.height
        raw_area_rate = (area_curr - area_prev) / dt

        # 5. Calculate raw and multi-frame regression depth rate
        raw_depth_rate: Optional[float] = None
        if (
            curr.depth_valid
            and prev.depth_valid
            and curr.depth_value is not None
            and prev.depth_value is not None
        ):
            delta_d = curr.depth_value - prev.depth_value
            raw_depth_rate = delta_d / dt

        # Multi-frame windowed regression depth rate (Phase 2B stabilization)
        effective_depth_rate = raw_depth_rate
        if self.stabilization_enabled:
            valid_depth_obs = [obs for obs in observations if obs.depth_valid and obs.depth_value is not None]
            if len(valid_depth_obs) >= 3:
                recent_obs = valid_depth_obs[-self.window_observations:]
                ts = [obs.timestamp for obs in recent_obs]
                ds = [obs.depth_value for obs in recent_obs]
                dt_span = ts[-1] - ts[0]
                if dt_span > 0.01:
                    t_mean = sum(ts) / len(ts)
                    d_mean = sum(ds) / len(ds)
                    denom = sum((t - t_mean) ** 2 for t in ts)
                    if denom > 1e-9:
                        effective_depth_rate = sum((t - t_mean) * (d - d_mean) for t, d in zip(ts, ds)) / denom

        # 6. Apply temporal smoothing (EMA or SMA) respecting track identity
        smoothed_vx, smoothed_vy, smoothed_speed, smoothed_area_rate, smoothed_depth_rate = (
            self._smooth_track_motion(
                track_id=track_id,
                raw_vx=raw_vx,
                raw_vy=raw_vy,
                raw_speed=raw_speed,
                raw_area_rate=raw_area_rate,
                raw_depth_rate=effective_depth_rate if effective_depth_rate is not None else raw_depth_rate,
            )
        )

        # 7. Classify approach state with hysteresis & optical cross-validation
        approach_state = self._classify_approach_state(
            track_id=track_id,
            depth_rate=smoothed_depth_rate if smoothed_depth_rate is not None else effective_depth_rate,
            curr_depth_valid=curr.depth_valid,
            curr_depth_reliability=curr.depth_reliability,
            smoothed_area_rate=smoothed_area_rate,
            area_curr=area_curr,
        )

        # 8. Determine motion reliability
        motion_reliability = self._assess_motion_reliability(
            curr_obs=curr,
            dt=dt,
            history_len=len(observations),
            has_depth_rate=(raw_depth_rate is not None),
        )

        return MotionEstimate(
            track_id=track_id,
            timestamp=curr.timestamp,
            frame_index=curr.frame_index,
            class_name=curr.class_name,
            dx=dx,
            dy=dy,
            dt=dt,
            raw_vx=raw_vx,
            raw_vy=raw_vy,
            raw_speed=raw_speed,
            smoothed_vx=smoothed_vx,
            smoothed_vy=smoothed_vy,
            smoothed_speed=smoothed_speed,
            raw_area_rate=raw_area_rate,
            smoothed_area_rate=smoothed_area_rate,
            raw_depth_rate=raw_depth_rate,
            smoothed_depth_rate=smoothed_depth_rate,
            approach_state=approach_state,
            motion_valid=True,
            motion_reliability=motion_reliability,
            depth_convention=self.depth_convention,
            latest_depth_value=float(curr.depth_value) if (curr.depth_valid and curr.depth_value is not None) else None,
        )

    def _smooth_track_motion(
        self,
        track_id: int,
        raw_vx: float,
        raw_vy: float,
        raw_speed: float,
        raw_area_rate: float,
        raw_depth_rate: Optional[float],
    ) -> Tuple[float, float, float, float, Optional[float]]:
        """Smooths raw motion estimates using EMA or windowed moving average."""
        if track_id not in self._track_states:
            self._track_states[track_id] = {
                "vx": raw_vx,
                "vy": raw_vy,
                "speed": raw_speed,
                "area_rate": raw_area_rate,
                "depth_rate": raw_depth_rate,
                "history": deque(maxlen=self.smoothing_window),
                "approach_state": "UNKNOWN",
                "consecutive_pos": 0,
                "consecutive_neg": 0,
                "consecutive_neutral": 0,
            }
            return raw_vx, raw_vy, raw_speed, raw_area_rate, raw_depth_rate

        state = self._track_states[track_id]
        alpha = self.ema_alpha

        # EMA for velocities and speed
        s_vx = alpha * raw_vx + (1.0 - alpha) * state["vx"]
        s_vy = alpha * raw_vy + (1.0 - alpha) * state["vy"]
        s_speed = alpha * raw_speed + (1.0 - alpha) * state["speed"]
        s_area = alpha * raw_area_rate + (1.0 - alpha) * state["area_rate"]

        # EMA for depth rate (handles missing depth gracefully)
        s_depth_rate: Optional[float] = None
        if raw_depth_rate is not None:
            if state["depth_rate"] is not None:
                s_depth_rate = alpha * raw_depth_rate + (1.0 - alpha) * state["depth_rate"]
            else:
                s_depth_rate = raw_depth_rate
        else:
            s_depth_rate = None

        # Update stored state
        state["vx"] = s_vx
        state["vy"] = s_vy
        state["speed"] = s_speed
        state["area_rate"] = s_area
        state["depth_rate"] = s_depth_rate

        return s_vx, s_vy, s_speed, s_area, s_depth_rate

    def _classify_approach_state(
        self,
        track_id: int,
        depth_rate: Optional[float],
        curr_depth_valid: bool,
        curr_depth_reliability: str,
        smoothed_area_rate: Optional[float] = None,
        area_curr: Optional[float] = None,
    ) -> str:
        """
        Classifies whether an obstacle is APPROACHING, RECEDING, STABLE, or UNKNOWN
        with temporal stabilization, hysteresis, and optical cross-validation.
        """
        if not curr_depth_valid or depth_rate is None:
            return "UNKNOWN"

        # Check reliability threshold
        rel = curr_depth_reliability.upper()
        if self.minimum_depth_reliability == "HIGH" and rel != "HIGH":
            return "UNKNOWN"
        elif self.minimum_depth_reliability == "MEDIUM" and rel not in ("HIGH", "MEDIUM"):
            return "UNKNOWN"

        if not self.stabilization_enabled:
            # Baseline instantaneous classification without hysteresis
            if self.depth_convention == "lower_is_closer":
                if depth_rate < -self.stable_threshold:
                    return "APPROACHING"
                elif depth_rate > self.stable_threshold:
                    return "RECEDING"
                else:
                    return "STABLE"
            else:
                if depth_rate > self.stable_threshold:
                    return "APPROACHING"
                elif depth_rate < -self.stable_threshold:
                    return "RECEDING"
                else:
                    return "STABLE"

        state = self._track_states.setdefault(track_id, {
            "approach_state": "UNKNOWN",
            "consecutive_pos": 0,
            "consecutive_neg": 0,
            "consecutive_neutral": 0,
        })

        # Evaluate directional evidence based on depth convention
        if self.depth_convention == "lower_is_closer":
            pos_evidence = depth_rate < -self.stable_threshold  # smaller depth means closer
            neg_evidence = depth_rate > self.stable_threshold   # larger depth means farther
        else:
            pos_evidence = depth_rate > self.stable_threshold   # larger disparity means closer
            neg_evidence = depth_rate < -self.stable_threshold  # smaller disparity means farther

        # Optical area cross-validation:
        # Under perspective projection, an approaching obstacle cannot have a strongly shrinking apparent area.
        # If normalized area rate is strongly negative, reject as positive closing evidence.
        if pos_evidence and self.optical_crossval_enabled:
            if smoothed_area_rate is not None and area_curr is not None and area_curr > 0:
                rel_area_rate = smoothed_area_rate / area_curr
                if rel_area_rate < self.area_shrink_threshold:
                    pos_evidence = False

        if pos_evidence:
            state["consecutive_pos"] = state.get("consecutive_pos", 0) + 1
            state["consecutive_neg"] = 0
            state["consecutive_neutral"] = 0
        elif neg_evidence:
            state["consecutive_neg"] = state.get("consecutive_neg", 0) + 1
            state["consecutive_pos"] = 0
            state["consecutive_neutral"] = 0
        else:
            state["consecutive_pos"] = 0
            state["consecutive_neg"] = 0
            state["consecutive_neutral"] = state.get("consecutive_neutral", 0) + 1

        prev_state = state.get("approach_state", "UNKNOWN")

        if prev_state == "APPROACHING":
            if state["consecutive_neg"] >= self.min_consecutive_receding:
                new_state = "RECEDING"
            elif state["consecutive_neutral"] >= 3:
                new_state = "STABLE"
            else:
                new_state = "APPROACHING"
        elif prev_state == "RECEDING":
            if state["consecutive_pos"] >= self.min_consecutive_approaching:
                new_state = "APPROACHING"
            elif state["consecutive_neutral"] >= 3:
                new_state = "STABLE"
            else:
                new_state = "RECEDING"
        else:  # UNKNOWN or STABLE
            if state["consecutive_pos"] >= self.min_consecutive_approaching:
                new_state = "APPROACHING"
            elif state["consecutive_neg"] >= self.min_consecutive_receding:
                new_state = "RECEDING"
            else:
                new_state = "STABLE"

        state["approach_state"] = new_state
        return new_state

    def _assess_motion_reliability(
        self,
        curr_obs: ObjectObservation,
        dt: float,
        history_len: int,
        has_depth_rate: bool,
    ) -> str:
        """Assesses prototype motion reliability indicator."""
        if not has_depth_rate or curr_obs.depth_reliability == "INVALID":
            return "LOW" if history_len >= self.minimum_history_observations else "UNKNOWN"

        if curr_obs.depth_reliability == "HIGH" and history_len >= 5 and dt < 0.2:
            return "HIGH"
        elif curr_obs.depth_reliability in ("HIGH", "MEDIUM") and history_len >= self.minimum_history_observations:
            return "MEDIUM"
        else:
            return "LOW"

    def estimate_all(
        self,
        temporal_history: TemporalHistory,
    ) -> Dict[int, MotionEstimate]:
        """
        Estimate motion for all active tracks in temporal history.
        """
        estimates: Dict[int, MotionEstimate] = {}
        for track_id in temporal_history.active_track_ids:
            obs_list = temporal_history.get(track_id)
            estimates[track_id] = self.estimate_track(track_id, obs_list)

        # Cleanup internal smoother states for tracks no longer in history
        self.cleanup_lost_tracks(temporal_history.active_track_ids)
        return estimates

    def cleanup_lost_tracks(self, active_track_ids: Sequence[int]) -> None:
        """Prunes smoother states for tracks that have been removed from history."""
        active_set = set(active_track_ids)
        to_delete = [tid for tid in self._track_states if tid not in active_set]
        for tid in to_delete:
            del self._track_states[tid]

    def reset(self) -> None:
        """Clears all stored motion and smoothing states."""
        self._track_states.clear()
