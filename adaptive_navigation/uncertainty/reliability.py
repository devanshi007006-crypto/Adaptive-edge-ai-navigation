"""
Uncertainty & Reliability Estimation Module.

Evaluates evidence quality and reliability across perception, temporal tracking,
depth estimation, motion analysis, camera-motion compensation, TTC, and risk evidence.

CRITICAL DISTINCTION:
- RISK SCORE answers: What is the estimated hazard level?
- RELIABILITY answers: How trustworthy and complete is the evidence supporting that score?
- Risk = HIGH and Reliability = LOW must remain possible and transparent.
- This is a heuristic runtime reliability indicator, NOT a scientifically validated
  ground-truth accuracy percentage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from temporal.history import ObjectObservation
from temporal.camera_motion import CameraMotionEstimate, CompensatedMotionEstimate
from risk.ttc import TTCResult
from risk.risk_engine import RiskFeatures, RiskAssessment


@dataclass
class ReliabilityAssessment:
    """
    Standardized multi-component reliability and uncertainty assessment for a tracked obstacle.
    """
    track_id: int
    class_name: str

    # 8 distinct component reliabilities [0.0, 1.0]
    detection_reliability: float
    tracking_reliability: float
    depth_reliability: float
    temporal_reliability: float
    motion_reliability: float
    camera_motion_reliability: float
    ttc_reliability: float
    evidence_reliability: float

    # Composite scores [0.0, 1.0]
    reliability_score: float             # Heuristic runtime reliability (0=none, 1=strong)
    uncertainty_score: float             # 1.0 - reliability_score

    # Categorical classification
    reliability_level: str               # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'
    tracking_state: str                  # 'NEW', 'STABLE', 'WEAK', 'LOST', 'UNKNOWN'

    # Conflict detection & diagnostic flags
    consistency_flags: List[str] = field(default_factory=list)
    primary_reason: str = ""


@dataclass
class SystemReliability:
    """
    Global system-level health and reliability assessment for fail-safe operation.
    """
    camera_healthy: bool
    processing_fps: float
    detector_available: bool
    depth_model_available: bool
    camera_motion_valid: bool
    active_tracks_count: int
    system_status: str                   # 'HEALTHY', 'DEGRADED', 'FAULT'


class ReliabilityEstimator:
    """
    Evaluates individual object evidence reliability and global system reliability.
    """

    def __init__(
        self,
        enabled: bool = True,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        consistency_enabled: bool = True,
    ) -> None:
        """
        Initialize the ReliabilityEstimator.
        """
        self.enabled = bool(enabled)

        # Configurable component weights
        default_weights = {
            "detection": 0.10,
            "tracking": 0.15,
            "depth": 0.20,
            "temporal": 0.10,
            "motion": 0.15,
            "camera_motion": 0.10,
            "ttc": 0.10,
            "evidence_coverage": 0.10,
        }
        self.weights = dict(default_weights if weights is None else weights)

        # Thresholds for categorical reliability level
        default_thresholds = {
            "high": 0.75,
            "medium": 0.50,
            "low": 0.25,
        }
        self.thresholds = dict(default_thresholds if thresholds is None else thresholds)

        self.consistency_enabled = bool(consistency_enabled)

    def assess_object(
        self,
        track_id: int,
        observations: Sequence[ObjectObservation],
        compensated_motion: Optional[CompensatedMotionEstimate],
        camera_motion: Optional[CameraMotionEstimate],
        ttc_result: Optional[TTCResult],
        risk_assessment: Optional[RiskAssessment],
    ) -> ReliabilityAssessment:
        """
        Assess evidence reliability and uncertainty for an individual tracked obstacle.
        """
        consistency_flags: List[str] = []
        obs_count = len(observations)
        latest_obs = observations[-1] if observations else None
        class_name = latest_obs.class_name if latest_obs else "unknown"

        # --- A. Detection Reliability ---
        # Bounded YOLO detection confidence
        conf = latest_obs.confidence if latest_obs else 0.0
        det_rel = float(min(1.0, max(0.0, conf)))
        if det_rel < 0.35:
            consistency_flags.append("LOW_DETECTION_CONFIDENCE")

        # --- B. Tracking Reliability ---
        # Evaluates track age, observation count, and continuity
        if obs_count >= 10:
            track_rel = 1.0
            track_state = "STABLE"
        elif obs_count >= 5:
            track_rel = 0.80
            track_state = "STABLE"
        elif obs_count >= 3:
            track_rel = 0.55
            track_state = "NEW"
        elif obs_count >= 1:
            track_rel = 0.30
            track_state = "NEW"
        else:
            track_rel = 0.0
            track_state = "UNKNOWN"

        # Check for frame gaps
        if obs_count >= 2:
            frame_gap = observations[-1].frame_index - observations[-2].frame_index
            if frame_gap > 1:
                track_rel = max(0.20, track_rel - 0.25)
                track_state = "WEAK"
                consistency_flags.append("LOW_TRACK_CONTINUITY")

        # --- C. Depth Reliability ---
        depth_rel = 0.0
        if latest_obs and latest_obs.depth_valid and latest_obs.depth_value is not None:
            r_str = latest_obs.depth_reliability.upper()
            if r_str == "HIGH":
                depth_rel = 1.0
            elif r_str == "MEDIUM":
                depth_rel = 0.70
            elif r_str == "LOW":
                depth_rel = 0.40
                consistency_flags.append("LOW_DEPTH_STABILITY")
            else:
                depth_rel = 0.0
                consistency_flags.append("INVALID_DEPTH")
        else:
            depth_rel = 0.0
            consistency_flags.append("MISSING_DEPTH")

        # Check for erratic depth jump between consecutive observations
        if obs_count >= 2 and observations[-1].depth_valid and observations[-2].depth_valid:
            d_curr = observations[-1].depth_value
            d_prev = observations[-2].depth_value
            if d_curr is not None and d_prev is not None and d_prev > 1e-4:
                rel_jump = abs(d_curr - d_prev) / d_prev
                if rel_jump > 0.60:
                    depth_rel = max(0.10, depth_rel - 0.30)
                    consistency_flags.append("UNREALISTIC_DEPTH_DELTA")

        # --- D. Temporal Reliability ---
        # Number of observations, timestamp validity, and dt stability
        if obs_count >= 5:
            temp_rel = 1.0
        elif obs_count >= 3:
            temp_rel = 0.70
        elif obs_count >= 2:
            temp_rel = 0.45
        else:
            temp_rel = 0.15

        if obs_count >= 2:
            dt = observations[-1].timestamp - observations[-2].timestamp
            if dt <= 0.0 or dt > 0.50:
                temp_rel = max(0.10, temp_rel - 0.40)
                consistency_flags.append("UNSTABLE_TIMESTAMP_DELTA")

        # --- E. Motion Reliability ---
        motion_rel = 0.0
        if compensated_motion and compensated_motion.motion_valid:
            m_rel_str = compensated_motion.reliability.upper()
            if m_rel_str == "HIGH":
                motion_rel = 1.0
            elif m_rel_str == "MEDIUM":
                motion_rel = 0.75
            elif m_rel_str == "LOW":
                motion_rel = 0.40
            elif m_rel_str == "FALLBACK":
                motion_rel = 0.50
                consistency_flags.append("UNCOMPENSATED_MOTION_FALLBACK")
            else:
                motion_rel = 0.20
        else:
            motion_rel = 0.10
            consistency_flags.append("INVALID_MOTION")

        # --- F. Camera Motion Reliability ---
        cam_rel = 0.0
        if camera_motion and camera_motion.valid:
            c_conf = camera_motion.confidence.upper()
            if c_conf == "HIGH":
                cam_rel = 1.0
            elif c_conf == "MEDIUM":
                cam_rel = 0.70
            elif c_conf == "LOW":
                cam_rel = 0.40
                consistency_flags.append("LOW_CAMERA_MOTION_INLIERS")
            else:
                cam_rel = 0.10
        else:
            cam_rel = 0.0
            consistency_flags.append("UNRELIABLE_CAMERA_MOTION")

        # --- G. TTC Reliability ---
        ttc_rel = 0.0
        if ttc_result:
            if ttc_result.ttc_valid and ttc_result.ttc_state == "VALID":
                t_rel_str = ttc_result.reliability.upper()
                if t_rel_str == "HIGH":
                    ttc_rel = 1.0
                elif t_rel_str == "MEDIUM":
                    ttc_rel = 0.75
                else:
                    ttc_rel = 0.50
            elif ttc_result.ttc_state == "NOT_CLOSING":
                ttc_rel = 0.80  # Confident that it's not closing
            elif ttc_result.ttc_state == "RELATIVE_DEPTH_ONLY":
                # Relative closing rate observed, but metric physical seconds are uncalibrated
                ttc_rel = 0.35
            elif ttc_result.ttc_state == "OUT_OF_RANGE":
                ttc_rel = 0.50
            else:
                ttc_rel = 0.10
                consistency_flags.append(f"TTC_{ttc_result.ttc_state}")
        else:
            ttc_rel = 0.0

        # --- H. Evidence Coverage Reliability ---
        if risk_assessment:
            ev_rel = float(min(1.0, max(0.0, risk_assessment.evidence_coverage)))
            if risk_assessment.risk_score > 0.75 and ev_rel < 0.50:
                consistency_flags.append("HIGH_RISK_LOW_EVIDENCE")
        else:
            ev_rel = 0.30

        # --- Compute Weighted Composite Reliability Score ---
        total_w = sum(self.weights.values())
        weighted_score = (
            self.weights.get("detection", 0.10) * det_rel
            + self.weights.get("tracking", 0.15) * track_rel
            + self.weights.get("depth", 0.20) * depth_rel
            + self.weights.get("temporal", 0.10) * temp_rel
            + self.weights.get("motion", 0.15) * motion_rel
            + self.weights.get("camera_motion", 0.10) * cam_rel
            + self.weights.get("ttc", 0.10) * ttc_rel
            + self.weights.get("evidence_coverage", 0.10) * ev_rel
        ) / max(1e-6, total_w)

        reliability_score = float(min(1.0, max(0.0, weighted_score)))
        uncertainty_score = float(max(0.0, min(1.0, 1.0 - reliability_score)))

        # Categorical Reliability Level
        th_h = self.thresholds.get("high", 0.75)
        th_m = self.thresholds.get("medium", 0.50)
        th_l = self.thresholds.get("low", 0.25)

        if reliability_score >= th_h:
            reliability_level = "HIGH"
        elif reliability_score >= th_m:
            reliability_level = "MEDIUM"
        elif reliability_score >= th_l:
            reliability_level = "LOW"
        else:
            reliability_level = "UNKNOWN"

        # Primary Explainable Reason
        if reliability_level == "HIGH":
            primary_reason = "STABLE_TRACK_AND_VALID_DEPTH"
        elif "INSUFFICIENT_EVIDENCE" in consistency_flags or ev_rel < 0.40:
            primary_reason = "INSUFFICIENT_EVIDENCE"
        elif "MISSING_DEPTH" in consistency_flags or depth_rel == 0.0:
            primary_reason = "MISSING_DEPTH"
        elif "UNRELIABLE_CAMERA_MOTION" in consistency_flags and cam_rel == 0.0:
            primary_reason = "UNRELIABLE_CAMERA_MOTION"
        elif track_state == "NEW" or obs_count < 3:
            primary_reason = "INSUFFICIENT_HISTORY"
        elif "LOW_DETECTION_CONFIDENCE" in consistency_flags:
            primary_reason = "LOW_DETECTION_CONFIDENCE"
        else:
            primary_reason = f"MODERATE_EVIDENCE_QUALITY_{reliability_level}"

        return ReliabilityAssessment(
            track_id=track_id,
            class_name=class_name,
            detection_reliability=det_rel,
            tracking_reliability=track_rel,
            depth_reliability=depth_rel,
            temporal_reliability=temp_rel,
            motion_reliability=motion_rel,
            camera_motion_reliability=cam_rel,
            ttc_reliability=ttc_rel,
            evidence_reliability=ev_rel,
            reliability_score=reliability_score,
            uncertainty_score=uncertainty_score,
            reliability_level=reliability_level,
            tracking_state=track_state,
            consistency_flags=consistency_flags,
            primary_reason=primary_reason,
        )

    def assess_all(
        self,
        observations_map: Dict[int, Sequence[ObjectObservation]],
        compensated_motion_map: Dict[int, CompensatedMotionEstimate],
        camera_motion: Optional[CameraMotionEstimate],
        ttc_map: Dict[int, TTCResult],
        risk_map: Dict[int, RiskAssessment],
    ) -> Dict[int, ReliabilityAssessment]:
        """
        Assess reliability for all active tracked obstacles.
        """
        results: Dict[int, ReliabilityAssessment] = {}
        for track_id, obs_list in observations_map.items():
            results[track_id] = self.assess_object(
                track_id=track_id,
                observations=obs_list,
                compensated_motion=compensated_motion_map.get(track_id),
                camera_motion=camera_motion,
                ttc_result=ttc_map.get(track_id),
                risk_assessment=risk_map.get(track_id),
            )
        return results

    def assess_system(
        self,
        camera_healthy: bool,
        fps: float,
        detector_ok: bool,
        depth_model_ok: bool,
        camera_motion: Optional[CameraMotionEstimate],
        active_tracks: int,
    ) -> SystemReliability:
        """
        Evaluates global system-level health and sensor operational integrity.
        """
        cam_motion_ok = bool(camera_motion and camera_motion.valid)
        
        if not camera_healthy or not detector_ok or not depth_model_ok:
            status = "FAULT"
        elif fps < 5.0 or not cam_motion_ok:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return SystemReliability(
            camera_healthy=camera_healthy,
            processing_fps=float(fps),
            detector_available=detector_ok,
            depth_model_available=depth_model_ok,
            camera_motion_valid=cam_motion_ok,
            active_tracks_count=int(active_tracks),
            system_status=status,
        )
