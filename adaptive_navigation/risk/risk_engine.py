"""
Multi-Factor Risk Assessment Engine.

Fuses multi-modal observations for each tracked obstacle:
- Time-to-Collision (TTC)
- Depth / Distance proximity
- Closing approach behavior
- Path corridor relevance (image-plane walking corridor)
- Object classification relevance
- Reliability & Evidence Coverage

CRITICAL SAFETY & SCIENTIFIC PRINCIPLES:
- Never decides danger using TTC alone.
- Fuses evidence conservatively: missing data is tracked via evidence_coverage,
  and does NOT silently default to safe zero.
- Preserves UNKNOWN when evidence is insufficient (< minimum_evidence_coverage).
- CRITICAL requires strong evidence coverage and confirmed collision relevance.
- Generates transparent, human-explainable primary and secondary reasons.
- No user-facing speech or navigation commands are generated here (Step 12+).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class RiskFeatures:
    """
    Aggregated multi-factor observations for a single tracked obstacle.
    """
    track_id: int
    class_name: str
    confidence: float

    # Depth & Distance
    depth_value: Optional[float]
    depth_type: str                   # 'metric' or 'relative'
    depth_valid: bool
    depth_reliability: str            # 'HIGH', 'MEDIUM', 'LOW', 'INVALID'

    # Motion & Approach
    raw_velocity: Tuple[Optional[float], Optional[float]]
    compensated_velocity: Tuple[Optional[float], Optional[float]]
    compensated_speed: Optional[float]
    approach_state: str               # 'APPROACHING', 'RECEDING', 'STABLE', 'UNKNOWN'
    motion_reliability: str           # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'

    # Time-to-Collision
    ttc_seconds: Optional[float]
    ttc_state: str                    # 'VALID', 'NOT_CLOSING', 'RELATIVE_DEPTH_ONLY', etc.
    ttc_valid: bool

    # Spatial Geometry
    bbox: Tuple[float, float, float, float]
    center_x: float
    center_y: float
    object_width: float
    object_height: float
    frame_width: int = 640
    frame_height: int = 480

    camera_motion_valid: bool = True


@dataclass
class RiskAssessment:
    """
    Multi-factor risk assessment output for an individual tracked obstacle.
    """
    track_id: int
    class_name: str

    risk_score: float                 # Normalized risk score in [0.0, 1.0]
    risk_level: str                   # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'UNKNOWN'

    ttc_contribution: float           # Normalized TTC risk contribution [0.0, 1.0]
    distance_contribution: float      # Normalized Distance risk contribution [0.0, 1.0]
    approach_contribution: float      # Normalized Approach risk contribution [0.0, 1.0]
    path_contribution: float          # Normalized Path risk contribution [0.0, 1.0]
    class_contribution: float         # Normalized Class risk contribution [0.0, 1.0]

    path_relevance: float             # Corridor intersection score [0.0, 1.0]
    path_state: str                   # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'

    evidence_coverage: float          # Ratio of available evidence weights [0.0, 1.0]
    risk_reliability: str             # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'

    primary_reason: str               # Explainable primary rationale
    secondary_reasons: List[str] = field(default_factory=list)


class RiskEngine:
    """
    Evaluates multi-factor collision and navigation risk for tracked obstacles.
    """

    def __init__(
        self,
        enabled: bool = True,
        weights: Optional[Dict[str, float]] = None,
        score_thresholds: Optional[Dict[str, float]] = None,
        ttc_thresholds: Optional[Dict[str, float]] = None,
        path_config: Optional[Dict[str, Union[bool, float]]] = None,
        minimum_evidence_coverage: float = 0.50,
        class_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the RiskEngine with configurable prototype weights and thresholds.
        """
        self.enabled = bool(enabled)
        
        # Feature weights (default sum to 1.0)
        default_weights = {
            "ttc": 0.35,
            "distance": 0.20,
            "approach": 0.15,
            "path": 0.20,
            "class": 0.10,
        }
        self.weights = dict(default_weights if weights is None else weights)

        # Score thresholds
        default_score_th = {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "critical": 0.85,
        }
        self.score_thresholds = dict(default_score_th if score_thresholds is None else score_thresholds)

        # TTC thresholds (seconds)
        default_ttc_th = {
            "critical": 1.0,
            "high": 2.0,
            "medium": 4.0,
        }
        self.ttc_thresholds = dict(default_ttc_th if ttc_thresholds is None else ttc_thresholds)

        # Path corridor configuration
        default_path_cfg = {
            "enabled": True,
            "center_ratio": 0.50,
            "corridor_width_ratio": 0.40,
        }
        self.path_config = dict(default_path_cfg if path_config is None else path_config)

        self.minimum_evidence_coverage = float(minimum_evidence_coverage)

        # Class weights
        default_class_w = {
            "person": 1.0,
            "car": 1.2,
            "bicycle": 1.1,
            "motorcycle": 1.2,
            "bus": 1.3,
            "truck": 1.3,
        }
        self.class_weights = dict(default_class_w if class_weights is None else class_weights)

    def compute_path_relevance(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, str]:
        """
        Computes 2D image-space path relevance based on intersection with central walking corridor.
        
        Returns:
            (path_relevance: float [0.0, 1.0], path_state: str 'HIGH'/'MEDIUM'/'LOW'/'UNKNOWN')
        """
        if not self.path_config.get("enabled", True) or frame_width <= 0:
            return 0.5, "UNKNOWN"

        center_ratio = float(self.path_config.get("center_ratio", 0.50))
        width_ratio = float(self.path_config.get("corridor_width_ratio", 0.40))

        corridor_center_x = frame_width * center_ratio
        half_corridor_w = (frame_width * width_ratio) / 2.0
        corridor_left = max(0.0, corridor_center_x - half_corridor_w)
        corridor_right = min(float(frame_width), corridor_center_x + half_corridor_w)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0

        # Case 1: Obstacle center is directly inside walking corridor
        if corridor_left <= cx <= corridor_right:
            return 1.0, "HIGH"

        # Case 2: Bounding box partially overlaps walking corridor
        if x1 < corridor_right and x2 > corridor_left:
            return 0.60, "MEDIUM"

        # Case 3: Completely outside corridor
        dist_to_corridor = corridor_left - x2 if x2 < corridor_left else x1 - corridor_right
        falloff_zone = frame_width * 0.25
        proximity_factor = max(0.0, 1.0 - (dist_to_corridor / max(1.0, falloff_zone)))
        relevance = float(proximity_factor * 0.25)
        return relevance, "LOW"

    def assess(self, features: RiskFeatures) -> RiskAssessment:
        """
        Assess multi-factor risk for a single tracked obstacle.
        """
        # 1. Compute Path Relevance
        path_relevance, path_state = self.compute_path_relevance(
            bbox=features.bbox,
            frame_width=features.frame_width,
            frame_height=features.frame_height,
        )

        # 2. Evaluate Feature Contributions & Track Evidence
        available_weight = 0.0
        weighted_sum = 0.0
        secondary_reasons = []

        # --- A. TTC Contribution ---
        w_ttc = self.weights.get("ttc", 0.35)
        ttc_contrib = 0.0
        if features.ttc_valid and features.ttc_seconds is not None:
            available_weight += w_ttc
            ttc_sec = features.ttc_seconds
            if ttc_sec <= self.ttc_thresholds.get("critical", 1.0):
                ttc_contrib = 1.0
                secondary_reasons.append("CRITICAL_TTC")
            elif ttc_sec <= self.ttc_thresholds.get("high", 2.0):
                ttc_contrib = 0.80
                secondary_reasons.append("SHORT_TTC")
            elif ttc_sec <= self.ttc_thresholds.get("medium", 4.0):
                ttc_contrib = 0.50
                secondary_reasons.append("MODERATE_TTC")
            else:
                ttc_contrib = max(0.0, 1.0 - (ttc_sec / 15.0))
            weighted_sum += w_ttc * ttc_contrib
        elif features.ttc_state == "NOT_CLOSING":
            available_weight += w_ttc
            ttc_contrib = 0.0  # Explicitly zero risk for receding/static
            secondary_reasons.append("OBJECT_NOT_CLOSING")
        elif features.ttc_state == "RELATIVE_DEPTH_ONLY":
            # Relative depth is active: use relative closing & proximity proxy
            available_weight += w_ttc
            if features.approach_state == "APPROACHING":
                d_val = float(features.depth_value) if (features.depth_valid and features.depth_value is not None) else 10.0
                norm_d = min(1.0, max(0.0, d_val / 20.0))
                ttc_contrib = min(1.0, 0.50 + 0.50 * norm_d)
                secondary_reasons.append("RELATIVE_CLOSING_PROXIMITY")
            elif features.approach_state == "STABLE":
                ttc_contrib = 0.15
            else:
                ttc_contrib = 0.0
            weighted_sum += w_ttc * ttc_contrib
        else:
            # INSUFFICIENT_HISTORY, etc. -> evidence missing
            ttc_contrib = 0.0

        # --- B. Distance / Proximity Contribution ---
        w_dist = self.weights.get("distance", 0.20)
        dist_contrib = 0.0
        if features.depth_valid and features.depth_value is not None:
            available_weight += w_dist
            d_val = float(features.depth_value)
            if features.depth_type == "metric":
                # Metric distance in meters: <2m = high risk, >10m = low
                if d_val < 2.0:
                    dist_contrib = 1.0
                    secondary_reasons.append("CLOSE_PROXIMITY_METRIC")
                elif d_val < 5.0:
                    dist_contrib = 0.65
                    secondary_reasons.append("MEDIUM_PROXIMITY_METRIC")
                else:
                    dist_contrib = max(0.0, 1.0 - (d_val / 12.0))
            else:
                # Relative depth (Depth Anything V2) + visual looming scale:
                # Depth Anything V2 relative disparity range is [0.0, 25.0]
                norm_depth = min(1.0, max(0.0, d_val / 25.0))
                # Visual scale proximity: obstacle height relative to camera frame height
                norm_scale = min(1.0, max(0.0, float(features.object_height) / max(1.0, float(features.frame_height))))
                dist_contrib = max(norm_depth, norm_scale * 0.70)
                if dist_contrib > 0.60:
                    secondary_reasons.append("HIGH_RELATIVE_PROXIMITY")
            weighted_sum += w_dist * dist_contrib

        # --- C. Approach / Closing Behavior ---
        w_app = self.weights.get("approach", 0.15)
        app_contrib = 0.0
        if features.approach_state != "UNKNOWN":
            available_weight += w_app
            if features.approach_state == "APPROACHING":
                app_contrib = 1.0
                secondary_reasons.append("APPROACHING_TRAJECTORY")
            elif features.approach_state == "STABLE":
                app_contrib = 0.20
            elif features.approach_state == "RECEDING":
                app_contrib = 0.0
                secondary_reasons.append("RECEDING_TRAJECTORY")
            weighted_sum += w_app * app_contrib

        # --- D. Path Corridor Relevance ---
        w_path = self.weights.get("path", 0.20)
        path_contrib = 0.0
        if path_state != "UNKNOWN":
            available_weight += w_path
            path_contrib = path_relevance
            if path_state == "HIGH":
                secondary_reasons.append("DIRECTLY_IN_PATH")
            elif path_state == "MEDIUM":
                secondary_reasons.append("NEAR_PATH_CORRIDOR")
            weighted_sum += w_path * path_contrib

        # --- E. Object Class Weight ---
        w_class = self.weights.get("class", 0.10)
        c_weight = self.class_weights.get(features.class_name.lower(), 1.0)
        # Normalize class weight relative to max class weight (1.5)
        norm_class = min(1.0, c_weight / 1.5)
        class_contrib = norm_class
        available_weight += w_class
        weighted_sum += w_class * class_contrib

        # 3. Compute Evidence Coverage
        total_possible_weight = sum(self.weights.values())
        evidence_coverage = available_weight / max(1e-6, total_possible_weight)

        # 4. Conservative Evidence Handling
        if evidence_coverage < self.minimum_evidence_coverage:
            return RiskAssessment(
                track_id=features.track_id,
                class_name=features.class_name,
                risk_score=0.0,
                risk_level="UNKNOWN",
                ttc_contribution=ttc_contrib,
                distance_contribution=dist_contrib,
                approach_contribution=app_contrib,
                path_contribution=path_contrib,
                class_contribution=class_contrib,
                path_relevance=path_relevance,
                path_state=path_state,
                evidence_coverage=evidence_coverage,
                risk_reliability="UNKNOWN",
                primary_reason="INSUFFICIENT_EVIDENCE",
                secondary_reasons=secondary_reasons,
            )

        # 5. Normalize Risk Score over Available Evidence
        raw_risk_score = weighted_sum / max(1e-6, available_weight)
        risk_score = max(0.0, min(1.0, float(raw_risk_score)))

        # 6. Determine Risk Level
        th_crit = self.score_thresholds.get("critical", 0.85)
        th_high = self.score_thresholds.get("high", 0.75)
        th_med = self.score_thresholds.get("medium", 0.50)
        th_low = self.score_thresholds.get("low", 0.25)

        # CRITICAL requires strong evidence coverage and confirmed collision relevance
        is_critical_eligible = (
            risk_score >= th_crit
            and evidence_coverage >= 0.70
            and (ttc_contrib >= 0.80 or (app_contrib >= 0.80 and path_relevance >= 0.80))
        )

        if is_critical_eligible:
            risk_level = "CRITICAL"
        elif risk_score >= th_high:
            risk_level = "HIGH"
        elif risk_score >= th_med:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 7. Select Explainable Primary Reason
        if risk_level in ("CRITICAL", "HIGH"):
            if ttc_contrib >= 0.80 and features.ttc_valid:
                primary_reason = "SHORT_TTC"
            elif app_contrib >= 0.80 and path_state == "HIGH":
                primary_reason = "APPROACHING_OBJECT_IN_PATH"
            elif ttc_contrib >= 0.70 and path_state in ("HIGH", "MEDIUM"):
                primary_reason = "APPROACHING_OBJECT_IN_PATH"
            elif path_state == "HIGH":
                primary_reason = "HIGH_PATH_RELEVANCE"
            elif dist_contrib >= 0.80:
                primary_reason = "CLOSE_OBJECT"
            else:
                primary_reason = "ELEVATED_MULTI_FACTOR_RISK"
        elif risk_level == "MEDIUM":
            if path_state in ("HIGH", "MEDIUM") and app_contrib > 0.0:
                primary_reason = "APPROACHING_NEAR_PATH"
            else:
                primary_reason = "MODERATE_PROXIMITY"
        else:
            if features.approach_state == "RECEDING":
                primary_reason = "RECEDING_OBJECT"
            elif path_state == "LOW":
                primary_reason = "OBJECT_OUTSIDE_PATH"
            else:
                primary_reason = "LOW_COLLISION_RELEVANCE"

        # 8. Determine Risk Reliability
        if evidence_coverage >= 0.80 and features.depth_reliability in ("HIGH", "MEDIUM"):
            risk_reliability = "HIGH"
        elif evidence_coverage >= 0.60:
            risk_reliability = "MEDIUM"
        else:
            risk_reliability = "LOW"

        return RiskAssessment(
            track_id=features.track_id,
            class_name=features.class_name,
            risk_score=risk_score,
            risk_level=risk_level,
            ttc_contribution=ttc_contrib,
            distance_contribution=dist_contrib,
            approach_contribution=app_contrib,
            path_contribution=path_contrib,
            class_contribution=class_contrib,
            path_relevance=path_relevance,
            path_state=path_state,
            evidence_coverage=evidence_coverage,
            risk_reliability=risk_reliability,
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
        )

    def assess_all(
        self,
        features_dict: Dict[int, RiskFeatures],
    ) -> Dict[int, RiskAssessment]:
        """
        Assess risk for all tracked obstacles in parallel.
        """
        return {tid: self.assess(feat) for tid, feat in features_dict.items()}
