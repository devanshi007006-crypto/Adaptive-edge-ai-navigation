"""
Path Geometry & Obstacle Overlap Analysis Module.

Defines the walking-path corridor and computes bounding-box overlap
and path relevance scores [0.0, 1.0].
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from navigation.spatial import SpatialObjectRepresentation


@dataclass
class PathOverlapAssessment:
    """Evaluation of an obstacle's spatial intersection with the walking corridor."""
    track_id: int
    overlap_state: str                  # 'OUTSIDE_PATH', 'PARTIAL_PATH_OVERLAP', 'INSIDE_PATH', 'UNKNOWN'
    overlap_ratio: float                # Intersection width / Obstacle width [0.0, 1.0]
    path_relevance: float               # Combined path criticality score [0.0, 1.0]
    distance_to_corridor_center: float  # Normalized horizontal distance from corridor center


class PathGeometryAnalyzer:
    """
    Evaluates geometric intersections between obstacles and the projected walking path corridor.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        path_cfg = self.config.get("path", {})

        self.enabled = bool(path_cfg.get("enabled", True))
        self.center_x = float(path_cfg.get("center_x", 0.50))
        self.width_ratio = float(path_cfg.get("width_ratio", 0.40))

        # Corridor boundaries in normalized horizontal coordinates [0.0, 1.0]
        self.left_boundary = max(0.0, self.center_x - (self.width_ratio / 2.0))
        self.right_boundary = min(1.0, self.center_x + (self.width_ratio / 2.0))

    def assess_obstacle(self, spatial_rep: SpatialObjectRepresentation) -> PathOverlapAssessment:
        """
        Assess bounding-box intersection and relevance to the walking corridor.
        """
        if not self.enabled:
            return PathOverlapAssessment(
                track_id=spatial_rep.track_id,
                overlap_state="UNKNOWN",
                overlap_ratio=0.0,
                path_relevance=0.0,
                distance_to_corridor_center=0.0,
            )

        obj_x1, _, obj_x2, _ = spatial_rep.bbox_norm
        obj_width = max(1e-4, obj_x2 - obj_x1)
        cx, cy = spatial_rep.center_norm

        dist_to_center = abs(cx - self.center_x)

        # 1. Compute 1D horizontal segment intersection
        inter_x1 = max(self.left_boundary, obj_x1)
        inter_x2 = min(self.right_boundary, obj_x2)
        inter_width = max(0.0, inter_x2 - inter_x1)

        overlap_ratio = min(1.0, max(0.0, inter_width / obj_width))

        # 2. Overlap state classification
        if overlap_ratio >= 0.85:
            overlap_state = "INSIDE_PATH"
        elif overlap_ratio > 0.05:
            overlap_state = "PARTIAL_PATH_OVERLAP"
        else:
            overlap_state = "OUTSIDE_PATH"

        # 3. Path Relevance Score [0.0, 1.0]
        # Considers horizontal overlap ratio and ground-plane vertical proximity (cy)
        # Objects in lower image (cy > 0.45) represent closer obstacles along walking surface
        v_proximity = min(1.0, max(0.0, (cy - 0.20) / 0.80))

        if overlap_state == "INSIDE_PATH":
            path_relevance = 0.70 + (0.30 * v_proximity)
        elif overlap_state == "PARTIAL_PATH_OVERLAP":
            path_relevance = (0.30 + 0.40 * overlap_ratio) + (0.30 * v_proximity)
        else:
            # Outside path: small relevance if very close to corridor boundary
            dist_from_border = max(0.0, min(abs(obj_x2 - self.left_boundary), abs(obj_x1 - self.right_boundary)))
            path_relevance = max(0.0, 0.25 - (dist_from_border * 0.5))

        path_relevance = float(min(1.0, max(0.0, path_relevance)))

        return PathOverlapAssessment(
            track_id=spatial_rep.track_id,
            overlap_state=overlap_state,
            overlap_ratio=overlap_ratio,
            path_relevance=path_relevance,
            distance_to_corridor_center=dist_to_center,
        )

    def assess_all(
        self,
        spatial_map: Dict[int, SpatialObjectRepresentation]
    ) -> Dict[int, PathOverlapAssessment]:
        """Assess corridor overlap for all tracked obstacles."""
        return {tid: self.assess_obstacle(s) for tid, s in spatial_map.items()}
