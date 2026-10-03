"""
Spatial Position & Image-Space Representation Module.

Extracts normalized obstacle coordinates and tracks image-space zones
(LEFT, CENTER, RIGHT) with temporal stability analysis.

CRITICAL DISTINCTION:
- Image-space zones represent 2D camera coordinate regions ONLY.
- They do NOT represent physical walking directions or safe travel vectors.
- Coordinate normalization: x_norm = center_x / width, y_norm = center_y / height.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence


@dataclass
class SpatialObjectRepresentation:
    """Normalized spatial geometric representation of a tracked obstacle."""
    track_id: int
    bbox_pixel: Tuple[float, float, float, float]       # (x1, y1, x2, y2) in pixels
    center_pixel: Tuple[float, float]                   # (cx, cy) in pixels
    center_norm: Tuple[float, float]                    # (x_norm, y_norm) in [0.0, 1.0]
    bbox_norm: Tuple[float, float, float, float]        # (x1_n, y1_n, x2_n, y2_n)
    size_norm: Tuple[float, float]                      # (width_norm, height_norm)
    spatial_zone: str                                   # 'LEFT', 'CENTER', 'RIGHT'
    spatial_stability: str                              # 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'
    horizontal_motion: Optional[float] = None           # Compensated vx (px/s)
    vertical_motion: Optional[float] = None             # Compensated vy (px/s)


class SpatialZoneTracker:
    """
    Maintains rolling positional zone history for an obstacle to verify spatial stability.
    """

    def __init__(self, history_len: int = 5) -> None:
        self.history_len = history_len
        self.zone_history: deque[str] = deque(maxlen=history_len)

    def update(self, current_zone: str) -> str:
        """
        Record current zone and evaluate stability state across history.
        """
        self.zone_history.append(current_zone)
        count = len(self.zone_history)
        if count < 2:
            return "UNKNOWN"

        occurrences = self.zone_history.count(current_zone)
        ratio = occurrences / count

        if ratio >= 1.0:
            return "HIGH"
        elif ratio >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"


class SpatialAnalyzer:
    """
    Computes normalized coordinates, determines image-space zones,
    and tracks spatial stability across frames.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        zones_cfg = self.config.get("zones", {})
        self.left_boundary = float(zones_cfg.get("left_boundary", 0.33))
        self.right_boundary = float(zones_cfg.get("right_boundary", 0.66))

        self._trackers: Dict[int, SpatialZoneTracker] = {}

    def analyze_object(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        horizontal_motion: Optional[float] = None,
        vertical_motion: Optional[float] = None,
    ) -> SpatialObjectRepresentation:
        """
        Compute spatial properties and zone classification for a single tracked obstacle.
        """
        fw = max(1.0, float(frame_width))
        fh = max(1.0, float(frame_height))

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        x_norm = min(1.0, max(0.0, cx / fw))
        y_norm = min(1.0, max(0.0, cy / fh))
        bbox_norm = (
            min(1.0, max(0.0, x1 / fw)),
            min(1.0, max(0.0, y1 / fh)),
            min(1.0, max(0.0, x2 / fw)),
            min(1.0, max(0.0, y2 / fh)),
        )
        size_norm = (w / fw, h / fh)

        # Image-space zone assignment
        if x_norm < self.left_boundary:
            zone = "LEFT"
        elif x_norm > self.right_boundary:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        # Stability tracking
        if track_id not in self._trackers:
            self._trackers[track_id] = SpatialZoneTracker()

        stability = self._trackers[track_id].update(zone)

        return SpatialObjectRepresentation(
            track_id=track_id,
            bbox_pixel=(x1, y1, x2, y2),
            center_pixel=(cx, cy),
            center_norm=(x_norm, y_norm),
            bbox_norm=bbox_norm,
            size_norm=size_norm,
            spatial_zone=zone,
            spatial_stability=stability,
            horizontal_motion=horizontal_motion,
            vertical_motion=vertical_motion,
        )

    def cleanup_stale_tracks(self, active_track_ids: Sequence[int]) -> None:
        """Removes spatial trackers for tracks no longer active."""
        active_set = set(active_track_ids)
        stale = [tid for tid in self._trackers if tid not in active_set]
        for tid in stale:
            del self._trackers[tid]
