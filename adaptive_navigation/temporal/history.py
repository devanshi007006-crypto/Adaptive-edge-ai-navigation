"""
Temporal History Buffer Module.

Maintains rolling observation history for each tracked object across frames.
Stores strictly numerical and categorical metadata (timestamps, frame indices,
Track IDs, classes, confidences, bounding boxes, centers, and relative depth).

CRITICAL CONSTRAINTS (STEP 6):
- Strictly temporal history storage; NO velocity, TTC, or risk estimation.
- Independent history buffer per Track ID.
- Rolling window with bounded memory (deque).
- No frame copies or heavy image buffers.
- Preserves missing depth as invalid without fabrication.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ObjectObservation:
    """
    Standardized snapshot of a tracked object at a single timestamp.
    
    Contains exact observation coordinates, bounding box dimensions,
    and depth estimates from the perception layer.
    """
    timestamp: float
    frame_index: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]              # (cx, cy)
    width: float
    height: float
    depth_value: Optional[float]             # Relative/Metric depth value or None
    depth_valid: bool                        # True if depth was successfully estimated
    depth_reliability: str                   # 'HIGH', 'MEDIUM', 'LOW', 'INVALID'
    is_metric: bool = False                  # False for relative depth

    @classmethod
    def from_tracked_depth(
        cls,
        obj,  # TrackedObjectDepth instance
        timestamp: float,
        frame_index: int
    ) -> "ObjectObservation":
        """Factory constructor to build an ObjectObservation from TrackedObjectDepth."""
        x1, y1, x2, y2 = obj.bbox
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        
        # Preserve actual depth validity: never replace missing depth with fake values
        d_val = float(obj.depth_value) if (obj.depth_value is not None and obj.depth_valid) else None
        
        return cls(
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            track_id=int(obj.track_id),
            class_id=int(obj.class_id),
            class_name=str(obj.class_name),
            confidence=float(obj.confidence),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            center=(float(obj.center_x), float(obj.center_y)),
            width=width,
            height=height,
            depth_value=d_val,
            depth_valid=bool(obj.depth_valid),
            depth_reliability=str(obj.depth_reliability),
            is_metric=bool(getattr(obj, "is_metric", False)),
        )


class TemporalHistory:
    """
    Rolling temporal history buffer indexing object observations by Track ID.
    
    Key Features:
    - Independent history per Track ID.
    - Rolling window using deque(maxlen=history_length).
    - Time-based validity check and automatic/manual cleanup of lost tracks.
    - Preserves observation gaps without fabricating synthetic frames.
    - Lightweight: strictly numerical metadata (NO full image frames).
    """

    def __init__(
        self,
        history_length: int = 30,
        max_history_age_seconds: float = 2.0,
        cleanup_after_seconds: float = 2.0,
        minimum_observations: int = 3,
    ) -> None:
        """
        Initialize the TemporalHistory buffer.
        
        Args:
            history_length: Maximum number of observations retained per Track ID.
            max_history_age_seconds: Maximum age before a track is considered stale.
            cleanup_after_seconds: Maximum inactive duration before pruning lost tracks.
            minimum_observations: Minimum observation count for sufficient history.
        """
        self.history_length = max(1, int(history_length))
        self.max_history_age_seconds = float(max_history_age_seconds)
        self.cleanup_after_seconds = float(cleanup_after_seconds)
        self.minimum_observations = max(1, int(minimum_observations))
        self._tracks: Dict[int, deque] = {}

    def update(
        self,
        observations: Sequence[ObjectObservation],
        current_timestamp: Optional[float] = None
    ) -> List[int]:
        """
        Append new observations for the current frame and prune stale tracks.
        
        Args:
            observations: Sequence of ObjectObservation for the current frame.
            current_timestamp: Timestamp of the current frame for cleanup calculations.
            
        Returns:
            List of track IDs pruned during this update.
        """
        active_ids = set()
        
        for obs in observations:
            tid = obs.track_id
            active_ids.add(tid)
            if tid not in self._tracks:
                self._tracks[tid] = deque(maxlen=self.history_length)
            self._tracks[tid].append(obs)
            
        # Determine current reference time for cleanup
        ref_time = current_timestamp
        if ref_time is None and observations:
            ref_time = max(o.timestamp for o in observations)
            
        pruned_ids = []
        if ref_time is not None and self.cleanup_after_seconds > 0:
            pruned_ids = self.cleanup(current_timestamp=ref_time)
            
        return pruned_ids

    def get(self, track_id: int) -> List[ObjectObservation]:
        """
        Return the chronological list of stored observations for the given Track ID.
        Returns an empty list if track_id is unknown.
        """
        if track_id in self._tracks:
            return list(self._tracks[track_id])
        return []

    def get_latest(self, track_id: int) -> Optional[ObjectObservation]:
        """Return the most recent observation for a Track ID, or None if not found."""
        if track_id in self._tracks and self._tracks[track_id]:
            return self._tracks[track_id][-1]
        return None

    def get_all(self) -> Dict[int, List[ObjectObservation]]:
        """Return a mapping of all active Track IDs to their observation lists."""
        return {tid: list(deq) for tid, deq in self._tracks.items()}

    def get_length(self, track_id: int) -> int:
        """Return the number of stored observations for a given Track ID."""
        if track_id in self._tracks:
            return len(self._tracks[track_id])
        return 0

    def has_sufficient_history(
        self,
        track_id: int,
        minimum_observations: Optional[int] = None
    ) -> bool:
        """
        Check if a track has accumulated enough observations for future motion analysis.
        """
        threshold = (
            self.minimum_observations
            if minimum_observations is None
            else max(1, int(minimum_observations))
        )
        return self.get_length(track_id) >= threshold

    def is_stale(
        self,
        track_id: int,
        current_timestamp: float,
        max_age: Optional[float] = None
    ) -> bool:
        """
        Check if a track's latest observation is older than the staleness threshold.
        """
        latest = self.get_latest(track_id)
        if latest is None:
            return True
        allowed_age = self.max_history_age_seconds if max_age is None else float(max_age)
        return (current_timestamp - latest.timestamp) > allowed_age

    def cleanup(self, current_timestamp: float) -> List[int]:
        """
        Remove tracks that have not been observed within cleanup_after_seconds.
        
        Returns:
            List of pruned Track IDs.
        """
        to_prune = []
        for tid, deq in self._tracks.items():
            if deq:
                latest_time = deq[-1].timestamp
                if (current_timestamp - latest_time) > self.cleanup_after_seconds:
                    to_prune.append(tid)
            else:
                to_prune.append(tid)

        for tid in to_prune:
            del self._tracks[tid]
            
        return to_prune

    def clear(self, track_id: int) -> bool:
        """Remove a specific Track ID from history. Returns True if found and removed."""
        if track_id in self._tracks:
            del self._tracks[track_id]
            return True
        return False

    def clear_all(self) -> None:
        """Clear all stored track histories."""
        self._tracks.clear()

    @property
    def active_track_ids(self) -> List[int]:
        """Return a list of all currently tracked IDs."""
        return list(self._tracks.keys())


# Compatibility aliases for legacy skeleton references
TrackSnapshot = ObjectObservation
TrackHistoryRecord = TemporalHistory
RollingTrackHistory = TemporalHistory
