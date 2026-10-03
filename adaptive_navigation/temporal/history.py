from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque

@dataclass
class TrackSnapshot:
    """Snapshot of a tracked object at a single timestamp."""
    timestamp: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    depth: float
    confidence: float

@dataclass
class TrackHistoryRecord:
    """Historical buffer for an individual tracked object."""
    track_id: int
    class_name: str
    snapshots: deque = field(default_factory=deque)

    def add_snapshot(self, snapshot: TrackSnapshot, max_length: int = 15) -> None:
        self.snapshots.append(snapshot)
        while len(self.snapshots) > max_length:
            self.snapshots.popleft()

    def get_trajectory(self) -> List[Tuple[float, float]]:
        return [s.center for s in self.snapshots]

    def get_depth_history(self) -> List[float]:
        return [s.depth for s in self.snapshots]

class RollingTrackHistory:
    """Maintains sliding-window history buffers across all active and recent tracks."""
    def __init__(self, max_history_length: int = 15):
        self.max_history_length = max_history_length
        self.tracks: Dict[int, TrackHistoryRecord] = {}

    def update(self, track_id: int, class_name: str, snapshot: TrackSnapshot) -> None:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackHistoryRecord(track_id=track_id, class_name=class_name)
        self.tracks[track_id].add_snapshot(snapshot, max_length=self.max_history_length)

    def prune_lost_tracks(self, active_track_ids: List[int], max_idle_age: float = 2.0, current_time: float = 0.0) -> None:
        to_delete = []
        for tid, record in self.tracks.items():
            if tid not in active_track_ids:
                if record.snapshots and (current_time - record.snapshots[-1].timestamp > max_idle_age):
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
