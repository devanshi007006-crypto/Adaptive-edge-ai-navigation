from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque, defaultdict
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml

import ultralytics
from ultralytics.trackers.bot_sort import BOTSORT
from .detector import Detection

@dataclass
class TrackedObject:
    """Standardized tracked object data contract with persistent track ID."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float] # (x1, y1, x2, y2) in pixel space
    center_x: float
    center_y: float
    width: float
    height: float
    timestamp: float

class TrackerInterface(ABC):
    """Abstract interface for multi-object tracking algorithms."""
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray, timestamp: float) -> List[TrackedObject]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

class _DetResultsAdapter:
    """Internal adapter to feed standardized Detection objects to Ultralytics BoT-SORT without re-detection."""
    def __init__(self, xyxy, conf, cls, orig_shape):
        self.xyxy = np.array(xyxy, dtype=np.float32) if len(xyxy) > 0 else np.empty((0, 4), dtype=np.float32)
        if len(xyxy) > 0:
            self.xywh = np.array(
                [[(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1] for x1, y1, x2, y2 in xyxy],
                dtype=np.float32,
            )
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32) if len(conf) > 0 else np.empty((0,), dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32) if len(cls) > 0 else np.empty((0,), dtype=np.float32)
        self.orig_shape = orig_shape

    def __getitem__(self, idx):
        return _DetResultsAdapter(self.xyxy[idx], self.conf[idx], self.cls[idx], self.orig_shape)

    def __len__(self):
        return len(self.conf)

class BoTSORTTracker(TrackerInterface):
    """BoT-SORT tracker wrapping the official Ultralytics BoT-SORT implementation."""
    def __init__(
        self,
        tracker_config: str = "botsort.yaml",
        track_high_thresh: Optional[float] = None,
        track_low_thresh: Optional[float] = None,
        new_track_thresh: Optional[float] = None,
        match_thresh: Optional[float] = None,
        track_buffer: Optional[int] = None,
    ):
        self.tracker_config = tracker_config
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.last_tracker_latency_ms: float = 0.0
        # Temporary short center trail strictly for debug visualization (Step 4 only)
        self.debug_trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=20))
        self.tracker: Optional[BOTSORT] = None

        self._init_tracker()

    def _init_tracker(self) -> None:
        """Loads configuration and instantiates the Ultralytics BOTSORT instance."""
        # Locate official ultralytics tracker configuration file
        cfg_path = Path(self.tracker_config)
        if not cfg_path.exists():
            pkg_trackers = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
            if pkg_trackers.exists():
                cfg_path = pkg_trackers

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
        else:
            # Fallback default BoT-SORT parameters
            cfg_dict = {
                "tracker_type": "botsort",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,
                "model": "auto",
            }

        # Override with explicit parameters if provided
        if self.track_high_thresh is not None:
            cfg_dict["track_high_thresh"] = float(self.track_high_thresh)
        if self.track_low_thresh is not None:
            cfg_dict["track_low_thresh"] = float(self.track_low_thresh)
        if self.new_track_thresh is not None:
            cfg_dict["new_track_thresh"] = float(self.new_track_thresh)
        if self.match_thresh is not None:
            cfg_dict["match_thresh"] = float(self.match_thresh)
        if self.track_buffer is not None:
            cfg_dict["track_buffer"] = int(self.track_buffer)

        args = SimpleNamespace(**cfg_dict)
        self.tracker = BOTSORT(args=args)
        print(f"[BoTSORTTracker] Initialized with config: {cfg_dict.get('tracker_type', 'botsort')}, buffer={cfg_dict.get('track_buffer', 30)}")

    def update(self, detections: List[Detection], frame: np.ndarray, timestamp: float) -> List[TrackedObject]:
        """Updates tracks with newly detected objects and returns active TrackedObjects."""
        if self.tracker is None:
            raise RuntimeError("ERROR: BoT-SORT tracker is not initialized.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        t_start = time.perf_counter()

        # If no detections exist for this frame, pass empty adapter to maintain lost track lifecycle
        if not detections:
            adapter = _DetResultsAdapter([], [], [], (h, w))
            try:
                self.tracker.update(adapter, img=frame)
            except Exception as e:
                print(f"Warning during empty tracker update: {e}")
            self.last_tracker_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return []

        # Build numpy detection arrays from standardized Detection objects
        xyxy = [d.bbox for d in detections]
        conf = [d.confidence for d in detections]
        cls_ids = [d.class_id for d in detections]
        det_map = {i: d for i, d in enumerate(detections)}

        adapter = _DetResultsAdapter(xyxy, conf, cls_ids, (h, w))

        try:
            tracks_out = self.tracker.update(adapter, img=frame)
        except Exception as e:
            print(f"ERROR during BoT-SORT update: {e}")
            self.last_tracker_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return []

        self.last_tracker_latency_ms = (time.perf_counter() - t_start) * 1000.0

        tracked_objects: List[TrackedObject] = []
        if tracks_out is None or len(tracks_out) == 0:
            return tracked_objects

        active_ids = set()

        for row in tracks_out:
            # BoT-SORT output row format: [x1, y1, x2, y2, track_id, conf, cls, idx]
            raw_x1, raw_y1, raw_x2, raw_y2 = row[:4]
            track_id = int(row[4])
            score = float(row[5])
            class_id = int(row[6])
            det_idx = int(row[7]) if len(row) > 7 else -1

            # Resolve class name from original detection if available
            if det_idx in det_map:
                class_name = det_map[det_idx].class_name
            else:
                class_name = str(class_id)

            # Constrain bounding box strictly within frame boundaries
            x1 = max(0.0, min(float(raw_x1), float(w - 1)))
            y1 = max(0.0, min(float(raw_y1), float(h - 1)))
            x2 = max(x1, min(float(raw_x2), float(w)))
            y2 = max(y1, min(float(raw_y2), float(h)))

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    center_x=center_x,
                    center_y=center_y,
                    width=box_width,
                    height=box_height,
                    timestamp=timestamp,
                )
            )

            active_ids.add(track_id)
            # Update temporary short center trail for debug visualization only
            self.debug_trails[track_id].append((int(center_x), int(center_y)))

        # Clean up trails for tracks no longer active
        dead_ids = [tid for tid in self.debug_trails if tid not in active_ids]
        for tid in dead_ids:
            if len(self.debug_trails[tid]) > 0:
                self.debug_trails[tid].popleft()
            if len(self.debug_trails[tid]) == 0:
                del self.debug_trails[tid]

        return tracked_objects

    def reset(self) -> None:
        """Reset internal tracker state and history."""
        self.debug_trails.clear()
        self._init_tracker()
