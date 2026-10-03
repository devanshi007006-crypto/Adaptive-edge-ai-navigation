"""
Ground Truth Representation and Dataset Loader for Step 16 Research Evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
import json
import os
import math


@dataclass
class GroundTruthBBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def iou(self, other: "GroundTruthBBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - inter
        return (inter / union) if union > 0 else 0.0


@dataclass
class GroundTruthObject:
    track_id: int
    class_name: str
    bbox: GroundTruthBBox
    depth: Optional[float] = None
    is_metric_depth: bool = False
    motion_state: str = "UNKNOWN"        # APPROACHING, RECEDING, STATIC, UNKNOWN
    velocity: Optional[float] = None     # m/s or relative units/s
    ttc_seconds: Optional[float] = None  # None if receding / static
    path_occupancy: str = "OFF_PATH"     # IN_PATH, NEAR_PATH, OFF_PATH
    risk_level: str = "UNKNOWN"          # LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN
    warning_required: bool = False
    safe_navigation_action: str = "CONTINUE" # CONTINUE, CAUTION, AVOID_LEFT, AVOID_RIGHT, STOP, UNKNOWN


@dataclass
class GroundTruthFrame:
    frame_index: int
    timestamp: float
    scenario: str = "STANDARD"
    environment: str = "NORMAL"
    objects: List[GroundTruthObject] = field(default_factory=list)
    overall_warning: str = "NO_WARNING"  # NO_WARNING, CAUTION, WARNING, CRITICAL
    overall_navigation: str = "CONTINUE" # CONTINUE, CAUTION, AVOID_LEFT, AVOID_RIGHT, STOP, UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    total_samples: int
    valid_samples: int
    missing_labels: int
    invalid_frames: int
    skipped_samples: int
    class_distribution: Dict[str, int]
    class_imbalance_ratio: float
    available_annotations: Dict[str, bool]
    unavailable_annotations: List[str]


class GroundTruthDataset:
    """
    Dataset manager for research benchmarking and ground-truth comparison.
    Supports file loading (JSON/JSONL) and canonical 10-scenario synthetic benchmark generation.
    """

    def __init__(self, name: str = "Adaptive Navigation Benchmark"):
        self.name = name
        self.frames: List[GroundTruthFrame] = []
        self.available_annotations: Dict[str, bool] = {
            "detection": False,
            "tracking": False,
            "depth": False,
            "motion": False,
            "ttc": False,
            "risk": False,
            "warning": False,
            "navigation": False,
        }

    def load_from_json(self, json_path: str) -> bool:
        """Loads annotated frames from a structured JSON file."""
        if not os.path.exists(json_path):
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.frames = []
            for item in data.get("frames", []):
                objs = []
                for obj_data in item.get("objects", []):
                    b = obj_data.get("bbox", [0, 0, 0, 0])
                    bbox = GroundTruthBBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])
                    obj = GroundTruthObject(
                        track_id=obj_data.get("track_id", -1),
                        class_name=obj_data.get("class_name", "object"),
                        bbox=bbox,
                        depth=obj_data.get("depth"),
                        is_metric_depth=obj_data.get("is_metric_depth", False),
                        motion_state=obj_data.get("motion_state", "UNKNOWN"),
                        velocity=obj_data.get("velocity"),
                        ttc_seconds=obj_data.get("ttc_seconds"),
                        path_occupancy=obj_data.get("path_occupancy", "OFF_PATH"),
                        risk_level=obj_data.get("risk_level", "UNKNOWN"),
                        warning_required=obj_data.get("warning_required", False),
                        safe_navigation_action=obj_data.get("safe_navigation_action", "CONTINUE"),
                    )
                    objs.append(obj)

                frame = GroundTruthFrame(
                    frame_index=item.get("frame_index", 0),
                    timestamp=item.get("timestamp", 0.0),
                    scenario=item.get("scenario", "STANDARD"),
                    environment=item.get("environment", "NORMAL"),
                    objects=objs,
                    overall_warning=item.get("overall_warning", "NO_WARNING"),
                    overall_navigation=item.get("overall_navigation", "CONTINUE"),
                    metadata=item.get("metadata", {}),
                )
                self.frames.append(frame)

            self._update_annotation_availability()
            return True
        except Exception as e:
            print(f"Error loading ground truth dataset from {json_path}: {e}")
            return False

    def generate_canonical_scenarios(self, frames_per_scenario: int = 10) -> None:
        """
        Synthesizes the 10 canonical scenarios specified in Step 16 Section 19
        for comprehensive, deterministic, and reproducible evaluation.
        """
        self.frames = []
        fps = 10.0
        dt = 1.0 / fps

        scenarios = [
            ("SCENARIO_1_STATIC_OBSTACLE", "STATIC"),
            ("SCENARIO_2_APPROACHING_PEDESTRIAN", "APPROACHING"),
            ("SCENARIO_3_APPROACHING_VEHICLE", "APPROACHING_FAST"),
            ("SCENARIO_4_MULTIPLE_OBSTACLES", "MULTI"),
            ("SCENARIO_5_CAMERA_MOTION", "EGO_MOTION"),
            ("SCENARIO_6_LOW_LIGHT", "LOW_LIGHT"),
            ("SCENARIO_7_PARTIAL_OCCLUSION", "OCCLUSION"),
            ("SCENARIO_8_OBJECT_ENTERING_PATH", "ENTERING"),
            ("SCENARIO_9_OBJECT_LEAVING_PATH", "LEAVING"),
            ("SCENARIO_10_CROWDED_ENVIRONMENT", "CROWDED"),
        ]

        frame_idx = 0
        timestamp = 0.0

        for sc_name, sc_type in scenarios:
            for f in range(frames_per_scenario):
                t = f * dt
                objs = []

                if sc_type == "STATIC":
                    bbox = GroundTruthBBox(x1=280.0, y1=200.0, x2=360.0, y2=400.0)
                    objs.append(GroundTruthObject(
                        track_id=101,
                        class_name="chair",
                        bbox=bbox,
                        depth=3.0,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=None,
                        path_occupancy="IN_PATH",
                        risk_level="MEDIUM",
                        warning_required=True,
                        safe_navigation_action="AVOID_LEFT",
                    ))
                    ov_warn = "CAUTION"
                    ov_nav = "AVOID_LEFT"

                elif sc_type == "APPROACHING":
                    dist = max(1.0, 6.0 - (f * 0.45))
                    speed = 1.2
                    ttc = dist / speed
                    risk = "CRITICAL" if dist < 2.5 else "HIGH" if dist < 4.5 else "MEDIUM"
                    warn = "CRITICAL" if dist < 2.5 else "WARNING"
                    bbox = GroundTruthBBox(x1=270.0 - f*2, y1=150.0 - f*4, x2=370.0 + f*2, y2=420.0 + f*4)
                    objs.append(GroundTruthObject(
                        track_id=102,
                        class_name="person",
                        bbox=bbox,
                        depth=dist,
                        is_metric_depth=True,
                        motion_state="APPROACHING",
                        velocity=-speed,
                        ttc_seconds=ttc,
                        path_occupancy="IN_PATH",
                        risk_level=risk,
                        warning_required=True,
                        safe_navigation_action="AVOID_RIGHT" if dist > 2.0 else "STOP",
                    ))
                    ov_warn = warn
                    ov_nav = "AVOID_RIGHT" if dist > 2.0 else "STOP"

                elif sc_type == "APPROACHING_FAST":
                    dist = max(1.5, 12.0 - (f * 1.0))
                    speed = 4.0
                    ttc = dist / speed
                    bbox = GroundTruthBBox(x1=240.0 - f*5, y1=180.0 - f*5, x2=400.0 + f*5, y2=380.0 + f*5)
                    objs.append(GroundTruthObject(
                        track_id=103,
                        class_name="car",
                        bbox=bbox,
                        depth=dist,
                        is_metric_depth=True,
                        motion_state="APPROACHING",
                        velocity=-speed,
                        ttc_seconds=ttc,
                        path_occupancy="IN_PATH",
                        risk_level="CRITICAL",
                        warning_required=True,
                        safe_navigation_action="STOP",
                    ))
                    ov_warn = "CRITICAL"
                    ov_nav = "STOP"

                elif sc_type == "MULTI":
                    objs.append(GroundTruthObject(
                        track_id=104,
                        class_name="trash bin",
                        bbox=GroundTruthBBox(270, 220, 350, 380),
                        depth=2.8,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=None,
                        path_occupancy="IN_PATH",
                        risk_level="HIGH",
                        warning_required=True,
                        safe_navigation_action="AVOID_LEFT",
                    ))
                    objs.append(GroundTruthObject(
                        track_id=105,
                        class_name="bench",
                        bbox=GroundTruthBBox(480, 250, 600, 390),
                        depth=4.5,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=None,
                        path_occupancy="OFF_PATH",
                        risk_level="LOW",
                        warning_required=False,
                        safe_navigation_action="CONTINUE",
                    ))
                    ov_warn = "WARNING"
                    ov_nav = "AVOID_LEFT"

                elif sc_type == "EGO_MOTION":
                    apparent_dist = max(1.5, 5.0 - (f * 0.3))
                    objs.append(GroundTruthObject(
                        track_id=106,
                        class_name="pole",
                        bbox=GroundTruthBBox(300, 100, 340, 420),
                        depth=apparent_dist,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=apparent_dist / 1.0,
                        path_occupancy="IN_PATH",
                        risk_level="MEDIUM",
                        warning_required=True,
                        safe_navigation_action="AVOID_LEFT",
                    ))
                    ov_warn = "CAUTION"
                    ov_nav = "AVOID_LEFT"

                elif sc_type == "LOW_LIGHT":
                    objs.append(GroundTruthObject(
                        track_id=107,
                        class_name="person",
                        bbox=GroundTruthBBox(290, 210, 350, 370),
                        depth=4.0,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=None,
                        path_occupancy="IN_PATH",
                        risk_level="MEDIUM",
                        warning_required=True,
                        safe_navigation_action="CAUTION",
                    ))
                    ov_warn = "CAUTION"
                    ov_nav = "CAUTION"

                elif sc_type == "OCCLUSION":
                    objs.append(GroundTruthObject(
                        track_id=108,
                        class_name="bicycle",
                        bbox=GroundTruthBBox(250, 200, 330, 360),
                        depth=3.2,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        ttc_seconds=None,
                        path_occupancy="NEAR_PATH",
                        risk_level="MEDIUM",
                        warning_required=True,
                        safe_navigation_action="CONTINUE",
                    ))
                    ov_warn = "CAUTION"
                    ov_nav = "CONTINUE"

                elif sc_type == "ENTERING":
                    cx = 150.0 + (f * 18.0)
                    in_path = cx >= 220.0
                    objs.append(GroundTruthObject(
                        track_id=109,
                        class_name="dog",
                        bbox=GroundTruthBBox(cx - 30, 280, cx + 30, 360),
                        depth=2.5,
                        is_metric_depth=True,
                        motion_state="APPROACHING" if in_path else "STATIC",
                        velocity=0.5,
                        ttc_seconds=5.0 if in_path else None,
                        path_occupancy="IN_PATH" if in_path else "NEAR_PATH",
                        risk_level="HIGH" if in_path else "LOW",
                        warning_required=in_path,
                        safe_navigation_action="AVOID_RIGHT" if in_path else "CONTINUE",
                    ))
                    ov_warn = "WARNING" if in_path else "NO_WARNING"
                    ov_nav = "AVOID_RIGHT" if in_path else "CONTINUE"

                elif sc_type == "LEAVING":
                    cx = 300.0 + (f * 18.0)
                    in_path = cx <= 380.0
                    objs.append(GroundTruthObject(
                        track_id=110,
                        class_name="person",
                        bbox=GroundTruthBBox(cx - 35, 180, cx + 35, 400),
                        depth=3.5,
                        is_metric_depth=True,
                        motion_state="RECEDING" if not in_path else "STATIC",
                        velocity=0.6,
                        ttc_seconds=None,
                        path_occupancy="IN_PATH" if in_path else "OFF_PATH",
                        risk_level="LOW" if not in_path else "MEDIUM",
                        warning_required=in_path,
                        safe_navigation_action="CONTINUE",
                    ))
                    ov_warn = "CAUTION" if in_path else "NO_WARNING"
                    ov_nav = "CONTINUE"

                elif sc_type == "CROWDED":
                    objs.append(GroundTruthObject(
                        track_id=111,
                        class_name="person",
                        bbox=GroundTruthBBox(150, 180, 220, 390),
                        depth=4.5,
                        is_metric_depth=True,
                        motion_state="STATIC",
                        velocity=0.0,
                        path_occupancy="OFF_PATH",
                        risk_level="LOW",
                        warning_required=False,
                        safe_navigation_action="CONTINUE",
                    ))
                    objs.append(GroundTruthObject(
                        track_id=112,
                        class_name="person",
                        bbox=GroundTruthBBox(280, 160, 360, 410),
                        depth=max(1.8, 5.0 - f*0.3),
                        is_metric_depth=True,
                        motion_state="APPROACHING",
                        velocity=-1.0,
                        ttc_seconds=(5.0 - f*0.3)/1.0,
                        path_occupancy="IN_PATH",
                        risk_level="HIGH",
                        warning_required=True,
                        safe_navigation_action="AVOID_RIGHT",
                    ))
                    objs.append(GroundTruthObject(
                        track_id=113,
                        class_name="person",
                        bbox=GroundTruthBBox(420, 200, 490, 390),
                        depth=3.8,
                        is_metric_depth=True,
                        motion_state="RECEDING",
                        velocity=0.5,
                        path_occupancy="NEAR_PATH",
                        risk_level="LOW",
                        warning_required=False,
                        safe_navigation_action="CONTINUE",
                    ))
                    ov_warn = "WARNING"
                    ov_nav = "AVOID_RIGHT"

                self.frames.append(GroundTruthFrame(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    scenario=sc_name,
                    environment="NORMAL",
                    objects=objs,
                    overall_warning=ov_warn,
                    overall_navigation=ov_nav,
                    metadata={"scenario_type": sc_type, "frame_in_scenario": f},
                ))
                frame_idx += 1
                timestamp += dt

        self._update_annotation_availability()

    def _update_annotation_availability(self) -> None:
        has_det = any(len(f.objects) > 0 for f in self.frames)
        has_trk = any(any(o.track_id >= 0 for o in f.objects) for f in self.frames)
        has_dep = any(any(o.depth is not None for o in f.objects) for f in self.frames)
        has_mot = any(any(o.motion_state != "UNKNOWN" for o in f.objects) for f in self.frames)
        has_ttc = any(any(o.ttc_seconds is not None for o in f.objects) for f in self.frames)
        has_risk = any(any(o.risk_level != "UNKNOWN" for o in f.objects) for f in self.frames)
        has_warn = any(f.overall_warning != "UNKNOWN" for f in self.frames)
        has_nav = any(f.overall_navigation != "UNKNOWN" for f in self.frames)

        self.available_annotations = {
            "detection": has_det,
            "tracking": has_trk,
            "depth": has_dep,
            "motion": has_mot,
            "ttc": has_ttc,
            "risk": has_risk,
            "warning": has_warn,
            "navigation": has_nav,
        }

    def get_quality_report(self) -> DataQualityReport:
        total = len(self.frames)
        valid = sum(1 for f in self.frames if len(f.objects) > 0)
        missing = total - valid
        class_dist: Dict[str, int] = {}
        for f in self.frames:
            for o in f.objects:
                class_dist[o.class_name] = class_dist.get(o.class_name, 0) + 1

        unavail = [k for k, v in self.available_annotations.items() if not v]
        counts = list(class_dist.values()) if class_dist else [1]
        imbalance = max(counts) / max(1, min(counts))

        return DataQualityReport(
            total_samples=total,
            valid_samples=valid,
            missing_labels=missing,
            invalid_frames=0,
            skipped_samples=0,
            class_distribution=class_dist,
            class_imbalance_ratio=round(imbalance, 2),
            available_annotations=self.available_annotations,
            unavailable_annotations=unavail,
        )
