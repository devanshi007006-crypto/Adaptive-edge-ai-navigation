from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time
from pathlib import Path
import numpy as np
import cv2
import torch

from .depth_anything_v2.dpt import DepthAnythingV2
from .tracker import TrackedObject

@dataclass
class DepthResult:
    """Full-frame depth estimation result."""
    depth_map: np.ndarray      # 2D float32 depth map matching frame dimensions
    is_metric: bool            # True only if calibrated physical meters; False for relative depth
    min_depth: float
    max_depth: float
    timestamp: float

@dataclass
class TrackedObjectDepth:
    """Object-level depth estimate associated with a tracked object."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center_x: float
    center_y: float
    depth_value: float         # Aggregated representative depth statistic
    depth_valid: bool          # Whether the depth estimate is valid
    depth_reliability: str     # "HIGH", "MEDIUM", "LOW", or "INVALID"
    is_metric: bool            # Explicit indicator: False = relative depth
    timestamp: float

class DepthEstimatorInterface(ABC):
    """Abstract interface for monocular depth estimation."""
    @abstractmethod
    def load_model(self, checkpoint_path: str, model_type: str = "vits", device: str = "auto") -> None:
        pass

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray, timestamp: float) -> DepthResult:
        pass

    @abstractmethod
    def get_object_depth(self, depth_result: DepthResult, tracked_obj: TrackedObject) -> TrackedObjectDepth:
        pass

class DepthAnythingV2Estimator(DepthEstimatorInterface):
    """Depth Anything V2 monocular depth estimator with object-level depth extraction."""
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        checkpoint_path: str = "models/depth/depth_anything_v2_vits.pth",
        model_type: str = "vits",
        device: str = "auto",
        input_size: int = 518,
        is_metric: bool = False,
        object_statistic: str = "median",
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.device_config = device
        self.input_size = int(input_size)
        self.is_metric = is_metric
        self.object_statistic = object_statistic

        self.model: Optional[DepthAnythingV2] = None
        self.resolved_device: str = "cpu"
        self.last_inference_latency_ms: float = 0.0

        self.load_model(self.checkpoint_path, self.model_type, self.device_config)

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"WARNING: CUDA requested ('{requested_device}') but not available. Using CPU.")
                return "cpu"
            return requested_device
        return "cpu"

    def load_model(self, checkpoint_path: str, model_type: str = "vits", device: str = "auto") -> None:
        """Loads Depth Anything V2 weights into memory onto the target device."""
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.resolved_device = self._resolve_device(device)

        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"ERROR: Unsupported model_type '{self.model_type}'. Choose from: {list(self.MODEL_CONFIGS.keys())}")

        # Resolve relative checkpoint path if necessary
        p = Path(self.checkpoint_path)
        if not p.is_absolute() and not p.exists():
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir / self.checkpoint_path
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(
                f"ERROR: Depth Anything V2 checkpoint not found at: '{self.checkpoint_path}'. "
                f"Please ensure official checkpoint is placed in models/depth/ (e.g. depth_anything_v2_vits.pth)."
            )

        print(f"[DepthAnythingV2Estimator] Loading '{self.model_type}' model from '{p}' onto '{self.resolved_device}'...")
        config = self.MODEL_CONFIGS[self.model_type]
        self.model = DepthAnythingV2(**config)

        state_dict = torch.load(str(p), map_location=self.resolved_device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.resolved_device)
        self.model.eval()
        print(f"[DepthAnythingV2Estimator] Loaded successfully. Metric Depth: {self.is_metric}")

    def estimate_depth(self, frame: np.ndarray, timestamp: float) -> DepthResult:
        """Estimates full-frame depth map with spatial dimensions matching the input frame."""
        if self.model is None:
            raise RuntimeError("ERROR: Depth model is not loaded.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            h, w = (480, 640)
            return DepthResult(np.zeros((h, w), dtype=np.float32), self.is_metric, 0.0, 0.0, timestamp)

        h, w = frame.shape[:2]
        t_start = time.perf_counter()

        try:
            with torch.no_grad():
                # infer_image takes BGR frame (numpy uint8) and returns 2D float32 depth map of shape (h, w)
                depth_map = self.model.infer_image(frame, input_size=self.input_size)
        except Exception as e:
            print(f"ERROR during Depth Anything V2 inference: {e}")
            depth_map = np.zeros((h, w), dtype=np.float32)

        self.last_inference_latency_ms = (time.perf_counter() - t_start) * 1000.0

        min_val = float(np.nanmin(depth_map)) if depth_map.size > 0 else 0.0
        max_val = float(np.nanmax(depth_map)) if depth_map.size > 0 else 0.0

        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            is_metric=self.is_metric,
            min_depth=min_val,
            max_depth=max_val,
            timestamp=timestamp,
        )

    def get_object_depth(self, depth_result: DepthResult, tracked_obj: TrackedObject) -> TrackedObjectDepth:
        """Calculates a robust object-level depth estimate from the object's bounded region."""
        depth_map = depth_result.depth_map
        h, w = depth_map.shape[:2]

        raw_x1, raw_y1, raw_x2, raw_y2 = tracked_obj.bbox
        x1 = max(0, min(int(round(raw_x1)), w - 1))
        y1 = max(0, min(int(round(raw_y1)), h - 1))
        x2 = max(x1 + 1, min(int(round(raw_x2)), w))
        y2 = max(y1 + 1, min(int(round(raw_y2)), h))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return TrackedObjectDepth(
                track_id=tracked_obj.track_id,
                class_id=tracked_obj.class_id,
                class_name=tracked_obj.class_name,
                confidence=tracked_obj.confidence,
                bbox=tracked_obj.bbox,
                center_x=tracked_obj.center_x,
                center_y=tracked_obj.center_y,
                depth_value=0.0,
                depth_valid=False,
                depth_reliability="INVALID",
                is_metric=self.is_metric,
                timestamp=depth_result.timestamp,
            )

        roi = depth_map[y1:y2, x1:x2]
        valid_mask = np.isfinite(roi)
        valid_pixels = roi[valid_mask]

        if len(valid_pixels) == 0:
            return TrackedObjectDepth(
                track_id=tracked_obj.track_id,
                class_id=tracked_obj.class_id,
                class_name=tracked_obj.class_name,
                confidence=tracked_obj.confidence,
                bbox=tracked_obj.bbox,
                center_x=tracked_obj.center_x,
                center_y=tracked_obj.center_y,
                depth_value=0.0,
                depth_valid=False,
                depth_reliability="INVALID",
                is_metric=self.is_metric,
                timestamp=depth_result.timestamp,
            )

        # Robust central statistic: median or percentile
        if self.object_statistic == "median":
            depth_val = float(np.median(valid_pixels))
        elif self.object_statistic == "mean":
            depth_val = float(np.mean(valid_pixels))
        elif self.object_statistic == "percentile_25":
            depth_val = float(np.percentile(valid_pixels, 25))
        else:
            depth_val = float(np.median(valid_pixels))

        # Reliability heuristic based on valid pixel ratio and ROI variance
        valid_ratio = len(valid_pixels) / float(box_w * box_h)
        std_val = float(np.std(valid_pixels))
        
        if valid_ratio > 0.85 and (std_val < (depth_result.max_depth - depth_result.min_depth) * 0.4):
            reliability = "HIGH"
        elif valid_ratio > 0.5:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        return TrackedObjectDepth(
            track_id=tracked_obj.track_id,
            class_id=tracked_obj.class_id,
            class_name=tracked_obj.class_name,
            confidence=tracked_obj.confidence,
            bbox=tracked_obj.bbox,
            center_x=tracked_obj.center_x,
            center_y=tracked_obj.center_y,
            depth_value=depth_val,
            depth_valid=True,
            depth_reliability=reliability,
            is_metric=self.is_metric,
            timestamp=depth_result.timestamp,
        )

    def extract_all_object_depths(
        self,
        depth_result: DepthResult,
        tracked_objects: List[TrackedObject],
    ) -> List[TrackedObjectDepth]:
        """Maps full list of tracked objects to their respective object-level depth estimates."""
        return [self.get_object_depth(depth_result, obj) for obj in tracked_objects]

    @staticmethod
    def colorize_depth(depth_result: DepthResult) -> np.ndarray:
        """Converts raw depth map into a colorized BGR representation for visualization only."""
        d_map = depth_result.depth_map
        if d_map.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        min_val = depth_result.min_depth
        max_val = depth_result.max_depth
        diff = max_val - min_val

        if diff > 1e-6:
            norm_depth = ((d_map - min_val) / diff * 255.0).astype(np.uint8)
        else:
            norm_depth = np.zeros_like(d_map, dtype=np.uint8)

        # Use inferno or turbo colormap for clear depth perception
        colorized = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
        return colorized
