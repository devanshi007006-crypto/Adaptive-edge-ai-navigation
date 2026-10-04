from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import os
import numpy as np
import torch
from ultralytics import YOLO

@dataclass
class Detection:
    """Standardized detection data contract consumed by downstream modules."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel space
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float
    timestamp: float
    policy_accepted: bool = True             # False if filtered by indoor navigation class policy
    filter_reason: Optional[str] = None      # Documented reason if policy_accepted is False

class DetectorInterface(ABC):
    """Abstract interface for object detection models."""
    @abstractmethod
    def load_model(self, model_name_or_path: str, device: str = "auto") -> None:
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        pass

class YOLOObjectDetector(DetectorInterface):
    """Ultralytics YOLO implementation conforming to the standardized detection interface."""
    def __init__(
        self,
        model_name_or_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "auto",
        classes_of_interest: Optional[List[str]] = None,
        class_filter_config: Optional[dict] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.image_size = int(image_size)
        self.device_config = device
        self.classes_of_interest = set(classes_of_interest) if classes_of_interest else None
        
        # Indoor Navigation Class Policy Configuration
        self.class_filter_config = class_filter_config or {}
        self.filter_enabled = bool(self.class_filter_config.get("enabled", False))
        self.policy_name = str(self.class_filter_config.get("policy", "indoor_navigation"))
        allowed = self.class_filter_config.get("allowed_classes")
        self.allowed_classes = set(allowed) if allowed is not None else None
        suppressed = self.class_filter_config.get("suppressed_classes")
        self.suppressed_classes = set(suppressed) if suppressed is not None else set()

        self.model: Optional[YOLO] = None
        self.resolved_device: str = "cpu"
        self.class_names: Dict[int, str] = {}
        self.last_inference_latency_ms: float = 0.0

        # Load model upon initialization
        self.load_model(self.model_name_or_path, self.device_config)

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve device string: 'auto' selects CUDA if available, otherwise CPU."""
        if requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        elif requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"WARNING: CUDA requested ('{requested_device}') but not available. Falling back to CPU.")
                return "cpu"
            return requested_device
        return "cpu"

    def load_model(self, model_name_or_path: str, device: str = "auto") -> None:
        """Loads the Ultralytics YOLO model onto the target device."""
        self.model_name_or_path = model_name_or_path
        self.resolved_device = self._resolve_device(device)
        print(f"[YOLOObjectDetector] Loading model '{self.model_name_or_path}' onto device '{self.resolved_device}'...")

        try:
            self.model = YOLO(self.model_name_or_path)
            # Query model class names dictionary
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = {}
            print(f"[YOLOObjectDetector] Model loaded successfully. Classes available: {len(self.class_names)}")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"ERROR: Failed to load YOLO model '{self.model_name_or_path}': {e}")

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        """Runs YOLO object detection on the input frame and returns standardized Detection objects."""
        if self.model is None:
            raise RuntimeError("ERROR: Cannot perform detection; YOLO model is not loaded.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        t_start = time.perf_counter()

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.resolved_device,
                verbose=False,
            )
        except Exception as e:
            print(f"ERROR during YOLO inference: {e}")
            self.last_inference_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return []

        self.last_inference_latency_ms = (time.perf_counter() - t_start) * 1000.0

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes_data = first_res.boxes
        xyxy_coords = boxes_data.xyxy.cpu().numpy()
        confidences = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy_coords)):
            conf = float(confidences[i])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(class_ids[i])
            cls_name = self.class_names.get(cls_id, str(cls_id))

            # Filter by classes of interest if configured
            if self.classes_of_interest and cls_name not in self.classes_of_interest:
                continue

            raw_x1, raw_y1, raw_x2, raw_y2 = xyxy_coords[i]

            # Constrain bounding box strictly within frame dimensions
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

            # Evaluate Indoor Navigation Class Policy (records metadata without destroying raw evidence)
            policy_accepted = True
            filter_reason = None
            if self.filter_enabled:
                if cls_name in self.suppressed_classes:
                    policy_accepted = False
                    filter_reason = f"suppressed_by_{self.policy_name}_policy"
                elif self.allowed_classes is not None and cls_name not in self.allowed_classes:
                    policy_accepted = False
                    filter_reason = f"unsupported_by_{self.policy_name}_policy"

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    center_x=center_x,
                    center_y=center_y,
                    width=box_width,
                    height=box_height,
                    timestamp=timestamp,
                    policy_accepted=policy_accepted,
                    filter_reason=filter_reason,
                )
            )

        return detections
