"""Perception module for camera acquisition, object detection, tracking, depth estimation, and preprocessing."""
from .camera import FramePacket, CameraSource
from .preprocessing import FramePreprocessor, PreprocessedFrame
from .detector import Detection, DetectorInterface, YOLOObjectDetector
from .tracker import TrackedObject, TrackerInterface, BoTSORTTracker
from .depth import DepthResult, TrackedObjectDepth, DepthEstimatorInterface, DepthAnythingV2Estimator

__all__ = [
    "FramePacket",
    "CameraSource",
    "FramePreprocessor",
    "PreprocessedFrame",
    "Detection",
    "DetectorInterface",
    "YOLOObjectDetector",
    "TrackedObject",
    "TrackerInterface",
    "BoTSORTTracker",
    "DepthResult",
    "TrackedObjectDepth",
    "DepthEstimatorInterface",
    "DepthAnythingV2Estimator",
]
