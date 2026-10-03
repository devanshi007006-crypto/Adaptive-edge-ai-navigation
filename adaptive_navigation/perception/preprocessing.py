from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

@dataclass
class PreprocessedFrame:
    """Container for validated and preprocessed frame data."""
    original_frame: np.ndarray
    processed_frame: np.ndarray
    original_shape: Tuple[int, int]  # (height, width)
    scale_factor: Tuple[float, float] # (scale_x, scale_y)
    timestamp: float
    frame_index: int

class FramePreprocessor:
    """Preprocesses input video frames for downstream computer vision models."""
    def __init__(self, target_size: Optional[Tuple[int, int]] = (640, 640), normalize: bool = False, to_rgb: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.to_rgb = to_rgb

    def process(self, frame: np.ndarray, timestamp: float, frame_index: int) -> PreprocessedFrame:
        if frame is None or frame.size == 0:
            raise ValueError(f"Invalid frame received at index {frame_index}")
        h, w = frame.shape[:2]
        processed = frame.copy()
        if self.to_rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        if self.target_size is not None and (w, h) != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LINEAR)
            scale_x = self.target_size[0] / float(w)
            scale_y = self.target_size[1] / float(h)
        else:
            scale_x, scale_y = 1.0, 1.0

        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        return PreprocessedFrame(
            original_frame=frame,
            processed_frame=processed,
            original_shape=(h, w),
            scale_factor=(scale_x, scale_y),
            timestamp=timestamp,
            frame_index=frame_index
        )
