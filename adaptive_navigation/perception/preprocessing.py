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
    """Preprocesses input video frames for downstream computer vision models.
    
    Adheres to Step 2 specification: keeps preprocessing separate from camera acquisition
    and avoids detector-specific transformations until Step 3.
    """
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        to_rgb: bool = False,
    ):
        self.target_size = target_size
        self.to_rgb = to_rgb

    @staticmethod
    def validate_frame(frame: np.ndarray) -> bool:
        """Validate that the frame exists, has positive dimensions, and contains valid pixel data."""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0 or len(frame.shape) < 2:
            return False
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return False
        return True

    def process(self, frame: np.ndarray, timestamp: float, frame_index: int) -> PreprocessedFrame:
        if not self.validate_frame(frame):
            raise ValueError(f"ERROR: Invalid frame received at index {frame_index}")

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

        return PreprocessedFrame(
            original_frame=frame,
            processed_frame=processed,
            original_shape=(h, w),
            scale_factor=(scale_x, scale_y),
            timestamp=timestamp,
            frame_index=frame_index,
        )
