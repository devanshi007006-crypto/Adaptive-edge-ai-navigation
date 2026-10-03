import time
import os
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Deque
from collections import deque
import numpy as np
import cv2

@dataclass
class FramePacket:
    """Stable data contract representing a captured camera/video frame."""
    frame: np.ndarray
    timestamp: float          # High-resolution monotonic timestamp (seconds)
    frame_index: int          # Sequentially incremented frame counter
    resolution: Tuple[int, int] # (width, height)
    capture_fps: float        # Measured capture frame rate

class CameraSource:
    """Manages OpenCV video capture for webcams and prerecorded video files."""
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[int] = None,
    ):
        # Normalize source: if string contains an integer, cast to int
        if isinstance(source, str) and source.strip().isdigit():
            self.source = int(source.strip())
        else:
            self.source = source

        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_video_file: bool = isinstance(self.source, str)
        self.frame_index: int = 0
        self._fps_history: Deque[float] = deque(maxlen=30)
        self._last_frame_time: Optional[float] = None

    def open(self) -> None:
        """Initialize the video capture device or file."""
        if self.is_video_file:
            # Check file existence explicitly
            if not os.path.exists(str(self.source)):
                raise FileNotFoundError(f"ERROR: Video file not found at: '{self.source}'")
            self.cap = cv2.VideoCapture(str(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"ERROR: Unable to open camera source: {self.source}")

        # Configure camera resolution and FPS if applicable (webcam hardware)
        if not self.is_video_file:
            if self.width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            if self.target_fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        self.frame_index = 0
        self._fps_history.clear()
        self._last_frame_time = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Optional[FramePacket]:
        """Capture the next frame and package it with high-resolution timing."""
        if not self.is_opened():
            return None

        ret, frame = self.cap.read()
        current_time = time.perf_counter()

        if not ret or frame is None or frame.size == 0:
            return None

        # Measure capture FPS based on inter-frame interval
        capture_fps = 0.0
        if self._last_frame_time is not None:
            delta_t = current_time - self._last_frame_time
            if delta_t > 0:
                instant_fps = 1.0 / delta_t
                self._fps_history.append(instant_fps)
                capture_fps = sum(self._fps_history) / len(self._fps_history)
        self._last_frame_time = current_time

        h, w = frame.shape[:2]
        packet = FramePacket(
            frame=frame,
            timestamp=current_time,
            frame_index=self.frame_index,
            resolution=(w, h),
            capture_fps=capture_fps,
        )
        self.frame_index += 1
        return packet

    def release(self) -> None:
        """Safely release the camera / video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
