"""
Camera Motion Estimation & Compensation Module using Sparse Optical Flow.

Estimates 2D image-plane ego-motion caused by camera movement (e.g., body-worn camera)
and compensates object-plane velocity estimates to distinguish true object motion
from apparent camera motion.

CRITICAL SCIENTIFIC & PRACTICAL LIMITATIONS:
- Monocular optical flow does NOT solve complete 3D camera pose.
- This module implements practical 2D image-plane compensation (translation/similarity).
- Raw object velocities and temporal histories are NEVER overwritten; compensated
  velocities are reported distinctly alongside raw values.
- Moving obstacle bounding boxes are masked out during feature selection to prevent
  foreground object motion from corrupting the dominant global background motion.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .motion import MotionEstimate


@dataclass
class CameraMotionEstimate:
    """
    Estimated 2D global camera motion between consecutive frames.
    """
    dx: float                            # Dominant global X displacement (pixels)
    dy: float                            # Dominant global Y displacement (pixels)
    camera_vx: float                     # Camera image-plane velocity X (pixels/second)
    camera_vy: float                     # Camera image-plane velocity Y (pixels/second)
    camera_speed: float                  # Camera image-plane speed (pixels/second)
    transform: Optional[np.ndarray]      # 2x3 Affine / Euclidean transformation matrix
    inlier_count: int                    # Inlier feature points after RANSAC
    total_features: int                  # Total tracked feature points
    inlier_ratio: float                  # inlier_count / max(1, total_features)
    confidence: str                      # 'HIGH', 'MEDIUM', 'LOW', 'INVALID'
    valid: bool                          # True if camera motion was successfully estimated
    dt: float                            # Frame time interval (seconds)
    feature_points_prev: Optional[np.ndarray] = None
    feature_points_curr: Optional[np.ndarray] = None
    inliers_mask: Optional[np.ndarray] = None


@dataclass
class CompensatedMotionEstimate:
    """
    Motion estimate for a tracked object after camera-motion compensation.
    Preserves both raw observed motion and compensated true motion.
    """
    track_id: int
    timestamp: float
    frame_index: int
    class_name: str

    # Raw observed image motion (from Step 7)
    raw_vx: Optional[float]
    raw_vy: Optional[float]
    raw_speed: Optional[float]

    # Camera-compensated image motion (Step 8)
    compensated_vx: Optional[float]
    compensated_vy: Optional[float]
    compensated_speed: Optional[float]

    # Global camera motion reference
    camera_dx: float
    camera_dy: float
    camera_vx: float
    camera_vy: float
    camera_motion_valid: bool

    # Preserved relative depth & approach state from Step 7
    depth_value: Optional[float]
    depth_rate: Optional[float]
    approach_state: str

    # Reliability flags
    motion_valid: bool
    reliability: str                     # 'HIGH', 'MEDIUM', 'LOW', 'FALLBACK', 'INVALID'


class CameraMotionEstimator:
    """
    Estimates dominant camera ego-motion via Lucas-Kanade sparse optical flow
    with RANSAC outlier rejection, masking out detected obstacle bounding boxes.
    """

    def __init__(
        self,
        enabled: bool = True,
        method: str = "sparse_optical_flow",
        max_features: int = 300,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        ransac_enabled: bool = True,
        minimum_features: int = 20,
        inlier_threshold: float = 3.0,
        max_dt_seconds: float = 0.5,
    ) -> None:
        """
        Initialize the CameraMotionEstimator.
        
        Args:
            enabled: Whether camera motion compensation is active.
            method: Optical flow method identifier ('sparse_optical_flow').
            max_features: Maximum Shi-Tomasi corners to detect.
            quality_level: Corner detection quality threshold.
            min_distance: Minimum distance between detected feature corners.
            ransac_enabled: Whether RANSAC outlier rejection is used.
            minimum_features: Minimum inliers required for valid camera motion.
            inlier_threshold: Maximum reprojection error in pixels for RANSAC inliers.
            max_dt_seconds: Maximum acceptable dt between consecutive frames.
        """
        self.enabled = bool(enabled)
        self.method = str(method)
        self.max_features = int(max_features)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.ransac_enabled = bool(ransac_enabled)
        self.minimum_features = max(4, int(minimum_features))
        self.inlier_threshold = float(inlier_threshold)
        self.max_dt_seconds = float(max_dt_seconds)

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Previous frame state
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self.last_latency_ms: float = 0.0

    def reset(self) -> None:
        """Reset internal frame history."""
        self._prev_gray = None
        self._prev_timestamp = None

    def estimate(
        self,
        curr_frame: np.ndarray,
        timestamp: float,
        object_bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> CameraMotionEstimate:
        """
        Estimate 2D camera ego-motion between previous frame and current frame.
        
        Args:
            curr_frame: Current BGR or grayscale image frame (H x W x 3 or H x W).
            timestamp: Actual monotonic frame timestamp in seconds.
            object_bboxes: Sequence of (x1, y1, x2, y2) bounding boxes to mask out.
            
        Returns:
            CameraMotionEstimate with global dx, dy, camera_vx, camera_vy, and validity.
        """
        if not self.enabled or curr_frame is None or curr_frame.size == 0:
            return self._invalid_estimate(0.0)

        # Convert current frame to grayscale
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame.copy()

        # Check if first frame
        if self._prev_gray is None or self._prev_timestamp is None:
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(0.0)

        # Check resolution match
        if curr_gray.shape != self._prev_gray.shape:
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(0.0)

        dt = timestamp - self._prev_timestamp
        if dt <= 0.0 or dt > self.max_dt_seconds:
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(dt)

        h, w = curr_gray.shape

        # Create mask excluding foreground objects so object motion doesn't skew camera motion
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if object_bboxes:
            for bbox in object_bboxes:
                x1 = max(0, int(bbox[0]) - 5)
                y1 = max(0, int(bbox[1]) - 5)
                x2 = min(w, int(bbox[2]) + 5)
                y2 = min(h, int(bbox[3]) + 5)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        # Step 1: Detect features in previous frame within background mask
        p0 = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )

        if p0 is None or len(p0) < self.minimum_features:
            # Low-texture or insufficient features
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(dt, total_features=(len(p0) if p0 is not None else 0))

        # Step 2: Track features into current frame via Lucas-Kanade Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            p0,
            None,
            **self.lk_params,
        )

        if p1 is None or st is None:
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(dt, total_features=len(p0))

        valid_mask = (st.flatten() == 1)
        good_prev = p0[valid_mask]
        good_curr = p1[valid_mask]

        total_tracked = len(good_prev)
        if total_tracked < self.minimum_features:
            self._prev_gray = curr_gray
            self._prev_timestamp = timestamp
            return self._invalid_estimate(dt, total_features=total_tracked)

        # Step 3: Estimate dominant global motion using RANSAC
        dx, dy = 0.0, 0.0
        transform_matrix = None
        inliers_mask = np.ones(total_tracked, dtype=bool)
        inlier_count = total_tracked

        if self.ransac_enabled and total_tracked >= 4:
            # Estimate 2D Euclidean / similarity transform (translation + rotation)
            M, inliers = cv2.estimateAffinePartial2D(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.inlier_threshold,
                maxIters=500,
            )
            if M is not None and inliers is not None:
                transform_matrix = M
                inliers_mask = (inliers.flatten() == 1)
                inlier_count = int(np.sum(inliers_mask))
                dx = float(M[0, 2])
                dy = float(M[1, 2])
            else:
                # Fallback to robust median translation
                displacements = good_curr - good_prev
                dx = float(np.median(displacements[:, 0, 0]))
                dy = float(np.median(displacements[:, 0, 1]))
                inlier_count = total_tracked
        else:
            displacements = good_curr - good_prev
            dx = float(np.median(displacements[:, 0, 0]))
            dy = float(np.median(displacements[:, 0, 1]))
            inlier_count = total_tracked

        inlier_ratio = inlier_count / max(1, total_tracked)

        # Update frame buffer
        self._prev_gray = curr_gray
        self._prev_timestamp = timestamp

        # Check if enough inliers survived
        if inlier_count < self.minimum_features:
            return self._invalid_estimate(dt, total_features=total_tracked)

        # Calculate camera velocity
        camera_vx = dx / dt
        camera_vy = dy / dt
        camera_speed = math.sqrt(camera_vx * camera_vx + camera_vy * camera_vy)

        # Assess confidence
        if inlier_count >= 50 and inlier_ratio >= 0.65:
            confidence = "HIGH"
        elif inlier_count >= 20 and inlier_ratio >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return CameraMotionEstimate(
            dx=dx,
            dy=dy,
            camera_vx=camera_vx,
            camera_vy=camera_vy,
            camera_speed=camera_speed,
            transform=transform_matrix,
            inlier_count=inlier_count,
            total_features=total_tracked,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            valid=True,
            dt=dt,
            feature_points_prev=good_prev,
            feature_points_curr=good_curr,
            inliers_mask=inliers_mask,
        )

    def compensate_motion(
        self,
        raw_motion: MotionEstimate,
        camera_motion: CameraMotionEstimate,
    ) -> CompensatedMotionEstimate:
        """
        Compensate a single object's motion estimate using the estimated camera motion.
        
        Compensated Object Velocity = Observed Velocity - Camera Ego Velocity
        """
        track_id = raw_motion.track_id
        timestamp = raw_motion.timestamp
        frame_index = raw_motion.frame_index
        class_name = raw_motion.class_name

        if not raw_motion.motion_valid:
            return CompensatedMotionEstimate(
                track_id=track_id,
                timestamp=timestamp,
                frame_index=frame_index,
                class_name=class_name,
                raw_vx=None,
                raw_vy=None,
                raw_speed=None,
                compensated_vx=None,
                compensated_vy=None,
                compensated_speed=None,
                camera_dx=camera_motion.dx,
                camera_dy=camera_motion.dy,
                camera_vx=camera_motion.camera_vx,
                camera_vy=camera_motion.camera_vy,
                camera_motion_valid=camera_motion.valid,
                depth_value=None,
                depth_rate=None,
                approach_state="UNKNOWN",
                motion_valid=False,
                reliability="INVALID",
            )

        # If camera motion is valid, subtract global camera velocity
        if camera_motion.valid:
            comp_vx = raw_motion.smoothed_vx - camera_motion.camera_vx
            comp_vy = raw_motion.smoothed_vy - camera_motion.camera_vy
            comp_speed = math.sqrt(comp_vx * comp_vx + comp_vy * comp_vy)

            # Reliability fusion
            if camera_motion.confidence == "HIGH" and raw_motion.motion_reliability == "HIGH":
                rel = "HIGH"
            elif camera_motion.confidence in ("HIGH", "MEDIUM") and raw_motion.motion_reliability in ("HIGH", "MEDIUM"):
                rel = "MEDIUM"
            else:
                rel = "LOW"
        else:
            # Fallback to uncompensated raw motion without fake zero adjustment
            comp_vx = raw_motion.smoothed_vx
            comp_vy = raw_motion.smoothed_vy
            comp_speed = raw_motion.smoothed_speed
            rel = "FALLBACK"

        return CompensatedMotionEstimate(
            track_id=track_id,
            timestamp=timestamp,
            frame_index=frame_index,
            class_name=class_name,
            raw_vx=raw_motion.smoothed_vx,
            raw_vy=raw_motion.smoothed_vy,
            raw_speed=raw_motion.smoothed_speed,
            compensated_vx=comp_vx,
            compensated_vy=comp_vy,
            compensated_speed=comp_speed,
            camera_dx=camera_motion.dx,
            camera_dy=camera_motion.dy,
            camera_vx=camera_motion.camera_vx,
            camera_vy=camera_motion.camera_vy,
            camera_motion_valid=camera_motion.valid,
            depth_value=raw_motion.raw_depth_rate,
            depth_rate=raw_motion.smoothed_depth_rate,
            approach_state=raw_motion.approach_state,
            motion_valid=True,
            reliability=rel,
        )

    def compensate_all(
        self,
        raw_motion_estimates: Dict[int, MotionEstimate],
        camera_motion: CameraMotionEstimate,
    ) -> Dict[int, CompensatedMotionEstimate]:
        """
        Compensate motion for all tracked obstacles.
        """
        compensated: Dict[int, CompensatedMotionEstimate] = {}
        for track_id, raw_m in raw_motion_estimates.items():
            compensated[track_id] = self.compensate_motion(raw_m, camera_motion)
        return compensated

    def _invalid_estimate(self, dt: float, total_features: int = 0) -> CameraMotionEstimate:
        """Creates an invalid CameraMotionEstimate."""
        return CameraMotionEstimate(
            dx=0.0,
            dy=0.0,
            camera_vx=0.0,
            camera_vy=0.0,
            camera_speed=0.0,
            transform=None,
            inlier_count=0,
            total_features=total_features,
            inlier_ratio=0.0,
            confidence="INVALID",
            valid=False,
            dt=dt,
        )


# Compatibility aliases for legacy skeleton references
CameraMotionCompensatorInterface = CameraMotionEstimator
OpticalFlowMotionCompensator = CameraMotionEstimator
