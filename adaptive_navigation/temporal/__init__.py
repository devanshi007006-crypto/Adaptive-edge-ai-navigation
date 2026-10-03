"""Temporal reasoning module for track history, velocity estimation, and smoothing."""
from .history import TrackSnapshot, TrackHistoryRecord, RollingTrackHistory
from .velocity import ApproachState, VelocityEstimate, VelocityEstimatorInterface, NumericalVelocityEstimator
from .smoothing import TemporalSmootherInterface, ExponentialMovingAverageSmoother
from .camera_motion import CameraMotionEstimate, CameraMotionCompensatorInterface, OpticalFlowMotionCompensator

__all__ = [
    "TrackSnapshot",
    "TrackHistoryRecord",
    "RollingTrackHistory",
    "ApproachState",
    "VelocityEstimate",
    "VelocityEstimatorInterface",
    "NumericalVelocityEstimator",
    "TemporalSmootherInterface",
    "ExponentialMovingAverageSmoother",
    "CameraMotionEstimate",
    "CameraMotionCompensatorInterface",
    "OpticalFlowMotionCompensator",
]
