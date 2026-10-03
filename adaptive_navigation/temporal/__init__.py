"""Temporal reasoning module for track history, velocity estimation, and smoothing."""
from .history import (
    ObjectObservation,
    TemporalHistory,
    TrackSnapshot,
    TrackHistoryRecord,
    RollingTrackHistory,
)
from .motion import MotionEstimate, MotionEstimator
from .camera_motion import (
    CameraMotionEstimate,
    CameraMotionEstimator,
    CompensatedMotionEstimate,
    CameraMotionCompensatorInterface,
    OpticalFlowMotionCompensator,
)
from .velocity import ApproachState, VelocityEstimate, VelocityEstimatorInterface, NumericalVelocityEstimator
from .smoothing import TemporalSmootherInterface, ExponentialMovingAverageSmoother

__all__ = [
    "ObjectObservation",
    "TemporalHistory",
    "MotionEstimate",
    "MotionEstimator",
    "CameraMotionEstimate",
    "CameraMotionEstimator",
    "CompensatedMotionEstimate",
    "CameraMotionCompensatorInterface",
    "OpticalFlowMotionCompensator",
    "TrackSnapshot",
    "TrackHistoryRecord",
    "RollingTrackHistory",
    "ApproachState",
    "VelocityEstimate",
    "VelocityEstimatorInterface",
    "NumericalVelocityEstimator",
    "TemporalSmootherInterface",
    "ExponentialMovingAverageSmoother",
]
