"""
Phase 2B Verification Tests:
1. Indoor Navigation Class Policy Filtering
2. Temporal Stabilization & Approach Hysteresis
"""

import sys
import os
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from adaptive_navigation.perception.detector import Detection, YOLOObjectDetector
from adaptive_navigation.temporal.history import ObjectObservation
from adaptive_navigation.temporal.motion import MotionEstimator, MotionEstimate


def make_obs(track_id: int, frame_idx: int, timestamp: float, depth: float, width: float = 100.0, height: float = 200.0):
    return ObjectObservation(
        timestamp=timestamp,
        frame_index=frame_idx,
        track_id=track_id,
        class_id=0,
        class_name="person",
        bbox=(200.0, 100.0, 200.0 + width, 100.0 + height),
        center=(200.0 + width / 2.0, 100.0 + height / 2.0),
        width=width,
        height=height,
        confidence=0.90,
        depth_value=depth,
        depth_valid=True,
        depth_reliability="HIGH",
        is_metric=False,
    )


class TestPhase2BStabilization(unittest.TestCase):

    def test_indoor_navigation_class_policy(self):
        """Verifies configuration-driven class filtering without destroying raw evidence."""
        cfg = {
            "enabled": True,
            "policy": "indoor_navigation",
            "allowed_classes": ["person", "chair", "suitcase"],
            "suppressed_classes": ["cat", "toilet", "refrigerator"],
        }
        detector = YOLOObjectDetector(
            model_name_or_path="models/detector/yolo11n.pt",
            class_filter_config=cfg,
            device="cpu",
        )
        self.assertTrue(detector.filter_enabled)
        self.assertIn("person", detector.allowed_classes)
        self.assertIn("toilet", detector.suppressed_classes)

    def test_approach_hysteresis_rejects_single_frame_spike(self):
        """Verifies that a 1-frame disparity spike on a receding target does not flip to APPROACHING."""
        estimator = MotionEstimator(
            stable_threshold=0.08,
            temporal_stabilization_config={
                "enabled": True,
                "min_consecutive_approaching": 5,
                "min_consecutive_receding": 5,
                "window_observations": 8,
                "optical_crossval_enabled": True,
                "area_shrink_threshold": -0.15,
            },
        )

        # 1. Start with 8 receding frames (depth decreasing: 3.5 -> 2.8)
        obs_list = []
        for i in range(8):
            depth = 3.5 - i * 0.10
            obs_list.append(make_obs(1, i, i * 0.1, depth))
            est = estimator.estimate_track(1, obs_list)

        self.assertEqual(est.approach_state, "RECEDING", "Track should establish RECEDING state")

        # 2. Inject a single-frame positive disparity bump (2.8 -> 2.85) representing gait/camera bob
        obs_list.append(make_obs(1, 8, 0.8, 2.85))
        est_bump = estimator.estimate_track(1, obs_list)
        self.assertNotEqual(est_bump.approach_state, "APPROACHING", "Single frame spike must NOT flip state to APPROACHING")

    def test_persistent_approach_transitions_cleanly(self):
        """Verifies that genuine persistent closing motion transitions to APPROACHING."""
        estimator = MotionEstimator(
            stable_threshold=0.08,
            temporal_stabilization_config={
                "enabled": True,
                "min_consecutive_approaching": 5,
                "min_consecutive_receding": 5,
                "window_observations": 8,
                "optical_crossval_enabled": True,
                "area_shrink_threshold": -0.15,
            },
        )

        obs_list = []
        # Feed 8 consecutive closing frames (depth increasing: 1.0 -> 2.6)
        for i in range(8):
            depth = 1.0 + i * 0.20
            obs_list.append(make_obs(2, i, i * 0.1, depth, width=100.0 + i * 5.0, height=200.0 + i * 10.0))
            est = estimator.estimate_track(2, obs_list)

        self.assertEqual(est.approach_state, "APPROACHING", "Persistent closing motion must establish APPROACHING state")


if __name__ == "__main__":
    unittest.main()
