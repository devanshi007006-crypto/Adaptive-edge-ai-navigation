"""
Test TTC and Risk Consistency across Approaching, Receding, and Static Targets.
Validates Step 10 of Phase 1A:
Verifies that TTC and Risk behavior are physically consistent across both
'higher_is_closer' (Depth Anything V2 disparity) and 'lower_is_closer' (metric distance).
"""

import sys
import os
import unittest

# Ensure workspace root and adaptive_navigation are in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from adaptive_navigation.temporal.history import ObjectObservation
from adaptive_navigation.risk.ttc import TTCEstimator, TTCResult
from adaptive_navigation.risk.risk_engine import RiskEngine, RiskFeatures, RiskAssessment


def make_obs(track_id: int, frame_idx: int, timestamp: float, depth: float, is_metric: bool = False):
    return ObjectObservation(
        timestamp=timestamp,
        frame_index=frame_idx,
        track_id=track_id,
        class_id=0,
        class_name="person",
        bbox=(270.0, 100.0, 370.0, 380.0),
        center=(320.0, 240.0),
        width=100.0,
        height=280.0,
        confidence=0.90,
        depth_value=depth,
        depth_valid=True,
        depth_reliability="HIGH",
        is_metric=is_metric,
    )


class TestTTCConsistency(unittest.TestCase):

    def setUp(self):
        self.risk_engine = RiskEngine()

    def test_higher_is_closer_convention(self):
        """Tests Depth Anything V2 convention where higher depth values mean closer."""
        estimator = TTCEstimator(depth_convention="higher_is_closer", minimum_closing_speed=0.05)

        # 1. Approaching target (depth increases over time: 1.0 -> 1.2 -> 1.4 -> 1.6)
        # dt = 0.1s, delta_d = 0.2 => closing_speed = 2.0 units/s
        # Expected TTC = 1.6 / 2.0 = 0.80 seconds
        obs_approaching = [
            make_obs(1, 0, 0.0, 1.0),
            make_obs(1, 1, 0.1, 1.2),
            make_obs(1, 2, 0.2, 1.4),
            make_obs(1, 3, 0.3, 1.6),
        ]
        res_app = estimator.estimate_track(track_id=1, observations=obs_approaching)
        self.assertTrue(res_app.ttc_valid, "Approaching object should have valid TTC")
        self.assertEqual(res_app.ttc_state, "VALID")
        self.assertIsNotNone(res_app.ttc_seconds)
        self.assertAlmostEqual(res_app.ttc_seconds, 0.80, places=2)
        self.assertAlmostEqual(res_app.closing_speed, 2.0, places=2)

        # 2. Receding target (depth decreases over time: 2.0 -> 1.8 -> 1.6 -> 1.4)
        obs_receding = [
            make_obs(2, 0, 0.0, 2.0),
            make_obs(2, 1, 0.1, 1.8),
            make_obs(2, 2, 0.2, 1.6),
            make_obs(2, 3, 0.3, 1.4),
        ]
        res_rec = estimator.estimate_track(track_id=2, observations=obs_receding)
        self.assertFalse(res_rec.ttc_valid, "Receding object must NOT have valid TTC")
        self.assertEqual(res_rec.ttc_state, "NOT_CLOSING")
        self.assertIsNone(res_rec.ttc_seconds)
        self.assertLess(res_rec.closing_speed, 0.0)

        # 3. Static target (depth is stable: 1.50 -> 1.50 -> 1.50 -> 1.50)
        obs_static = [
            make_obs(3, 0, 0.0, 1.50),
            make_obs(3, 1, 0.1, 1.50),
            make_obs(3, 2, 0.2, 1.50),
            make_obs(3, 3, 0.3, 1.50),
        ]
        res_stat = estimator.estimate_track(track_id=3, observations=obs_static)
        self.assertFalse(res_stat.ttc_valid, "Static object must NOT have valid TTC")
        self.assertEqual(res_stat.ttc_state, "NOT_CLOSING")
        self.assertIsNone(res_stat.ttc_seconds)
        self.assertAlmostEqual(res_stat.closing_speed, 0.0, places=2)

        # Risk Engine Verification:
        # Feed features into RiskEngine and ensure approaching risk > receding and static risk
        feat_app = RiskFeatures(
            track_id=1,
            class_name="person",
            confidence=0.9,
            depth_value=1.6,
            depth_type="relative",
            depth_valid=True,
            depth_reliability="HIGH",
            raw_velocity=(0.0, 0.0),
            compensated_velocity=(0.0, 0.0),
            compensated_speed=0.0,
            approach_state="APPROACHING",
            motion_reliability="HIGH",
            ttc_seconds=res_app.ttc_seconds,
            ttc_state=res_app.ttc_state,
            ttc_valid=res_app.ttc_valid,
            bbox=(270.0, 100.0, 370.0, 380.0),
            center_x=320.0,
            center_y=240.0,
            object_width=100.0,
            object_height=280.0,
        )
        feat_rec = RiskFeatures(
            track_id=2,
            class_name="person",
            confidence=0.9,
            depth_value=1.4,
            depth_type="relative",
            depth_valid=True,
            depth_reliability="HIGH",
            raw_velocity=(0.0, 0.0),
            compensated_velocity=(0.0, 0.0),
            compensated_speed=0.0,
            approach_state="RECEDING",
            motion_reliability="HIGH",
            ttc_seconds=res_rec.ttc_seconds,
            ttc_state=res_rec.ttc_state,
            ttc_valid=res_rec.ttc_valid,
            bbox=(270.0, 100.0, 370.0, 380.0),
            center_x=320.0,
            center_y=240.0,
            object_width=100.0,
            object_height=280.0,
        )
        feat_stat = RiskFeatures(
            track_id=3,
            class_name="person",
            confidence=0.9,
            depth_value=1.5,
            depth_type="relative",
            depth_valid=True,
            depth_reliability="HIGH",
            raw_velocity=(0.0, 0.0),
            compensated_velocity=(0.0, 0.0),
            compensated_speed=0.0,
            approach_state="STABLE",
            motion_reliability="HIGH",
            ttc_seconds=res_stat.ttc_seconds,
            ttc_state=res_stat.ttc_state,
            ttc_valid=res_stat.ttc_valid,
            bbox=(270.0, 100.0, 370.0, 380.0),
            center_x=320.0,
            center_y=240.0,
            object_width=100.0,
            object_height=280.0,
        )

        risk_app = self.risk_engine.assess(feat_app)
        risk_rec = self.risk_engine.assess(feat_rec)
        risk_stat = self.risk_engine.assess(feat_stat)

        self.assertGreater(risk_app.risk_score, risk_rec.risk_score)
        self.assertGreater(risk_app.risk_score, risk_stat.risk_score)
        self.assertEqual(risk_rec.ttc_contribution, 0.0)
        self.assertEqual(risk_stat.ttc_contribution, 0.0)
        self.assertGreater(risk_app.ttc_contribution, 0.5)

    def test_lower_is_closer_convention(self):
        """Tests metric distance convention where lower distance (meters) means closer."""
        estimator = TTCEstimator(depth_convention="lower_is_closer", minimum_closing_speed=0.1)

        # 1. Approaching target (distance drops: 5.0m -> 4.5m -> 4.0m -> 3.5m)
        # dt = 0.1s, delta_d = -0.5m => closing_speed = 5.0 m/s
        # Expected TTC = 3.5m / 5.0 m/s = 0.70 seconds
        obs_approaching = [
            make_obs(10, 0, 0.0, 5.0, is_metric=True),
            make_obs(10, 1, 0.1, 4.5, is_metric=True),
            make_obs(10, 2, 0.2, 4.0, is_metric=True),
            make_obs(10, 3, 0.3, 3.5, is_metric=True),
        ]
        res_app = estimator.estimate_track(track_id=10, observations=obs_approaching)
        self.assertTrue(res_app.ttc_valid)
        self.assertEqual(res_app.ttc_state, "VALID")
        self.assertAlmostEqual(res_app.ttc_seconds, 0.70, places=2)
        self.assertAlmostEqual(res_app.closing_speed, 5.0, places=2)

        # 2. Receding target (distance grows: 3.5m -> 4.0m -> 4.5m -> 5.0m)
        obs_receding = [
            make_obs(20, 0, 0.0, 3.5, is_metric=True),
            make_obs(20, 1, 0.1, 4.0, is_metric=True),
            make_obs(20, 2, 0.2, 4.5, is_metric=True),
            make_obs(20, 3, 0.3, 5.0, is_metric=True),
        ]
        res_rec = estimator.estimate_track(track_id=20, observations=obs_receding)
        self.assertFalse(res_rec.ttc_valid)
        self.assertEqual(res_rec.ttc_state, "NOT_CLOSING")
        self.assertIsNone(res_rec.ttc_seconds)

        # 3. Static target
        obs_static = [
            make_obs(30, 0, 0.0, 4.0, is_metric=True),
            make_obs(30, 1, 0.1, 4.0, is_metric=True),
            make_obs(30, 2, 0.2, 4.0, is_metric=True),
            make_obs(30, 3, 0.3, 4.0, is_metric=True),
        ]
        res_stat = estimator.estimate_track(track_id=30, observations=obs_static)
        self.assertFalse(res_stat.ttc_valid)
        self.assertEqual(res_stat.ttc_state, "NOT_CLOSING")
        self.assertIsNone(res_stat.ttc_seconds)


if __name__ == "__main__":
    unittest.main()
