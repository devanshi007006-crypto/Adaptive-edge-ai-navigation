"""
Unit tests for Phase 2C: 2:1 Depth Cadence and Forward Ego-Motion Radial Divergence Compensation.
"""

import unittest
import numpy as np

from adaptive_navigation.temporal.camera_motion import (
    CameraMotionEstimator,
    CameraMotionEstimate,
    CompensatedMotionEstimate,
)
from adaptive_navigation.temporal.motion import MotionEstimate


class TestPhase2CEgoMotion(unittest.TestCase):
    """Verifies forward camera ego-motion estimation and compensation."""

    def setUp(self):
        self.estimator = CameraMotionEstimator(
            enabled=True,
            forward_compensation_config={
                "enabled": True,
                "divergence_threshold": 0.015,
                "divergence_weight": 1.0,
            }
        )

    def test_divergence_calculation_expanding_flow(self):
        """Synthetic radially expanding flow field must yield positive forward divergence."""
        h, w = 480, 640
        cx, cy = w / 2.0, h / 2.0

        # Create synthetic feature grid in previous frame
        xs = np.linspace(100, 540, 10)
        ys = np.linspace(80, 400, 10)
        p0_list = []
        p1_list = []
        dt = 0.0333  # ~30 fps
        expansion_rate = 0.03  # 3% radial expansion per second

        for x in xs:
            for y in ys:
                rx = x - cx
                ry = y - cy
                r = np.sqrt(rx * rx + ry * ry)
                if r < 10:
                    continue
                # radially outward displacement
                dr = expansion_rate * r * dt
                dx = dr * (rx / r)
                dy = dr * (ry / r)
                p0_list.append([x, y])
                p1_list.append([x + dx, y + dy])

        p0 = np.array(p0_list, dtype=np.float32).reshape(-1, 1, 2)
        p1 = np.array(p1_list, dtype=np.float32).reshape(-1, 1, 2)

        # Directly compute radial divergence using estimator formula
        p0_v = p0.reshape(-1, 2)
        p1_v = p1.reshape(-1, 2)
        rx = p0_v[:, 0] - cx
        ry = p0_v[:, 1] - cy
        r = np.sqrt(rx * rx + ry * ry) + 1e-5
        disp = p1_v - p0_v
        rad_disp = (disp[:, 0] * rx + disp[:, 1] * ry) / r
        divs = (rad_disp / r) / dt
        forward_divergence = float(np.median(divs))

        self.assertAlmostEqual(forward_divergence, expansion_rate, places=3)
        self.assertTrue(forward_divergence > self.estimator.divergence_threshold)

    def test_forward_ego_motion_absorbs_receding_gait_spike(self):
        """When walking forward behind a receding target, gait-induced micro-closing is absorbed."""
        # Simulated forward camera motion: divergence = 0.03/s
        cam_motion = CameraMotionEstimate(
            dx=0.0, dy=0.0,
            camera_vx=0.0, camera_vy=0.0, camera_speed=0.0,
            transform=None, inlier_count=50, total_features=60,
            inlier_ratio=0.83, confidence="HIGH", valid=True,
            dt=0.033,
            forward_divergence=0.03,
            is_forward_ego=True,
        )

        # Receding pedestrian experiencing a 1-frame gait oscillation:
        # Apparent raw depth rate = +0.06 (mild positive), but area is stable/shrinking (area_rate <= 0)
        raw_m = MotionEstimate(
            track_id=1, timestamp=1.0, frame_index=30, class_name="person",
            dx=0.0, dy=0.0, dt=0.033,
            raw_vx=0.0, raw_vy=0.0, raw_speed=0.0,
            smoothed_vx=0.0, smoothed_vy=0.0, smoothed_speed=0.0,
            raw_area_rate=-10.0, smoothed_area_rate=-5.0,
            raw_depth_rate=0.06, smoothed_depth_rate=0.06,
            approach_state="APPROACHING",  # Raw state falsely flagged approaching
            motion_valid=True, motion_reliability="HIGH",
            depth_convention="higher_is_closer",
            latest_depth_value=3.0,
        )

        comp = self.estimator.compensate_motion(raw_m, cam_motion)

        # Ego depth rate should be ~ 0.03 * 3.0 = 0.09
        self.assertAlmostEqual(comp.ego_depth_rate, 0.09, places=2)
        # Compensated depth rate = 0.06 - 0.09 = -0.03
        self.assertAlmostEqual(comp.compensated_depth_rate, -0.03, places=2)
        # Approach state must be stabilized to STABLE or RECEDING, eliminating the false APPROACHING spike
        self.assertEqual(comp.approach_state, "STABLE")
        self.assertEqual(comp.world_motion_state, "STATIONARY")

    def test_genuine_oncoming_hazard_remains_approaching(self):
        """Oncoming pedestrian with rapid closure rate must remain APPROACHING despite forward ego-motion."""
        cam_motion = CameraMotionEstimate(
            dx=0.0, dy=0.0,
            camera_vx=0.0, camera_vy=0.0, camera_speed=0.0,
            transform=None, inlier_count=50, total_features=60,
            inlier_ratio=0.83, confidence="HIGH", valid=True,
            dt=0.033,
            forward_divergence=0.03,
            is_forward_ego=True,
        )

        # Strong closing motion: depth rate = +1.20, area actively expanding (+500 px^2/s)
        raw_m = MotionEstimate(
            track_id=1, timestamp=1.0, frame_index=30, class_name="person",
            dx=0.0, dy=0.0, dt=0.033,
            raw_vx=0.0, raw_vy=0.0, raw_speed=0.0,
            smoothed_vx=0.0, smoothed_vy=0.0, smoothed_speed=0.0,
            raw_area_rate=500.0, smoothed_area_rate=450.0,
            raw_depth_rate=1.20, smoothed_depth_rate=1.20,
            approach_state="APPROACHING",
            motion_valid=True, motion_reliability="HIGH",
            depth_convention="higher_is_closer",
            latest_depth_value=3.0,
        )

        comp = self.estimator.compensate_motion(raw_m, cam_motion)

        # Compensated rate = 1.20 - 0.09 = +1.11 >> threshold
        self.assertAlmostEqual(comp.compensated_depth_rate, 1.11, places=2)
        # Must strictly preserve APPROACHING state
        self.assertEqual(comp.approach_state, "APPROACHING")
        self.assertEqual(comp.world_motion_state, "DYNAMIC_APPROACHING")


if __name__ == "__main__":
    unittest.main()
