"""
Smoke test suite to verify module imports and structure integrity.
"""

import sys
import os
import unittest

# Ensure repo root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


class TestModuleImports(unittest.TestCase):
    def test_import_adaptive_navigation(self):
        import adaptive_navigation
        self.assertIsNotNone(adaptive_navigation)

    def test_import_temporal(self):
        from adaptive_navigation.temporal import (
            TemporalHistory,
            ObjectObservation,
            MotionEstimator,
            CameraMotionEstimator,
        )
        self.assertIsNotNone(TemporalHistory)

    def test_import_risk(self):
        from adaptive_navigation.risk import (
            TTCEstimator,
            RiskEngine,
            RiskFeatures,
            RiskAssessment,
        )
        self.assertIsNotNone(RiskEngine)

    def test_import_uncertainty(self):
        from adaptive_navigation.uncertainty import (
            ReliabilityEstimator,
            SystemReliability,
        )
        self.assertIsNotNone(ReliabilityEstimator)

    def test_import_warning(self):
        from adaptive_navigation.warning import (
            WarningStateMachine,
            WarningDecision,
            GlobalWarningDecision,
        )
        self.assertIsNotNone(WarningStateMachine)

    def test_import_navigation(self):
        from adaptive_navigation.navigation import (
            NavigationEngine,
            SpatialAnalyzer,
            PathGeometryAnalyzer,
        )
        self.assertIsNotNone(NavigationEngine)

    def test_import_audio(self):
        from adaptive_navigation.audio import TTSEngine
        self.assertIsNotNone(TTSEngine)

    def test_import_hardware(self):
        from adaptive_navigation.hardware import DeviceManager
        self.assertIsNotNone(DeviceManager)

    def test_config_loading(self):
        import yaml
        config_path = os.path.join(_repo_root, "configs", "final.yaml")
        self.assertTrue(os.path.exists(config_path), f"Config file missing at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("system", cfg)
        self.assertIn("detector", cfg)
        self.assertIn("depth", cfg)
        self.assertIn("temporal", cfg)
        self.assertIn("risk", cfg)


if __name__ == "__main__":
    unittest.main()
