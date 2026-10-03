"""Main pipeline orchestrator for the Adaptive Edge-AI Navigation System."""
import argparse
import time
import sys
import os
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from perception import FramePreprocessor, YOLOObjectDetector, BoTSORTTracker, DepthAnythingV2Estimator
from temporal import RollingTrackHistory, NumericalVelocityEstimator, ExponentialMovingAverageSmoother, OpticalFlowMotionCompensator
from risk import AnalyticalTTCEstimator, RiskFeatureExtractor, UncertaintyEstimator, RiskScorer, RiskStateMachine
from navigation import ZoneDivider, FreeSpaceAnalyzer, NavigationEngine
from feedback import Pyttsx3SpeechEngine, MessageGenerator, WarningManager
from evaluation import NavigationLogger, MetricsCalculator, ReportGenerator, FrameLogRecord

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_dry_run_validation(config: dict) -> bool:
    print("=" * 60)
    print("ADAPTIVE NAVIGATION PIPELINE - DRY RUN VALIDATION")
    print("=" * 60)
    print("[1/6] Validating Config Schema...")
    required_sections = ["camera", "perception", "temporal", "risk", "navigation", "feedback", "evaluation"]
    for s in required_sections:
        assert s in config, f"Missing required config section: {s}"
    print("       Config validated successfully.")

    print("[2/6] Instantiating Perception modules...")
    preprocessor = FramePreprocessor()
    detector = YOLOObjectDetector()
    tracker = BoTSORTTracker()
    depth_estimator = DepthAnythingV2Estimator()
    print("       Perception modules instantiated.")

    print("[3/6] Instantiating Temporal modules...")
    history = RollingTrackHistory()
    velocity = NumericalVelocityEstimator()
    smoother = ExponentialMovingAverageSmoother()
    camera_motion = OpticalFlowMotionCompensator()
    print("       Temporal modules instantiated.")

    print("[4/6] Instantiating Risk modules...")
    ttc = AnalyticalTTCEstimator()
    features = RiskFeatureExtractor()
    uncertainty = UncertaintyEstimator()
    scorer = RiskScorer(config["risk"]["weights"], config["risk"]["score_thresholds"])
    risk_fsm = RiskStateMachine()
    print("       Risk modules instantiated.")

    print("[5/6] Instantiating Navigation modules...")
    zones = ZoneDivider()
    free_space = FreeSpaceAnalyzer()
    nav_engine = NavigationEngine()
    print("       Navigation modules instantiated.")

    print("[6/6] Instantiating Feedback & Evaluation modules...")
    tts = Pyttsx3SpeechEngine()
    msg_gen = MessageGenerator()
    warning_mgr = WarningManager(tts)
    logger = NavigationLogger(config["evaluation"]["logging"]["output_dir"])
    metrics_calc = MetricsCalculator()
    reporter = ReportGenerator()
    print("       Feedback and Evaluation modules instantiated.")

    print("=" * 60)
    print("ALL MODULE INTERFACES VERIFIED SUCCESSFULLY (STEP 1 PASS)")
    print("=" * 60)
    return True

def main():
    parser = argparse.ArgumentParser(description="Adaptive Edge-AI Navigation Prototype")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Path to test video file")
    parser.add_argument("--cam", type=int, default=None, help="Camera index")
    parser.add_argument("--debug", action="store_true", help="Enable visualization debug view")
    parser.add_argument("--dry-run", action="store_true", help="Run module instantiation validation")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        run_dry_run_validation(config)
        return

    print("System initialized. Run with --dry-run for Step 1 validation.")

if __name__ == "__main__":
    main()
