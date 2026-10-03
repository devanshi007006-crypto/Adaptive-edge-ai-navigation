"""
Unified Entrypoint for Adaptive Edge-AI Navigation System.
Supports Research, Demo, Real-World Testing, and Deployment Modes.
Complies with Step 19 specification.
"""

import os
import sys
import argparse
import yaml

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from adaptive_navigation.main import run_perception_pipeline, load_config

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Edge-AI Navigation System (Step 19 Unified CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["research", "demo", "real_world", "deployment"],
        default=None,
        help="Operational execution mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/deployment.yaml)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to test video file input"
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=None,
        help="USB camera index (e.g. 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override YOLO detection model path"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical preview window (headless)"
    )
    args = parser.parse_args()

    # Determine configuration file
    if args.config is not None:
        config_path = args.config
    elif args.mode is not None:
        mode_map = {
            "research": "configs/evaluation.yaml",
            "demo": "configs/development.yaml",
            "real_world": "configs/real_world.yaml",
            "deployment": "configs/deployment.yaml"
        }
        config_path = mode_map[args.mode]
    else:
        config_path = "configs/development.yaml"

    print("=" * 75)
    print("ADAPTIVE EDGE-AI NAVIGATION SYSTEM (STEP 19)")
    print(f"Loading Configuration: {config_path}")
    print("=" * 75)

    config = load_config(config_path)

    # Determine source override
    source = args.cam if args.cam is not None else args.video

    run_perception_pipeline(
        config=config,
        source_override=source,
        model_override=args.model,
        max_frames=args.max_frames,
        headless=args.headless or config.get("system", {}).get("headless", False)
    )

if __name__ == "__main__":
    main()
