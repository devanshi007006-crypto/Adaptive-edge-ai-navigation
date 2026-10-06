"""
Canonical Final Research Prototype Runner — Layer 1 + Layer 2 Monocular Edge-AI Navigation.

Full 15-Stage Pipeline:
Laptop Webcam / Video Stream -> YOLO26n Object Detection -> BoT-SORT Multi-Object Tracking
-> Depth Anything V2 TRT FP16 -> Adaptive Computation Controller -> Temporal Motion Estimation
-> 2D Lucas-Kanade Ego-Motion Compensation -> Scale-Invariant Disparity TTC -> Dynamic Risk Engine
-> Spatial Path Analysis -> User Navigation Command (FORWARD, LEFT, RIGHT, STOP)
-> Non-Blocking Audio Guidance -> Real-Time HUD & Telemetry.

Usage:
    python scripts/run/run_final_prototype.py [--video path/to/video.mp4] [--headless] [--max-frames N]
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adaptive_navigation.perception.camera import CameraSource, FramePacket
from adaptive_navigation.perception.detector import YOLOObjectDetector
from adaptive_navigation.perception.tracker import BoTSORTTracker
from adaptive_navigation.perception.depth import DepthAnythingV2Estimator
from adaptive_navigation.depth.adaptive_controller import AdaptiveComputationController
from adaptive_navigation.temporal.history import TemporalHistory, ObjectObservation
from adaptive_navigation.temporal.motion import MotionEstimator
from adaptive_navigation.temporal.camera_motion import CameraMotionEstimator
from adaptive_navigation.risk.ttc import TTCEstimator
from adaptive_navigation.risk.risk_engine import RiskEngine, RiskFeatures
from adaptive_navigation.uncertainty.reliability import ReliabilityEstimator
from adaptive_navigation.warning.state_machine import WarningStateMachine
from adaptive_navigation.warning.message_generator import WarningMessageGenerator
from adaptive_navigation.navigation.spatial import SpatialAnalyzer
from adaptive_navigation.navigation.path_geometry import PathGeometryAnalyzer
from adaptive_navigation.navigation.navigation_decision import NavigationEngine
from adaptive_navigation.audio.tts import TTSEngine
from adaptive_navigation.main import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FinalPrototype")

# User-Facing Command Mapping Contract
USER_COMMAND_MAP = {
    "CONTINUE": "FORWARD",
    "AVOID_LEFT": "LEFT",
    "AVOID_RIGHT": "RIGHT",
    "STOP": "STOP",
    "HAZARD_STOP": "STOP"
}


def run_prototype(
    video_path: Optional[str] = None,
    headless: bool = False,
    max_frames: Optional[int] = None,
    config_path: str = "configs/final.yaml"
) -> Dict[str, Any]:
    """Executes canonical final research prototype pipeline."""

    logger.info("Initializing Canonical Layer 1 + Layer 2 Research Prototype...")

    # Load configuration
    config = load_config(config_path)

    # Resolve primary detector path (YOLO26n primary)
    yolo26_path = REPO_ROOT / "models" / "detector" / "yolo26n.pt"
    if yolo26_path.exists():
        detector_model = str(yolo26_path)
    else:
        detector_model = config.get("detector", {}).get("model", "models/detector/yolo26n.pt")

    # Resolve depth engine path (TensorRT FP16 primary)
    trt_depth_path = REPO_ROOT / "models" / "deployment" / "depth_anything_v2_vits_fp16.engine"
    if trt_depth_path.exists():
        depth_checkpoint = str(trt_depth_path)
        logger.info(f"[Prototype] TensorRT FP16 Depth Engine Active: {depth_checkpoint}")
    else:
        depth_checkpoint = config.get("depth", {}).get("checkpoint", "models/depth/depth_anything_v2_vits.pth")

    # 1. Sensing Source
    if video_path and os.path.exists(video_path):
        logger.info(f"[Prototype] Video Source Active: {video_path}")
        cap = cv2.VideoCapture(video_path)
        is_live = False
    else:
        logger.info("[Prototype] Laptop Webcam Active (Device 0)...")
        cap = cv2.VideoCapture(0)
        is_live = True

    if not cap.isOpened():
        raise RuntimeError("ERROR: Failed to open video source or webcam.")

    # 2. Pipeline Modules Initialization
    detector = YOLOObjectDetector(
        model_name_or_path=detector_model,
        confidence_threshold=config.get("detector", {}).get("confidence_threshold", 0.28),
        iou_threshold=config.get("detector", {}).get("iou_threshold", 0.45),
        device="auto"
    )

    tracker = BoTSORTTracker()

    depth_estimator = DepthAnythingV2Estimator(
        checkpoint_path=depth_checkpoint,
        device="auto"
    )

    adaptive_controller = AdaptiveComputationController(config=config)
    temporal_history = TemporalHistory()
    motion_estimator = MotionEstimator()
    camera_motion_estimator = CameraMotionEstimator()
    ttc_estimator = TTCEstimator()
    risk_engine = RiskEngine(config.get("risk", {}))
    reliability_estimator = ReliabilityEstimator()
    warning_machine = WarningStateMachine(config.get("warning", {}))
    message_generator = WarningMessageGenerator(config)
    spatial_analyzer = SpatialAnalyzer(config.get("navigation", {}))
    path_analyzer = PathGeometryAnalyzer(config.get("navigation", {}))
    nav_engine = NavigationEngine(config.get("navigation", {}))
    tts_engine = TTSEngine(config=config)

    frame_idx = 0
    fps_buffer = []
    telemetry_records = []
    last_depth_result = None
    global_warn_state = "NO_WARNING"

    t_start_total = time.time()

    logger.info("Pipeline Ready. Starting Live Feed Processing Loop...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            t_loop_start = time.perf_counter()
            frame_ts = time.time() - t_start_total

            # Stage 1-2: Detection (YOLO26n Primary)
            raw_detections = detector.detect(frame, timestamp=frame_ts)
            active_detections = [d for d in raw_detections if d.policy_accepted]

            # Stage 3: Multi-Object Tracking (BoT-SORT)
            tracked_objects = tracker.update(active_detections, frame, frame_ts)

            # Stage 4: Adaptive Computation Evaluation & Depth Estimation
            max_risk = 0.0

            adaptive_decision = adaptive_controller.evaluate(
                frame_idx=frame_idx,
                global_warning_state=global_warn_state,
                max_risk_score=max_risk,
                has_approaching_objects=False,
                object_count=len(tracked_objects)
            )

            if adaptive_decision.should_run_depth or last_depth_result is None:
                depth_result = depth_estimator.estimate_depth(frame, frame_ts)
                last_depth_result = depth_result
            else:
                depth_result = last_depth_result

            # Stage 5: Extract Object-Level Depths & Update Temporal History
            object_depths = depth_estimator.extract_all_object_depths(depth_result, tracked_objects)
            observations = [
                ObjectObservation.from_tracked_depth(obj, frame_ts, frame_idx)
                for obj in object_depths
            ]
            temporal_history.update(observations, current_timestamp=frame_ts)

            # Stage 6: Motion Estimation
            motion_estimates = motion_estimator.estimate_all(temporal_history)

            # Stage 7: 2D Lucas-Kanade Ego-Motion Compensation
            obstacle_bboxes = [obj.bbox for obj in tracked_objects]
            cam_motion = camera_motion_estimator.estimate(frame, timestamp=frame_ts, object_bboxes=obstacle_bboxes)
            compensated_estimates = camera_motion_estimator.compensate_all(motion_estimates, cam_motion)

            # Stage 8: Scale-Invariant Disparity TTC
            ttc_results = ttc_estimator.estimate_all(temporal_history, compensated_estimates)

            # Stage 9: Dynamic Risk Features & Risk Assessment Engine
            fh, fw = frame.shape[:2]
            risk_features_map = {}
            for obj in tracked_objects:
                tid = obj.track_id
                comp_m = compensated_estimates.get(tid)
                ttc_r = ttc_results.get(tid)
                od = next((d for d in object_depths if d.track_id == tid), None)
                risk_features_map[tid] = RiskFeatures(
                    track_id=tid,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    depth_value=od.depth_value if od else None,
                    depth_type="metric" if (od and od.is_metric) else "relative",
                    depth_valid=od.depth_valid if od else False,
                    depth_reliability=od.depth_reliability if od else "INVALID",
                    raw_velocity=(comp_m.raw_vx if comp_m else None, comp_m.raw_vy if comp_m else None),
                    compensated_velocity=(comp_m.compensated_vx if comp_m else None, comp_m.compensated_vy if comp_m else None),
                    compensated_speed=comp_m.compensated_speed if comp_m else None,
                    approach_state=comp_m.approach_state if comp_m else "UNKNOWN",
                    motion_reliability=comp_m.reliability if comp_m else "UNKNOWN",
                    ttc_seconds=ttc_r.ttc_seconds if ttc_r else None,
                    ttc_state=ttc_r.ttc_state if ttc_r else "UNKNOWN",
                    ttc_valid=ttc_r.ttc_valid if ttc_r else False,
                    bbox=obj.bbox,
                    center_x=obj.center_x,
                    center_y=obj.center_y,
                    object_width=obj.width,
                    object_height=obj.height,
                    frame_width=fw,
                    frame_height=fh,
                    camera_motion_valid=cam_motion.valid,
                )

            risk_assessments = risk_engine.assess_all(risk_features_map)

            # Stage 10: Reliability Assessment
            obs_map = {obj.track_id: temporal_history.get(obj.track_id) for obj in object_depths}
            reliability_assessments = reliability_estimator.assess_all(
                observations_map=obs_map,
                compensated_motion_map=compensated_estimates,
                camera_motion=cam_motion,
                ttc_map=ttc_results,
                risk_map=risk_assessments,
            )

            # Stage 11: Temporal Warning State Machine
            warning_decisions, global_warning = warning_machine.update(
                risk_assessments=risk_assessments,
                reliability_assessments=reliability_assessments,
                ttc_results=ttc_results,
                compensated_motion=compensated_estimates,
                timestamp=frame_ts,
                frame_index=frame_idx,
            )
            global_warn_state = global_warning.state

            # Stage 12: Spatial Corridor Analysis & Navigation Decision
            spatial_reprs = {
                obj.track_id: spatial_analyzer.analyze_object(
                    track_id=obj.track_id,
                    bbox=obj.bbox,
                    frame_width=fw,
                    frame_height=fh,
                    horizontal_motion=compensated_estimates.get(obj.track_id).compensated_vx if obj.track_id in compensated_estimates else None,
                    vertical_motion=compensated_estimates.get(obj.track_id).compensated_vy if obj.track_id in compensated_estimates else None,
                )
                for obj in tracked_objects
            }
            path_overlaps = path_analyzer.assess_all(spatial_reprs)
            nav_decisions, scene_nav = nav_engine.evaluate(
                spatial_objects=spatial_reprs,
                path_assessments=path_overlaps,
                warning_decisions=warning_decisions,
                global_warning=global_warning
            )

            user_cmd = USER_COMMAND_MAP.get(scene_nav.navigation_state, "FORWARD")

            # Stage 13: Spoken Warning & Audio Feedback
            warning_msg = message_generator.generate(
                global_warning=global_warning,
                track_decisions=warning_decisions,
                current_time=frame_ts,
                scene_nav=scene_nav
            )

            if warning_msg.should_speak and warning_msg.text:
                tts_engine.speak(warning_msg.text, priority=warning_msg.priority)

            t_loop_end = time.perf_counter()
            lat_ms = (t_loop_end - t_loop_start) * 1000.0
            fps = 1.0 / max(1e-6, t_loop_end - t_loop_start)
            fps_buffer.append(fps)

            # Stage 14: Telemetry Record
            record = {
                "frame_idx": frame_idx,
                "timestamp": round(frame_ts, 3),
                "detector": "YOLO26n",
                "objects_count": len(object_depths),
                "global_risk": global_warning.state,
                "nav_internal": scene_nav.navigation_state,
                "user_command": user_cmd,
                "computation_mode": adaptive_decision.computation_mode,
                "depth_reused": adaptive_decision.depth_reused,
                "latency_ms": round(lat_ms, 2),
                "fps": round(fps, 2)
            }
            telemetry_records.append(record)

            # Stage 15: Visual Research HUD
            if not headless:
                hud = frame.copy()
                color = (0, 255, 0) if global_warning.state == "NO_WARNING" else ((0, 165, 255) if global_warning.state == "CAUTION" else (0, 0, 255))
                cv2.putText(hud, f"YOLO26n PROTOTYPE | FPS: {fps:.1f} | Latency: {lat_ms:.1f}ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(hud, f"RISK: [{global_warning.state}] | MODE: [{adaptive_decision.computation_mode}]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(hud, f"NAV COMMAND: [{user_cmd}] (Internal: {scene_nav.navigation_state})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
                
                # Draw tracked bounding boxes
                for obj in object_depths:
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                    cv2.rectangle(hud, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(hud, f"{obj.class_name.upper()} #{obj.track_id} ({obj.depth_value:.2f})", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                cv2.imshow("Layer 1 + Layer 2 Research Prototype (YOLO26n)", hud)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    mean_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0
    mean_lat = float(np.mean([r["latency_ms"] for r in telemetry_records])) if telemetry_records else 0.0

    summary = {
        "status": "SUCCESS",
        "detector": "YOLO26n",
        "total_frames": frame_idx,
        "mean_fps": round(mean_fps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "telemetry_records_count": len(telemetry_records)
    }

    logger.info(f"[Prototype Complete] Processed {frame_idx} frames at {mean_fps:.2f} FPS ({mean_lat:.2f} ms latency).")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Canonical Final Research Prototype Runner (YOLO26n Primary)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to webcam)")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV window rendering")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    res = run_prototype(video_path=args.video, headless=args.headless, max_frames=args.max_frames)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
