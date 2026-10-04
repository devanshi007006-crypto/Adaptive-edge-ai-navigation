"""Main pipeline orchestrator for the Adaptive Edge-AI Navigation System."""
import argparse
import time
import sys
import os
import json
import csv
import yaml
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from risk import TTCEstimator, TTCResult, RiskEngine, RiskFeatures, RiskAssessment
from uncertainty.reliability import ReliabilityEstimator, ReliabilityAssessment, SystemReliability
from warning.state_machine import WarningStateMachine, WarningDecision, GlobalWarningDecision
from warning.message_generator import WarningMessageGenerator, WarningMessage
from audio.tts import TTSEngine
from navigation.spatial import SpatialAnalyzer
from navigation.path_geometry import PathGeometryAnalyzer
from navigation.navigation_decision import NavigationEngine
from temporal import (
    TemporalHistory,
    ObjectObservation,
    MotionEstimator,
    MotionEstimate,
    CameraMotionEstimator,
    CameraMotionEstimate,
    CompensatedMotionEstimate,
)
from perception import (
    CameraSource,
    FramePacket,
    FramePreprocessor,
    YOLOObjectDetector,
    Detection,
    BoTSORTTracker,
    TrackedObject,
    DepthAnythingV2Estimator,
    DepthResult,
    TrackedObjectDepth,
)

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

def run_perception_pipeline(
    config: dict,
    source_override=None,
    model_override=None,
    device_override=None,
    max_frames: int = None,
    headless: bool = False,
    telemetry_dir: str = None,
    telemetry_prefix: str = "real_telemetry",
    depth_cadence_override: int = None,
) -> None:
    """Executes Step 13: Camera -> Frame Validation -> YOLO Detection -> BoT-SORT Tracking -> Depth Anything V2 -> Object-Level Depth -> Temporal History -> Motion Estimation -> Camera Motion Compensation -> TTC Estimation -> Multi-Factor Risk Assessment -> Uncertainty & Reliability -> Warning Decision State Machine -> User-Facing Audio/TTS Layer."""
    cam_cfg = config.get("camera", {})
    source = source_override if source_override is not None else cam_cfg.get("source", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 30)

    # Resolve Canonical Execution Device: CUDA default if available with CPU fallback
    canonical_device = device_override if device_override is not None else config.get("system", {}).get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if canonical_device == "auto":
        canonical_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    det_cfg = config.get("detector", {})
    model_name = model_override if model_override is not None else det_cfg.get("model", "models/detector/yolo11n.pt")
    conf_thresh = det_cfg.get("confidence_threshold", 0.25)
    iou_thresh = det_cfg.get("iou_threshold", 0.45)
    img_size = det_cfg.get("image_size", 640)
    det_device = device_override if device_override is not None else det_cfg.get("device", canonical_device)

    track_cfg = config.get("tracker", {})
    tracker_config = track_cfg.get("tracker_config", "botsort.yaml")
    track_high = track_cfg.get("track_high_thresh", 0.25)
    track_low = track_cfg.get("track_low_thresh", 0.1)
    new_track = track_cfg.get("new_track_thresh", 0.25)
    match_thresh = track_cfg.get("match_thresh", 0.8)
    track_buffer = track_cfg.get("track_buffer", 30)

    depth_cfg = config.get("depth", {})
    depth_checkpoint = depth_cfg.get("checkpoint", "models/depth/depth_anything_v2_vits.pth")
    depth_type = depth_cfg.get("model_type", "vits")
    depth_device = device_override if device_override is not None else depth_cfg.get("device", canonical_device)
    depth_input_size = depth_cfg.get("input_size", 518)
    is_metric = depth_cfg.get("is_metric", False)
    obj_stat = depth_cfg.get("object_statistic", "median")
    depth_cadence = depth_cadence_override if depth_cadence_override is not None else int(depth_cfg.get("cadence", 1))
    depth_cadence = max(1, depth_cadence)

    temporal_cfg = config.get("temporal", {})
    history_length = temporal_cfg.get("history_length", 30)
    max_history_age = temporal_cfg.get("max_history_age_seconds", 2.0)
    cleanup_after = temporal_cfg.get("cleanup_after_seconds", 2.0)
    min_obs = temporal_cfg.get("minimum_observations", 3)

    motion_cfg = config.get("motion", {})
    min_dt = motion_cfg.get("minimum_dt_seconds", 0.01)
    max_gap = motion_cfg.get("max_valid_time_gap_seconds", 0.5)
    motion_min_obs = motion_cfg.get("minimum_history_observations", 3)
    smoothing_method = motion_cfg.get("smoothing_method", "ema")
    smoothing_window = motion_cfg.get("smoothing_window", 5)
    stable_thresh = motion_cfg.get("stable_threshold", 0.05)
    min_depth_rel = motion_cfg.get("minimum_depth_reliability", "MEDIUM")
    depth_convention = motion_cfg.get("depth_convention", "higher_is_closer")

    cam_motion_cfg = config.get("camera_motion", {})
    cam_motion_enabled = cam_motion_cfg.get("enabled", True)
    cam_method = cam_motion_cfg.get("method", "sparse_optical_flow")
    cam_max_feat = cam_motion_cfg.get("max_features", 300)
    cam_qual = cam_motion_cfg.get("quality_level", 0.01)
    cam_min_dist = cam_motion_cfg.get("min_distance", 7.0)
    cam_ransac = cam_motion_cfg.get("ransac_enabled", True)
    cam_min_feat = cam_motion_cfg.get("minimum_features", 20)
    cam_inlier_th = cam_motion_cfg.get("inlier_threshold", 3.0)
    cam_max_dt = cam_motion_cfg.get("max_dt_seconds", 0.5)
    show_flow = cam_motion_cfg.get("show_flow", False)

    ttc_cfg = config.get("ttc", {})
    ttc_enabled = ttc_cfg.get("enabled", True)
    ttc_min_obs = ttc_cfg.get("minimum_history_observations", 3)
    ttc_min_close_spd = ttc_cfg.get("minimum_closing_speed", 0.05)
    ttc_max_gap = ttc_cfg.get("maximum_time_gap_seconds", 0.5)
    ttc_max_sec = ttc_cfg.get("max_ttc_seconds", 30.0)

    risk_cfg = config.get("risk", {})
    risk_enabled = risk_cfg.get("enabled", True)
    risk_weights = risk_cfg.get("weights")
    risk_score_th = risk_cfg.get("score_thresholds")
    risk_ttc_th = risk_cfg.get("ttc_thresholds")
    risk_path_cfg = risk_cfg.get("path")
    risk_min_cov = risk_cfg.get("minimum_evidence_coverage", 0.50)
    risk_class_w = risk_cfg.get("class_weights")

    debug_cfg = config.get("debug", {})
    display_enabled = debug_cfg.get("display", True) and not headless
    show_trails = debug_cfg.get("show_trails", True)
    show_depth_inset = debug_cfg.get("show_depth_inset", True)
    window_name = debug_cfg.get("window_name", "Adaptive Navigation - User-Facing Warning & TTS Layer")

    print("=" * 75)
    print("STEP 13: CAMERA + YOLO + BoT-SORT + DEPTH + TEMPORAL + MOTION + CAM COMP + TTC + RISK + UNCERTAINTY + WARNING + TTS")
    print("=" * 75)
    print(f"Camera Source:   {source} ({width}x{height} @ {fps} FPS)")
    print(f"YOLO Detector:   {model_name} | Conf: {conf_thresh} | Device: {det_device}")
    print(f"Tracker:         BoT-SORT (buffer={track_buffer}, match={match_thresh})")
    print(f"Depth Model:     Depth Anything V2 ({depth_type}) | Metric: {is_metric} | Stat: {obj_stat} | Cadence: {depth_cadence}:1")
    print(f"Temporal Buffer: MaxLen: {history_length} | MaxAge: {max_history_age}s | Cleanup: {cleanup_after}s | MinObs: {min_obs}")
    print(f"Motion Estimator: Method: {smoothing_method} (win={smoothing_window}) | StableThresh: {stable_thresh} | Convention: {depth_convention}")
    print(f"Camera Motion:    Enabled: {cam_motion_enabled} | Method: {cam_method} | MaxFeat: {cam_max_feat} | RANSAC: {cam_ransac}")
    print(f"TTC Estimator:    Enabled: {ttc_enabled} | MinClosingSpd: {ttc_min_close_spd} | MaxTTC: {ttc_max_sec}s")
    print(f"Risk Engine:      Enabled: {risk_enabled} | MinCoverage: {risk_min_cov} | PathCorridor: {risk_path_cfg.get('corridor_width_ratio', 0.4) if risk_path_cfg else 0.4}")
    print(f"Display Mode:    {'Active Window' if display_enabled else 'Headless'}")
    print("Press 'q' in preview window or Ctrl+C in terminal to stop.")
    print("-" * 75)

    # 1. Initialize Camera
    preprocessor = FramePreprocessor()
    camera = CameraSource(source=source, width=width, height=height, target_fps=fps)
    try:
        camera.open()
    except Exception as e:
        print(f"\nERROR: Unable to open camera source: {e}")
        return

    # 2. Initialize Detector
    try:
        detector = YOLOObjectDetector(
            model_name_or_path=model_name,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            image_size=img_size,
            device=det_device,
            class_filter_config=det_cfg.get("class_filter"),
        )
    except Exception as e:
        camera.release()
        print(f"\nERROR: Failed to initialize detector: {e}")
        return

    # 3. Initialize BoT-SORT Tracker
    try:
        tracker = BoTSORTTracker(
            tracker_config=tracker_config,
            track_high_thresh=track_high,
            track_low_thresh=track_low,
            new_track_thresh=new_track,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
        )
    except Exception as e:
        camera.release()
        print(f"\nERROR: Failed to initialize BoT-SORT tracker: {e}")
        return

    # 4. Initialize Depth Anything V2 Estimator
    try:
        depth_estimator = DepthAnythingV2Estimator(
            checkpoint_path=depth_checkpoint,
            model_type=depth_type,
            device=depth_device,
            input_size=depth_input_size,
            is_metric=is_metric,
            object_statistic=obj_stat,
        )
    except Exception as e:
        camera.release()
        print(f"\nERROR: Failed to initialize Depth Anything V2: {e}")
        return

    # 5. Initialize Temporal History Buffer
    temporal_history = TemporalHistory(
        history_length=history_length,
        max_history_age_seconds=max_history_age,
        cleanup_after_seconds=cleanup_after,
        minimum_observations=min_obs,
    )

    # 6. Initialize Motion Estimator (Step 7)
    motion_estimator = MotionEstimator(
        minimum_dt_seconds=min_dt,
        max_valid_time_gap_seconds=max_gap,
        minimum_history_observations=motion_min_obs,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window,
        stable_threshold=stable_thresh,
        minimum_depth_reliability=min_depth_rel,
        depth_convention=depth_convention,
        temporal_stabilization_config=motion_cfg.get("temporal_stabilization"),
    )

    # 7. Initialize Camera Motion Estimator (Step 8)
    camera_motion_estimator = CameraMotionEstimator(
        enabled=cam_motion_enabled,
        method=cam_method,
        max_features=cam_max_feat,
        quality_level=cam_qual,
        min_distance=cam_min_dist,
        ransac_enabled=cam_ransac,
        minimum_features=cam_min_feat,
        inlier_threshold=cam_inlier_th,
        max_dt_seconds=cam_max_dt,
        forward_compensation_config=cam_motion_cfg.get("forward_compensation"),
    )

    # 8. Initialize TTC Estimator (Step 9)
    ttc_estimator = TTCEstimator(
        enabled=ttc_enabled,
        minimum_history_observations=ttc_min_obs,
        minimum_closing_speed=ttc_min_close_spd,
        maximum_time_gap_seconds=ttc_max_gap,
        max_ttc_seconds=ttc_max_sec,
        depth_convention=depth_convention,
    )

    # 10. Initialize Uncertainty & Reliability Estimator (Step 11)
    unc_cfg = config.get("uncertainty", {})
    unc_enabled = unc_cfg.get("enabled", True)
    unc_weights = unc_cfg.get("weights", None)
    unc_thresholds = unc_cfg.get("thresholds", None)
    unc_consistency = unc_cfg.get("consistency", {}).get("enabled", True)

    reliability_estimator = ReliabilityEstimator(
        enabled=unc_enabled,
        weights=unc_weights,
        thresholds=unc_thresholds,
        consistency_enabled=unc_consistency,
    )
    print(f"Uncertainty:      Enabled: {unc_enabled} | Consistency: {unc_consistency} | Weights: {list(unc_weights.keys()) if unc_weights else 'default'}")

    # 11. Initialize Warning Decision State Machine (Step 12)
    warning_cfg = config.get("warning", {})
    warning_enabled = warning_cfg.get("enabled", True)
    warning_state_machine = WarningStateMachine(warning_cfg)
    print(f"Warning Machine:  Enabled: {warning_enabled} | HistLen: {warning_cfg.get('history_length', 10)} | Grace: {warning_cfg.get('lost_track_grace_seconds', 0.5)}s")

    # 12. Initialize Spatial Navigation & Path Engine (Step 14)
    nav_cfg = config.get("navigation", {})
    spatial_analyzer = SpatialAnalyzer(nav_cfg)
    path_analyzer = PathGeometryAnalyzer(nav_cfg)
    nav_engine = NavigationEngine(nav_cfg)
    print(f"Navigation:       Enabled: True | DirHysteresis: {nav_cfg.get('direction_change_frames', 3)}")

    # 13. Initialize Warning Message Generator & Audio/TTS Engine (Step 13)
    message_generator = WarningMessageGenerator(config)
    tts_engine = TTSEngine(config)
    print(f"Message Gen:      RepeatInterval: {config.get('audio', {}).get('repeat_interval_seconds', 2.0)}s | ContinuousGuidance: True")
    print(f"TTS Engine:       Backend: {tts_engine.active_backend} | Available: {tts_engine.is_available()}")

    # 9. Initialize Multi-Factor Risk Assessment Engine (Step 10)
    risk_engine = RiskEngine(
        enabled=risk_enabled,
        weights=risk_weights,
        score_thresholds=risk_score_th,
        ttc_thresholds=risk_ttc_th,
        path_config=risk_path_cfg,
        minimum_evidence_coverage=risk_min_cov,
        class_weights=risk_class_w,
    )

    frames_processed = 0
    loop_times = []
    last_loop_time = time.perf_counter()
    telemetry_records = []
    cached_depth_result: Optional[DepthResult] = None

    def get_color_for_id(track_id: int) -> tuple:
        b = (track_id * 67 + 50) % 205 + 50
        g = (track_id * 131 + 80) % 205 + 50
        r = (track_id * 193 + 110) % 205 + 50
        return (int(b), int(g), int(r))

    try:
        while True:
            packet: FramePacket = camera.read_frame()

            if packet is None:
                if camera.is_video_file:
                    print("\nEnd of video stream reached.")
                else:
                    print("\nWarning: Failed to capture frame from camera.")
                break

            # Frame validation
            if not preprocessor.validate_frame(packet.frame):
                print(f"Warning: Dropped invalid frame at index {packet.frame_index}")
                continue

            t_frame_start = time.perf_counter()

            # Stage 1: Run YOLO Object Detection (raw detections preserved)
            detections: list[Detection] = detector.detect(packet.frame, packet.timestamp)
            det_latency_ms = detector.last_inference_latency_ms

            # Stage 2: Separate active navigation hazards from policy-filtered detections
            active_detections = [d for d in detections if d.policy_accepted]
            filtered_detections = [d for d in detections if not d.policy_accepted]

            # Run BoT-SORT Tracking on active hazard detections
            tracked_objects: list[TrackedObject] = tracker.update(active_detections, packet.frame, packet.timestamp)
            track_latency_ms = tracker.last_tracker_latency_ms

            # Stage 3: Run Depth Anything V2 (Step 6 / Adaptive Depth Cadence)
            is_depth_frame = (frames_processed % depth_cadence == 0) or (cached_depth_result is None)
            if is_depth_frame:
                depth_result: DepthResult = depth_estimator.estimate_depth(packet.frame, packet.timestamp)
                depth_latency_ms = depth_estimator.last_inference_latency_ms
                cached_depth_result = depth_result
            else:
                depth_latency_ms = 0.0
                depth_result = DepthResult(
                    depth_map=cached_depth_result.depth_map,
                    is_metric=cached_depth_result.is_metric,
                    min_depth=cached_depth_result.min_depth,
                    max_depth=cached_depth_result.max_depth,
                    timestamp=packet.timestamp,
                )

            # Stage 4: Extract Object-Level Depth for each tracked obstacle
            object_depths: list[TrackedObjectDepth] = depth_estimator.extract_all_object_depths(depth_result, tracked_objects)

            # Stage 5: Update Temporal History Buffer (strictly observation records, no velocity/TTC yet)
            observations = [
                ObjectObservation.from_tracked_depth(obj, packet.timestamp, packet.frame_index)
                for obj in object_depths
            ]
            temporal_history.update(observations, current_timestamp=packet.timestamp)

            # Stage 6: Estimate Velocity, Depth Rate, Approach State, and Temporal Smoothing
            t_motion_start = time.perf_counter()
            motion_estimates: dict[int, MotionEstimate] = motion_estimator.estimate_all(temporal_history)
            motion_latency_ms = (time.perf_counter() - t_motion_start) * 1000.0

            # Stage 7: Estimate Global Camera Motion & Compensate Object Velocity (Step 8)
            t_cam_start = time.perf_counter()
            obstacle_bboxes = [obj.bbox for obj in tracked_objects]
            camera_motion: CameraMotionEstimate = camera_motion_estimator.estimate(
                packet.frame,
                timestamp=packet.timestamp,
                object_bboxes=obstacle_bboxes,
            )
            compensated_estimates: dict[int, CompensatedMotionEstimate] = camera_motion_estimator.compensate_all(
                motion_estimates,
                camera_motion,
            )
            cam_latency_ms = (time.perf_counter() - t_cam_start) * 1000.0

            # Stage 8: Estimate Time-to-Collision (TTC) for each Track ID (Step 9)
            t_ttc_start = time.perf_counter()
            ttc_results: dict[int, TTCResult] = ttc_estimator.estimate_all(
                temporal_history,
                compensated_estimates,
            )
            ttc_latency_ms = (time.perf_counter() - t_ttc_start) * 1000.0

            # Stage 9: Multi-Factor Risk Assessment Engine (Step 10)
            t_risk_start = time.perf_counter()
            fh, fw = packet.frame.shape[:2]
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
                    camera_motion_valid=camera_motion.valid,
                )
            risk_assessments: dict[int, RiskAssessment] = risk_engine.assess_all(risk_features_map)
            risk_latency_ms = (time.perf_counter() - t_risk_start) * 1000.0

            # Stage 10: Uncertainty & Reliability Estimation Layer (Step 11)
            t_rel_start = time.perf_counter()
            obs_map = {obj.track_id: temporal_history.get(obj.track_id) for obj in object_depths}
            reliability_assessments: dict[int, ReliabilityAssessment] = reliability_estimator.assess_all(
                observations_map=obs_map,
                compensated_motion_map=compensated_estimates,
                camera_motion=camera_motion,
                ttc_map=ttc_results,
                risk_map=risk_assessments,
            )
            system_reliability = reliability_estimator.assess_system(
                camera_healthy=(packet is not None and packet.frame is not None),
                fps=overall_fps if 'overall_fps' in locals() else 0.0,
                detector_ok=True,
                depth_model_ok=True,
                camera_motion=camera_motion,
                active_tracks=len(object_depths),
            )
            rel_latency_ms = (time.perf_counter() - t_rel_start) * 1000.0

            # Stage 11: Temporal Risk Stabilization & Warning Decision (Step 12)
            t_warn_start = time.perf_counter()
            warning_decisions, global_warning = warning_state_machine.update(
                risk_assessments=risk_assessments,
                reliability_assessments=reliability_assessments,
                ttc_results=ttc_results,
                compensated_motion=compensated_estimates,
                timestamp=packet.timestamp,
                frame_index=packet.frame_index,
            )
            warn_latency_ms = (time.perf_counter() - t_warn_start) * 1000.0

            # Stage 12: Spatial Navigation & Path Analysis (Step 14)
            t_nav_start = time.perf_counter()
            spatial_reprs = {
                obj.track_id: spatial_analyzer.analyze_object(
                    track_id=obj.track_id,
                    bbox=obj.bbox,
                    frame_width=packet.frame.shape[1],
                    frame_height=packet.frame.shape[0],
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
                global_warning=global_warning,
                system_reliability_score=system_reliability.system_score if hasattr(system_reliability, 'system_score') else 1.0,
            )
            nav_latency_ms = (time.perf_counter() - t_nav_start) * 1000.0

            # Stage 13: Continuous User-Facing Warning Message Generation & Audio/TTS (Step 13 & 14)
            t_tts_start = time.perf_counter()
            warning_message: WarningMessage = message_generator.generate(
                global_warning=global_warning,
                track_decisions=warning_decisions,
                current_time=packet.timestamp,
                scene_nav=scene_nav,
            )
            if warning_message.should_speak and tts_engine.is_available():
                tts_engine.speak(warning_message.text, priority=warning_message.priority)
            tts_latency_ms = (time.perf_counter() - t_tts_start) * 1000.0

            t_frame_end = time.perf_counter()
            total_frame_latency_ms = (t_frame_end - t_frame_start) * 1000.0

            if telemetry_dir is not None:
                telemetry_records.append({
                    "frame_index": int(packet.frame_index),
                    "timestamp_sec": round(float(packet.timestamp), 4),
                    "raw_detections_count": len(detections),
                    "active_detections_count": len(active_detections),
                    "filtered_detections_count": len(filtered_detections),
                    "active_tracks_count": len(object_depths),
                    "raw_detections": [
                        {
                            "class_name": d.class_name,
                            "confidence": round(float(d.confidence), 3),
                            "bbox": [round(float(v), 1) for v in d.bbox],
                            "policy_accepted": bool(d.policy_accepted),
                            "filter_reason": d.filter_reason,
                        }
                        for d in detections
                    ],
                    "latencies_ms": {
                        "detector": round(float(det_latency_ms), 2),
                        "tracker": round(float(track_latency_ms), 2),
                        "depth": round(float(depth_latency_ms), 2),
                        "motion": round(float(motion_latency_ms), 2),
                        "camera_motion": round(float(cam_latency_ms), 2),
                        "ttc": round(float(ttc_latency_ms), 2),
                        "risk": round(float(risk_latency_ms), 2),
                        "reliability": round(float(rel_latency_ms), 2),
                        "warning": round(float(warn_latency_ms), 2),
                        "navigation": round(float(nav_latency_ms), 2),
                        "tts": round(float(tts_latency_ms), 2),
                        "total_frame": round(float(total_frame_latency_ms), 2),
                    },
                    "camera_motion": {
                        "dx": round(float(camera_motion.dx), 2),
                        "dy": round(float(camera_motion.dy), 2),
                        "valid": bool(camera_motion.valid),
                    },
                    "device": {
                        "canonical": canonical_device,
                        "detector": detector.resolved_device,
                        "depth": depth_estimator.resolved_device,
                        "cuda_available": torch.cuda.is_available(),
                        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                        "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if torch.cuda.is_available() else 0.0,
                        "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2) if torch.cuda.is_available() else 0.0,
                    },
                    "system_health": system_reliability.system_status if hasattr(system_reliability, 'system_status') else "UNKNOWN",
                    "global_warning": {
                        "state": global_warning.state,
                        "selected_track_id": global_warning.selected_track_id,
                        "reason": global_warning.reason,
                    },
                    "navigation": {
                        "state": scene_nav.navigation_state,
                        "safe_direction": scene_nav.safe_direction,
                        "reason": scene_nav.reason,
                    },
                    "audio": {
                        "spoken": bool(warning_message.should_speak),
                        "text": str(warning_message.text) if warning_message.should_speak else "",
                        "priority": str(warning_message.priority),
                    },
                    "objects": [
                        {
                            "track_id": int(o.track_id),
                            "class_name": str(o.class_name),
                            "confidence": round(float(o.confidence), 3),
                            "bbox": [round(float(v), 1) for v in o.bbox],
                            "depth_value": round(float(o.depth_value), 3) if o.depth_value is not None else None,
                            "depth_valid": bool(o.depth_valid),
                            "approach_state": str(comp_m.approach_state) if comp_m else "UNKNOWN",
                            "closing_speed": round(float(ttc_r.closing_speed), 3) if (ttc_r and ttc_r.closing_speed is not None) else None,
                            "ttc_seconds": round(float(ttc_r.ttc_seconds), 2) if (ttc_r and ttc_r.ttc_seconds is not None) else None,
                            "ttc_state": str(ttc_r.ttc_state) if ttc_r else "UNKNOWN",
                            "ttc_valid": bool(ttc_r.ttc_valid) if ttc_r else False,
                            "risk_score": round(float(risk_ass.risk_score), 3) if risk_ass else 0.0,
                            "risk_level": str(risk_ass.risk_level) if risk_ass else "UNKNOWN",
                            "risk_primary_reason": str(risk_ass.primary_reason) if risk_ass else "",
                            "warning_state": str(warn_dec.state) if warn_dec else "NO_WARNING",
                        }
                        for o, comp_m, ttc_r, risk_ass, warn_dec in [
                            (
                                item,
                                compensated_estimates.get(item.track_id),
                                ttc_results.get(item.track_id),
                                risk_assessments.get(item.track_id),
                                warning_decisions.get(item.track_id),
                            )
                            for item in object_depths
                        ]
                    ],
                })

            frames_processed += 1

            # Overall loop FPS
            t_now = time.perf_counter()
            dt_loop = t_now - last_loop_time
            last_loop_time = t_now
            instant_loop_fps = (1.0 / dt_loop) if dt_loop > 0 else 0.0
            loop_times.append(instant_loop_fps)
            if len(loop_times) > 30:
                loop_times.pop(0)
            overall_fps = sum(loop_times) / len(loop_times)

            # Periodic console report
            if frames_processed % 10 == 0 or frames_processed == 1:
                def _obj_summary(o):
                    w_item = warning_decisions.get(o.track_id)
                    r_item = risk_assessments.get(o.track_id)
                    rel_item = reliability_assessments.get(o.track_id)
                    w_str = f"Warn={w_item.state}" if w_item else "Warn=N/A"
                    r_str = f"Risk={r_item.risk_level}({r_item.risk_score:.2f})" if r_item else "Risk=N/A"
                    rel_str = f"Rel={rel_item.reliability_level}({rel_item.reliability_score:.2f})" if rel_item else "Rel=N/A"
                    return f"ID:{o.track_id}({o.class_name})[{w_str} | {r_str} | {rel_str}]"

                depth_info = ", ".join(_obj_summary(o) for o in object_depths) if object_depths else "No active tracks"
                speech_info = f"TTS: \"{warning_message.text}\"" if warning_message.should_speak else "TTS: (silent)"
                print(
                    f"[Frame {packet.frame_index:05d}] "
                    f"Loop FPS: {overall_fps:4.1f} | "
                    f"Det: {det_latency_ms:4.0f}ms | "
                    f"Track: {track_latency_ms:3.0f}ms | "
                    f"Depth: {depth_latency_ms:4.0f}ms | "
                    f"Risk: {risk_latency_ms:3.1f}ms | "
                    f"Rel: {rel_latency_ms:3.1f}ms | "
                    f"Warn: {warn_latency_ms:3.1f}ms | "
                    f"TTS: {tts_latency_ms:3.1f}ms | "
                    f"Alert: [{global_warning.state}] | {speech_info} | "
                    f"Nav: [{scene_nav.navigation_state} | SafeDir:{scene_nav.safe_direction}] | "
                    f"Objects ({len(object_depths)}): [{depth_info}]"
                )

            # Debug visual overlay
            if display_enabled:
                display_frame = packet.frame.copy()

                # Visual trails
                if show_trails:
                    for tid, trail in tracker.debug_trails.items():
                        pts = list(trail)
                        color = get_color_for_id(tid)
                        for k in range(1, len(pts)):
                            cv2.line(display_frame, pts[k - 1], pts[k], color, 2)

                # Annotate tracked objects with ID, Class, Confidence, and Relative Depth
                for obj in object_depths:
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                    color = get_color_for_id(obj.track_id)

                    # cv2.rectangle replaced below

                    depth_str = f"{obj.depth_value:.2f} (rel)" if obj.depth_valid else "N/A"
                    label = f"ID: {obj.track_id} | {obj.class_name} | depth: {depth_str}"
                    hist_len = temporal_history.get_length(obj.track_id)
                    comp_m = compensated_estimates.get(obj.track_id)
                    ttc_res = ttc_results.get(obj.track_id)
                    if ttc_res and ttc_res.ttc_valid and ttc_res.ttc_seconds is not None:
                        ttc_str = f"TTC: {ttc_res.ttc_seconds:.1f}s"
                    elif ttc_res:
                        ttc_str = f"TTC: {ttc_res.ttc_state}"
                    else:
                        ttc_str = "TTC: UNKNOWN"

                    risk_ass = risk_assessments.get(obj.track_id)
                    level_color_map = {
                        "CRITICAL": (0, 0, 255),    # Red
                        "HIGH": (0, 140, 255),      # Orange
                        "MEDIUM": (0, 255, 255),    # Yellow
                        "LOW": (0, 255, 0),         # Green
                        "UNKNOWN": (200, 200, 200), # Gray
                    }
                    risk_color = level_color_map.get(risk_ass.risk_level if risk_ass else "UNKNOWN", color)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), risk_color, 2)

                    warn_dec = warning_decisions.get(obj.track_id)
                    warn_color_map = {
                        "CRITICAL": (0, 0, 255),    # Red
                        "WARNING": (0, 140, 255),   # Orange
                        "CAUTION": (0, 255, 255),   # Yellow
                        "NO_WARNING": (0, 255, 0),  # Green
                        "UNKNOWN": (200, 200, 200), # Gray
                    }
                    warn_color = warn_color_map.get(warn_dec.state if warn_dec else "UNKNOWN", color)

                    # Draw outer box colored by stabilized warning state
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), warn_color, 2)

                    rel_ass = reliability_assessments.get(obj.track_id)
                    w_state = warn_dec.state if warn_dec else "NO_WARNING"
                    label = f"ID:{obj.track_id} | {obj.class_name} | [{w_state}]"

                    if risk_ass and rel_ass:
                        sub_label = f"Risk:{risk_ass.risk_level}({risk_ass.risk_score:.2f}) | Rel:{rel_ass.reliability_level}({rel_ass.reliability_score:.2f}) | {ttc_str}"
                        third_label = f"{warn_dec.primary_reason if warn_dec else rel_ass.primary_reason}"
                    elif risk_ass:
                        sub_label = f"Risk:{risk_ass.risk_level}({risk_ass.risk_score:.2f}) | {ttc_str}"
                        third_label = f"Path:{risk_ass.path_state}"
                    else:
                        sub_label = f"{ttc_str} | hist:{hist_len}f"
                        third_label = "Status: UNKNOWN"

                    cv2.putText(display_frame, label, (x1, max(20, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                    cv2.putText(display_frame, sub_label, (x1, max(34, y1 - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, warn_color, 1)
                    cv2.putText(display_frame, third_label, (x1, max(48, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
                    cv2.circle(display_frame, (int(obj.center_x), int(obj.center_y)), 4, color, -1)

                # Show colorized depth inset in bottom-right corner if enabled
                if show_depth_inset:
                    color_depth = DepthAnythingV2Estimator.colorize_depth(depth_result)
                    inset_h, inset_w = 120, 160
                    small_depth = cv2.resize(color_depth, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
                    fh, fw = display_frame.shape[:2]
                    display_frame[fh - inset_h - 10 : fh - 10, fw - inset_w - 10 : fw - 10] = small_depth
                    cv2.rectangle(display_frame, (fw - inset_w - 10, fh - inset_h - 10), (fw - 10, fh - 10), (255, 255, 255), 1)
                    cv2.putText(display_frame, "Relative Depth", (fw - inset_w - 5, fh - inset_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # HUD Statistics
                overlay_top = f"Frame: {packet.frame_index} | Tracks: {len(object_depths)} | Loop FPS: {overall_fps:.1f}"
                overlay_sub = f"Det:{det_latency_ms:.0f}ms|Trk:{track_latency_ms:.0f}ms|Dep:{depth_latency_ms:.0f}ms|Mot:{motion_latency_ms:.1f}ms|Cam:{cam_latency_ms:.1f}ms|TTC:{ttc_latency_ms:.1f}ms|Risk:{risk_latency_ms:.1f}ms|Rel:{rel_latency_ms:.1f}ms|Warn:{warn_latency_ms:.1f}ms|TTS:{tts_latency_ms:.1f}ms"
                overlay_cam = f"Camera Motion: dx={camera_motion.dx:+.1f}px dy={camera_motion.dy:+.1f}px | SysHealth: [{system_reliability.system_status}]"
                alert_color_map = {
                    "CRITICAL": (0, 0, 255),
                    "WARNING": (0, 140, 255),
                    "CAUTION": (0, 255, 255),
                    "NO_WARNING": (0, 255, 0),
                    "UNKNOWN": (200, 200, 200),
                }
                a_color = alert_color_map.get(global_warning.state, (255, 255, 255))
                overlay_alert = f"GLOBAL ALERT: [{global_warning.state}] (TID:{global_warning.selected_track_id}) - {global_warning.reason}"
                overlay_nav = f"NAV GUIDANCE: [{scene_nav.navigation_state}] SafeDir: [{scene_nav.safe_direction}] - {scene_nav.reason}"
                speech_text = warning_message.text if warning_message.text else "(Silent)"
                overlay_speech = f"TTS: \"{speech_text}\" [{warning_message.priority}] (Spoke: {warning_message.should_speak})"
                cv2.putText(display_frame, overlay_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(display_frame, overlay_sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 0), 2)
                cv2.putText(display_frame, overlay_cam, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
                cv2.putText(display_frame, overlay_alert, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, a_color, 2)
                cv2.putText(display_frame, overlay_nav, (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1)
                cv2.putText(display_frame, overlay_speech, (10, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

                                # Show optical flow inliers if requested
                if show_flow and camera_motion.feature_points_curr is not None:
                    pts = camera_motion.feature_points_curr
                    inliers = camera_motion.inliers_mask if camera_motion.inliers_mask is not None else np.ones(len(pts), dtype=bool)
                    for idx, pt in enumerate(pts):
                        color = (0, 255, 0) if inliers[idx] else (0, 0, 255)
                        cv2.circle(display_frame, (int(pt[0][0]), int(pt[0][1])), 2, color, -1)

                                # Draw semi-transparent walking path corridor lines if enabled
                if risk_path_cfg and risk_path_cfg.get("enabled", True):
                    c_center = fw * float(risk_path_cfg.get("center_ratio", 0.50))
                    c_half = (fw * float(risk_path_cfg.get("corridor_width_ratio", 0.40))) / 2.0
                    x_left, x_right = int(c_center - c_half), int(c_center + c_half)
                    cv2.line(display_frame, (x_left, 0), (x_left, fh), (255, 255, 0), 1)
                    cv2.line(display_frame, (x_right, 0), (x_right, fh), (255, 255, 0), 1)
                    cv2.putText(display_frame, "Walking Corridor", (x_left + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested exit via 'q'.")
                    break

            if max_frames is not None and frames_processed >= max_frames:
                print(f"\nReached maximum requested frame count ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        camera.release()
        tts_engine.shutdown()
        if display_enabled:
            cv2.destroyAllWindows()
        print("-" * 75)
        print(f"Camera released. Total frames processed: {frames_processed}")

        if telemetry_dir is not None and telemetry_records:
            os.makedirs(telemetry_dir, exist_ok=True)
            # 1. Save hierarchical JSON
            json_path = os.path.join(telemetry_dir, f"{telemetry_prefix}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(telemetry_records, f, indent=2)

            # 2. Save per-frame summary CSV
            frame_csv_path = os.path.join(telemetry_dir, f"{telemetry_prefix}_frames.csv")
            with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_index", "timestamp_sec",
                    "raw_detections", "active_detections", "filtered_detections",
                    "active_tracks",
                    "det_latency_ms", "track_latency_ms", "depth_latency_ms",
                    "motion_latency_ms", "cam_latency_ms", "ttc_latency_ms",
                    "risk_latency_ms", "rel_latency_ms", "warn_latency_ms",
                    "nav_latency_ms", "tts_latency_ms", "total_frame_ms",
                    "cam_dx", "cam_dy", "cam_valid", "system_health",
                    "global_warning_state", "global_warning_track_id", "global_warning_reason",
                    "nav_state", "safe_direction", "nav_reason",
                    "tts_spoken", "tts_text",
                    "device_name", "vram_allocated_mb", "vram_reserved_mb"
                ])
                for r in telemetry_records:
                    lats = r["latencies_ms"]
                    cam = r["camera_motion"]
                    gw = r["global_warning"]
                    nav = r["navigation"]
                    aud = r["audio"]
                    dev = r.get("device", {})
                    writer.writerow([
                        r["frame_index"], r["timestamp_sec"],
                        r.get("raw_detections_count", 0), r.get("active_detections_count", 0), r.get("filtered_detections_count", 0),
                        r["active_tracks_count"],
                        lats["detector"], lats["tracker"], lats["depth"],
                        lats["motion"], lats["camera_motion"], lats["ttc"],
                        lats["risk"], lats["reliability"], lats["warning"],
                        lats["navigation"], lats["tts"], lats["total_frame"],
                        cam["dx"], cam["dy"], cam["valid"], r["system_health"],
                        gw["state"], gw["selected_track_id"], gw["reason"],
                        nav["state"], nav["safe_direction"], nav["reason"],
                        aud["spoken"], aud["text"],
                        dev.get("device_name", "N/A"), dev.get("vram_allocated_mb", 0.0), dev.get("vram_reserved_mb", 0.0)
                    ])

            # 3. Save object-level CSV
            obj_csv_path = os.path.join(telemetry_dir, f"{telemetry_prefix}_objects.csv")
            with open(obj_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_index", "timestamp_sec", "track_id", "class_name", "confidence",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "depth_value", "depth_valid", "approach_state", "closing_speed",
                    "ttc_seconds", "ttc_state", "ttc_valid",
                    "risk_score", "risk_level", "risk_primary_reason", "warning_state"
                ])
                for r in telemetry_records:
                    f_idx = r["frame_index"]
                    t_sec = r["timestamp_sec"]
                    for obj in r["objects"]:
                        bbox = obj["bbox"]
                        writer.writerow([
                            f_idx, t_sec, obj["track_id"], obj["class_name"], obj["confidence"],
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            obj["depth_value"], obj["depth_valid"], obj["approach_state"], obj["closing_speed"],
                            obj["ttc_seconds"], obj["ttc_state"], obj["ttc_valid"],
                            obj["risk_score"], obj["risk_level"], obj["risk_primary_reason"], obj["warning_state"]
                        ])
            print(f"[Telemetry] Saved real per-frame telemetry to {telemetry_dir} (prefix='{telemetry_prefix}', frames={len(telemetry_records)})")
        print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Edge-AI Navigation - Complete Real Perception & Risk Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Path to test video file")
    parser.add_argument("--cam", type=int, default=None, help="Camera index")
    parser.add_argument("--model", type=str, default=None, help="Override YOLO model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda', 'cuda:0', 'cpu', or 'auto'")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (for testing)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI preview window")
    parser.add_argument("--telemetry-dir", type=str, default=None, help="Directory to save per-frame telemetry logs")
    parser.add_argument("--telemetry-prefix", type=str, default="real_telemetry", help="Prefix for telemetry files (e.g. gpu_telemetry)")
    parser.add_argument("--depth-cadence", type=int, default=None, help="Depth inference cadence: 1 = every frame, 2 = every 2nd frame (2:1 cadence)")
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.cam if args.cam is not None else args.video

    run_perception_pipeline(
        config=config,
        source_override=source,
        model_override=args.model,
        device_override=args.device,
        max_frames=args.max_frames,
        headless=args.headless,
        telemetry_dir=args.telemetry_dir,
        telemetry_prefix=args.telemetry_prefix,
        depth_cadence_override=args.depth_cadence,
    )

if __name__ == "__main__":
    main()
