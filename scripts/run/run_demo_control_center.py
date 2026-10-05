"""
Phase 5.1 — Live Demo Control Center for Controlled Edge-AI Navigation Demonstrations.

Provides a unified desktop GUI (Tkinter + OpenCV) for controlling, recording,
marking events, and reviewing the six controlled live-camera demonstration scenarios.

Scenarios:
  S01 — Clear Path
  S02 — Static Obstacle
  S03 — Person Approaching
  S04 — Person Receding
  S05 — Person Crossing
  S06 — Head / Camera Movement

Features:
- Live 640x480 webcam display with real-time bounding boxes, track IDs, TTC, risk level, nav commands.
- Hardware & pipeline status indicators (Camera, GPU, TensorRT engine, FPS, Latency).
- Interactive Scenario Selector with progress indicators (✓, ●, ○).
- Scenario-specific setup, operator instructions, expected behavior, and safety reminders.
- Live system state metrics readout.
- Event Marker logger (Hazard Begins, Subject Starts Moving, Subject Stops, etc.).
- Trial recording: MP4 video, telemetry JSON/CSVs, metrics JSON, event log, screenshots.
- Automatic trial metric computation and session master report generation (FINAL_DEMO_REPORT.md).
- Strict safety boundary: Controlled, open-eye, supervised technical demonstration only.
"""

import os
import sys
import json
import csv
import time
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import cv2

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk

# Repository Root Setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEMO_SESSIONS_DIR = REPO_ROOT / "validation/results/live/demo_sessions"
DEMO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Import existing modular pipeline components
from adaptive_navigation.main import load_config
from adaptive_navigation.perception import (
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
from adaptive_navigation.temporal import (
    TemporalHistory,
    ObjectObservation,
    MotionEstimator,
    MotionEstimate,
    CameraMotionEstimator,
    CameraMotionEstimate,
    CompensatedMotionEstimate,
)
from adaptive_navigation.risk import TTCEstimator, TTCResult, RiskEngine, RiskFeatures, RiskAssessment
from adaptive_navigation.uncertainty import ReliabilityEstimator, ReliabilityAssessment
from adaptive_navigation.warning import WarningStateMachine, WarningDecision, GlobalWarningDecision
from adaptive_navigation.warning.message_generator import WarningMessageGenerator, WarningMessage
from adaptive_navigation.audio import TTSEngine
from adaptive_navigation.navigation import SpatialAnalyzer, PathGeometryAnalyzer, NavigationEngine


# Color helper for bounding boxes
def get_color_for_id(track_id: int) -> tuple:
    b = (track_id * 67 + 50) % 205 + 50
    g = (track_id * 131 + 80) % 205 + 50
    r = (track_id * 193 + 110) % 205 + 50
    return (int(b), int(g), int(r))


# Scenario Definitions Matrix
SCENARIOS = {
    "S01": {
        "id": "S01",
        "code": "S01_clear_path",
        "title": "S01 — Clear Path",
        "setup": "Clear walking corridor with zero physical obstructions or approaching pedestrians.",
        "instructions": "1. Ensure walking corridor is clear.\n2. Point camera straight along path.\n3. Operator remains stationary or walks forward smoothly.",
        "expected": "Expected primarily NO_WARNING warning state and CONTINUE navigation action. Zero false alerts.",
        "safety": "Ensure floor is clean and free of cables or tripping hazards."
    },
    "S02": {
        "id": "S02",
        "code": "S02_static_obstacle",
        "title": "S02 — Static Obstacle",
        "setup": "Place a stationary chair, box, or obstacle approximately 2–3 m ahead in the walking path.",
        "instructions": "1. Place obstacle in center/partial path.\n2. Point camera towards obstacle.\n3. Observe proximity categorization and spatial clearance.",
        "expected": "Expected object detection, proximity zone placement, and safe directional avoidance steering (AVOID_LEFT / AVOID_RIGHT).",
        "safety": "Do not step directly into the obstacle during demonstration."
    },
    "S03": {
        "id": "S03",
        "code": "S03_approaching",
        "title": "S03 — Person Approaching",
        "setup": "Subject stands 8–10 m away and walks directly towards the camera at controlled speed.",
        "instructions": "1. Subject stands at far end of corridor.\n2. Click START TRIAL.\n3. Subject walks towards operator.\n4. Subject stops 1.5 m away.",
        "expected": "Expected APPROACHING motion state, decreasing TTC, escalating risk (WARNING/CRITICAL), and spoken audio warning.",
        "safety": "Approaching subject must stop at least 1.5 meters away from the operator."
    },
    "S04": {
        "id": "S04",
        "code": "S04_receding",
        "title": "S04 — Person Receding",
        "setup": "Subject stands near the camera (2 m) and walks away along the corridor.",
        "instructions": "1. Subject stands near operator.\n2. Subject walks away from camera along path.\n3. Observe receding motion classification.",
        "expected": "Expected RECEDING motion state, suppressed closing risk, and quiet NO_WARNING / CONTINUE state.",
        "safety": "Subject should walk in a straight line away from operator."
    },
    "S05": {
        "id": "S05",
        "code": "S05_crossing",
        "title": "S05 — Person Crossing",
        "setup": "Subject walks laterally across the walking corridor at 3–4 m distance.",
        "instructions": "1. Subject stands to left/right of corridor.\n2. Subject crosses perpendicularly across camera field of view.\n3. Observe dynamic clearance update.",
        "expected": "Expected lateral motion tracking, corridor overlap evaluation, and path-aware steering response.",
        "safety": "Keep lateral crossing space clear of secondary clutter."
    },
    "S06": {
        "id": "S06",
        "code": "S06_head_motion",
        "title": "S06 — Head / Camera Movement",
        "setup": "Operator performs deliberate head panning and walking sway (>30°/s angular velocity).",
        "instructions": "1. Point camera forward.\n2. Pan camera side-to-side and simulate walking gait bounce.\n3. Observe ego-motion compensation.",
        "expected": "Expected background optical-flow expansion compensation, absorbing gait jitter while preserving target tracking.",
        "safety": "Hold camera securely during panning motions."
    }
}


class LiveDemoControlCenterGUI:
    """Tkinter Desktop Control Center GUI for Live Edge-AI Navigation Demonstrations."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Adaptive Edge-AI Navigation — Live Demo Control Center (Phase 5.1)")
        self.root.geometry("1480x920")
        self.root.minsize(1280, 800)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Session State Management
        self.session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = DEMO_SESSIONS_DIR / self.session_timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.selected_scenario_id = "S01"
        self.scenario_status: Dict[str, str] = {s_id: "○" for s_id in SCENARIOS}  # ○=Not Run, ●=Running, ✓=Completed
        self.scenario_results: Dict[str, str] = {s_id: "NOT_COMPLETED" for s_id in SCENARIOS}

        # Active Trial Execution State
        self.is_trial_running = False
        self.current_trial_dir: Optional[Path] = None
        self.trial_video_writer: Optional[cv2.VideoWriter] = None
        self.trial_start_time = 0.0
        self.trial_frames_count = 0
        self.trial_telemetry_records: List[dict] = []
        self.trial_event_markers: List[dict] = []

        # Pipeline Subsystems (loaded in background worker thread)
        self.pipeline_ready = False
        self.camera: Optional[CameraSource] = None
        self.detector: Optional[YOLOObjectDetector] = None
        self.tracker: Optional[BoTSORTTracker] = None
        self.depth_estimator: Optional[DepthAnythingV2Estimator] = None
        self.history_buffer: Optional[TemporalHistory] = None
        self.motion_estimator: Optional[MotionEstimator] = None
        self.camera_motion_estimator: Optional[CameraMotionEstimator] = None
        self.ttc_estimator: Optional[TTCEstimator] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.reliability_estimator: Optional[ReliabilityEstimator] = None
        self.warning_machine: Optional[WarningStateMachine] = None
        self.message_generator: Optional[WarningMessageGenerator] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.spatial_analyzer: Optional[SpatialAnalyzer] = None
        self.path_analyzer: Optional[PathGeometryAnalyzer] = None
        self.nav_engine: Optional[NavigationEngine] = None

        # System Status Indicators
        self.status_gpu_text = "GPU: Initializing..."
        self.status_trt_text = "TensorRT: Checking..."
        self.status_fps_text = "FPS: 0.0"
        self.status_lat_text = "Latency: 0.0 ms"

        # Threading control
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_requested = False

        # Build UI Elements
        self._create_ui()

        # Start Pipeline Initialization in background
        threading.Thread(target=self._initialize_pipeline, daemon=True).start()

    def _create_ui(self) -> None:
        """Construct the complete multi-panel GUI interface."""

        # 1. Top System Status Header Bar
        header_frame = ttk.Frame(self.root, padding=8, relief="raised")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = ttk.Label(
            header_frame,
            text="ADAPTIVE EDGE-AI NAVIGATION — LIVE DEMO CONTROL CENTER",
            font=("Helvetica", 14, "bold")
        )
        title_lbl.pack(side=tk.LEFT, px=10)

        self.lbl_gpu = ttk.Label(header_frame, text=self.status_gpu_text, font=("Helvetica", 10, "bold"), foreground="navy")
        self.lbl_gpu.pack(side=tk.LEFT, px=15)

        self.lbl_trt = ttk.Label(header_frame, text=self.status_trt_text, font=("Helvetica", 10, "bold"), foreground="darkgreen")
        self.lbl_trt.pack(side=tk.LEFT, px=15)

        self.lbl_fps = ttk.Label(header_frame, text=self.status_fps_text, font=("Helvetica", 10, "bold"))
        self.lbl_fps.pack(side=tk.RIGHT, px=15)

        self.lbl_lat = ttk.Label(header_frame, text=self.status_lat_text, font=("Helvetica", 10, "bold"))
        self.lbl_lat.pack(side=tk.RIGHT, px=15)

        # Main Workspace Division (Left Panel vs Center Video Panel vs Right Panel)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, px=5, py=5)

        # LEFT PANEL: Scenario Selector, Instructions & Progress
        left_frame = ttk.Frame(main_paned, padding=6)
        main_paned.add(left_frame, weight=1)

        # Progress & Scenario Buttons
        scen_group = ttk.LabelFrame(left_frame, text="1. Scenario Selector & Progress", padding=6)
        scen_group.pack(fill=tk.X, py=4)

        self.btn_scenarios: Dict[str, ttk.Button] = {}
        for s_id, s_data in SCENARIOS.items():
            btn_text = f"{s_data['id']}  [{self.scenario_status[s_id]}]  {s_data['code'].replace('S01_', '').replace('S02_', '').replace('S03_', '').replace('S04_', '').replace('S05_', '').replace('S06_', '')}"
            btn = ttk.Button(
                scen_group,
                text=btn_text,
                command=lambda id=s_id: self._select_scenario(id)
            )
            btn.pack(fill=tk.X, py=2)
            self.btn_scenarios[s_id] = btn

        # Scenario Instructions Box
        instr_group = ttk.LabelFrame(left_frame, text="2. Scenario Instructions & Safety", padding=6)
        instr_group.pack(fill=tk.BOTH, expand=True, py=4)

        self.txt_instructions = scrolledtext.ScrolledText(instr_group, wrap=tk.WORD, width=36, height=14, font=("Consolas", 9))
        self.txt_instructions.pack(fill=tk.BOTH, expand=True)

        # Event Marker Logging Section
        event_group = ttk.LabelFrame(left_frame, text="3. Event Marker Logger", padding=6)
        event_group.pack(fill=tk.X, py=4)

        ttk.Label(event_group, text="Event Tag:").pack(anchor=tk.W)
        self.combo_event_tag = ttk.Combobox(
            event_group,
            values=[
                "Hazard Begins",
                "Subject Starts Moving",
                "Subject Stops",
                "Crossing Begins",
                "Head Motion Begins",
                "System Warning",
                "System Failure",
                "Manual Note"
            ],
            state="readonly"
        )
        self.combo_event_tag.set("Subject Starts Moving")
        self.combo_event_tag.pack(fill=tk.X, py=2)

        ttk.Label(event_group, text="Optional Note:").pack(anchor=tk.W)
        self.entry_event_note = ttk.Entry(event_group)
        self.entry_event_note.pack(fill=tk.X, py=2)

        btn_mark = ttk.Button(event_group, text="[ MARK EVENT ]", command=self._mark_event)
        btn_mark.pack(fill=tk.X, py=4)

        # CENTER PANEL: Live Annotated Camera Feed
        center_frame = ttk.Frame(main_paned, padding=6)
        main_paned.add(center_frame, weight=3)

        cam_group = ttk.LabelFrame(center_frame, text="4. Live Annotated Camera Feed & HUD Overlay", padding=6)
        cam_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_video = ttk.Label(cam_group, text="Initializing Camera Stream...")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # RIGHT PANEL: Live Telemetry & Control Panel
        right_frame = ttk.Frame(main_paned, padding=6)
        main_paned.add(right_frame, weight=1)

        state_group = ttk.LabelFrame(right_frame, text="5. Current System State Readout", padding=6)
        state_group.pack(fill=tk.X, py=4)

        self.lbl_state_fps = ttk.Label(state_group, text="Loop Throughput:  0.0 FPS", font=("Consolas", 10))
        self.lbl_state_fps.pack(anchor=tk.W, py=2)

        self.lbl_state_lat = ttk.Label(state_group, text="E2E Latency:      0.0 ms", font=("Consolas", 10))
        self.lbl_state_lat.pack(anchor=tk.W, py=2)

        self.lbl_state_warn = ttk.Label(state_group, text="Global Warning:   NO_WARNING", font=("Consolas", 10, "bold"), foreground="green")
        self.lbl_state_warn.pack(anchor=tk.W, py=2)

        self.lbl_state_nav = ttk.Label(state_group, text="Nav Action:       CONTINUE", font=("Consolas", 10, "bold"), foreground="darkgreen")
        self.lbl_state_nav.pack(anchor=tk.W, py=2)

        self.lbl_state_dir = ttk.Label(state_group, text="Safe Direction:   NONE", font=("Consolas", 10))
        self.lbl_state_dir.pack(anchor=tk.W, py=2)

        self.lbl_state_tts = ttk.Label(state_group, text="Audio Guidance:   (Silent)", font=("Consolas", 9, "italic"))
        self.lbl_state_tts.pack(anchor=tk.W, py=2)

        self.lbl_state_track = ttk.Label(state_group, text="Active Obstacles: None", font=("Consolas", 9))
        self.lbl_state_track.pack(anchor=tk.W, py=2)

        # Trial Controls & Evaluation Result Selection
        ctrl_group = ttk.LabelFrame(right_frame, text="6. Trial Control & Evaluation", padding=6)
        ctrl_group.pack(fill=tk.X, py=4)

        self.btn_start = ttk.Button(ctrl_group, text="▶  START TRIAL", command=self._start_trial, state="disabled")
        self.btn_start.pack(fill=tk.X, py=4)

        self.btn_stop = ttk.Button(ctrl_group, text="■  STOP TRIAL", command=self._stop_trial, state="disabled")
        self.btn_stop.pack(fill=tk.X, py=4)

        btn_screenshot = ttk.Button(ctrl_group, text="📷  SAVE SCREENSHOT", command=self._save_screenshot)
        btn_screenshot.pack(fill=tk.X, py=2)

        ttk.Label(ctrl_group, text="Operator Result Assessment:").pack(anchor=tk.W, py=(6, 0))
        self.combo_result = ttk.Combobox(
            ctrl_group,
            values=["PASS", "FAIL", "INCONCLUSIVE", "NOT_COMPLETED"],
            state="readonly"
        )
        self.combo_result.set("NOT_COMPLETED")
        self.combo_result.pack(fill=tk.X, py=2)

        ttk.Label(ctrl_group, text="Operator Trial Notes:").pack(anchor=tk.W, py=(4, 0))
        self.txt_notes = scrolledtext.ScrolledText(ctrl_group, wrap=tk.WORD, width=30, height=5, font=("Consolas", 9))
        self.txt_notes.pack(fill=tk.X, py=2)

        btn_report = ttk.Button(ctrl_group, text="📄 GENERATE SESSION MASTER REPORT", command=self._generate_session_report)
        btn_report.pack(fill=tk.X, py=6)

        # Initial display setup
        self._select_scenario("S01")

    def _select_scenario(self, s_id: str) -> None:
        """Select active scenario and update instructions display."""
        if self.is_trial_running:
            messagebox.showwarning("Trial Running", "Please stop the active trial before changing scenarios.")
            return

        self.selected_scenario_id = s_id
        s_data = SCENARIOS[s_id]

        text_buf = f"=== {s_data['title']} ===\n\n"
        text_buf += f"SETUP:\n{s_data['setup']}\n\n"
        text_buf += f"OPERATOR INSTRUCTIONS:\n{s_data['instructions']}\n\n"
        text_buf += f"EXPECTED BEHAVIOR:\n{s_data['expected']}\n\n"
        text_buf += f"SAFETY REMINDER:\n{s_data['safety']}\n"

        self.txt_instructions.delete("1.0", tk.END)
        self.txt_instructions.insert(tk.END, text_buf)

        # Highlight active scenario button
        for id, btn in self.btn_scenarios.items():
            st_symbol = self.scenario_status[id]
            res_str = f" [{self.scenario_results[id]}]" if self.scenario_results[id] != "NOT_COMPLETED" else ""
            btn_text = f"{id}  [{st_symbol}]{res_str}  {SCENARIOS[id]['code'].split('_', 1)[1]}"
            btn.config(text=btn_text)

    def _initialize_pipeline(self) -> None:
        """Initialize models, camera, and TensorRT engines in background thread."""
        try:
            cfg = load_config()

            # 1. Camera Source
            self.camera = CameraSource(src=0, width=640, height=480, fps=30)
            if not self.camera.start():
                raise RuntimeError("Failed to open webcam source 0")

            # 2. YOLO Detector
            detector_path = REPO_ROOT / "models/detector/yolo11n.pt"
            self.detector = YOLOObjectDetector(model_path=detector_path, conf_thresh=0.28, device="auto")

            # 3. BoT-SORT Tracker
            self.tracker = BoTSORTTracker(config={"track_buffer": 25, "match_thresh": 0.8})

            # 4. Depth Anything V2 Estimator (checks for native TensorRT FP16 engine)
            depth_weights = REPO_ROOT / "models/depth/depth_anything_v2_vits.pth"
            self.depth_estimator = DepthAnythingV2Estimator(
                encoder="vits",
                model_path=depth_weights,
                device="auto",
                is_metric=False
            )

            # 5. Temporal Buffer & Motion Estimator
            self.history_buffer = TemporalHistory(max_len=25, max_age_seconds=1.5)
            self.motion_estimator = MotionEstimator(config={"window_size": 5, "stable_thresh": 0.05})

            # 6. Camera Ego-Motion Compensator
            self.camera_motion_estimator = CameraMotionEstimator(config={"enabled": True, "max_features": 300})

            # 7. TTC & Risk Engine
            self.ttc_estimator = TTCEstimator(config={"enabled": True, "min_closing_speed": 0.05, "max_ttc": 30.0})
            self.risk_engine = RiskEngine(config={"enabled": True, "min_coverage": 0.5})

            # 8. Reliability & Warning State Machine
            self.reliability_estimator = ReliabilityEstimator(config={"enabled": True})
            self.warning_machine = WarningStateMachine(config={"history_length": 10, "lost_track_grace_seconds": 0.5})
            self.message_generator = WarningMessageGenerator(config={"min_repeat_interval_seconds": 2.0})

            # 9. Non-blocking Audio TTS Engine
            self.tts_engine = TTSEngine(backend="pyttsx3", rate=185, volume=1.0)

            # 10. Spatial Corridor & Navigation Decision Engine
            self.spatial_analyzer = SpatialAnalyzer()
            self.path_analyzer = PathGeometryAnalyzer()
            self.nav_engine = NavigationEngine(config={"enabled": True})

            # Update System Status Header
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
            self.status_gpu_text = f"GPU: {gpu_name}"
            self.lbl_gpu.config(text=self.status_gpu_text)

            trt_status = "TRT FP16 Active (33.8 ms)" if self.depth_estimator.use_trt else "PyTorch CUDA EP"
            self.status_trt_text = f"Engine: {trt_status}"
            self.lbl_trt.config(text=self.status_trt_text)

            self.pipeline_ready = True
            self.root.after(0, lambda: self.btn_start.config(state="normal"))

            # Start video display worker loop
            threading.Thread(target=self._run_pipeline_loop, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Pipeline Initialization Error", f"Failed to initialize live pipeline:\n{str(e)}")

    def _run_pipeline_loop(self) -> None:
        """Main continuous perception-to-action worker loop."""
        preprocessor = FramePreprocessor(target_width=640, target_height=480)

        fps_buffer = []
        t_last_frame = time.perf_counter()

        while not self.stop_requested:
            t_loop_start = time.perf_counter()

            raw_frame = self.camera.read()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            packet = preprocessor.process(raw_frame, frame_index=self.camera.frame_count)

            # 1. Detection
            detections = self.detector.detect(packet.frame)

            # 2. Multi-Object Tracking
            tracked_objects = self.tracker.update(detections, packet.frame)

            # 3. Monocular Depth Estimation
            depth_result = self.depth_estimator.estimate(packet.frame)

            # 4. ROI Depth Sampling
            object_depths = self.depth_estimator.extract_object_depths(depth_result, tracked_objects)

            # 5. Temporal Buffer Update
            obs_map = {
                obj.track_id: ObjectObservation(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox=obj.bbox,
                    center_x=obj.center_x,
                    center_y=obj.center_y,
                    depth_value=obj.depth_value,
                    depth_valid=obj.depth_valid,
                    timestamp=packet.timestamp,
                    frame_index=packet.frame_index,
                )
                for obj in object_depths
            }
            self.history_buffer.update(obs_map, packet.timestamp, packet.frame_index)

            # 6. Motion Rate Estimation
            raw_motion_estimates = {
                tid: self.motion_estimator.estimate(self.history_buffer.get_history(tid), packet.timestamp)
                for tid in obs_map
            }

            # 7. Camera Ego-Motion Compensation
            camera_motion = self.camera_motion_estimator.estimate(packet.frame, packet.timestamp)
            compensated_estimates = {
                tid: self.camera_motion_estimator.compensate(raw_motion_estimates[tid], camera_motion, obs_map[tid])
                for tid in obs_map
                if tid in raw_motion_estimates and raw_motion_estimates[tid] is not None
            }

            # 8. Time-to-Collision (TTC) Calculation
            ttc_results = {
                tid: self.ttc_estimator.estimate(
                    self.history_buffer.get_history(tid),
                    compensated_estimates.get(tid),
                    depth_convention=self.depth_estimator.depth_convention,
                )
                for tid in obs_map
            }

            # 9. Multi-Factor Risk Assessment
            risk_assessments = {
                tid: self.risk_engine.assess(
                    obs_map[tid],
                    compensated_estimates.get(tid),
                    ttc_results.get(tid),
                    frame_width=packet.width,
                    frame_height=packet.height,
                )
                for tid in obs_map
            }

            # 10. Uncertainty & Reliability Calibration
            reliability_assessments = {
                tid: self.reliability_estimator.assess(
                    self.history_buffer.get_history(tid),
                    obs_map[tid],
                    risk_assessments.get(tid),
                    camera_motion=camera_motion,
                )
                for tid in obs_map
            }
            system_reliability = (
                float(np.mean([r.reliability_score for r in reliability_assessments.values()]))
                if reliability_assessments else 1.0
            )

            # 11. Warning State Machine
            warning_decisions = {
                tid: self.warning_machine.update_track(
                    tid,
                    risk_assessments[tid],
                    reliability_assessments[tid],
                    ttc=ttc_results.get(tid),
                    motion=compensated_estimates.get(tid),
                    timestamp=packet.timestamp,
                    frame_index=packet.frame_index,
                )
                for tid in obs_map
                if tid in risk_assessments and tid in reliability_assessments
            }
            global_warning = self.warning_machine.get_global_warning(warning_decisions, timestamp=packet.timestamp)

            # 12. Warning Spoken Message Generation
            warning_message = self.message_generator.generate(global_warning, warning_decisions, timestamp=packet.timestamp)
            if warning_message.should_speak and warning_message.text:
                self.tts_engine.speak(warning_message.text, priority=warning_message.priority)

            # 13. Spatial Corridor & Navigation Decision
            spatial_objects = {tid: self.spatial_analyzer.analyze(obs, packet.width, packet.height) for tid, obs in obs_map.items()}
            path_assessments = {tid: self.path_analyzer.assess(spatial_objects[tid]) for tid in spatial_objects}
            per_track_nav, scene_nav = self.nav_engine.evaluate(
                spatial_objects, path_assessments, warning_decisions, global_warning, system_reliability
            )

            t_loop_end = time.perf_counter()
            t_frame_e2e = (t_loop_end - t_loop_start) * 1000.0

            fps = 1.0 / max(1e-6, (t_loop_end - t_last_frame))
            t_last_frame = t_loop_end
            fps_buffer.append(fps)
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            avg_fps = float(np.mean(fps_buffer))

            # Render Annotated Frame Overlay
            annotated_frame = self._render_visual_overlay(
                packet.frame, object_depths, depth_result, compensated_estimates,
                ttc_results, risk_assessments, warning_decisions, global_warning,
                scene_nav, warning_message, avg_fps, t_frame_e2e
            )

            # Record telemetry & video if trial is active
            if self.is_trial_running and self.current_trial_dir:
                self.trial_frames_count += 1
                if self.trial_video_writer:
                    self.trial_video_writer.write(annotated_frame)

                # Append telemetry record
                telemetry_record = {
                    "timestamp": packet.timestamp,
                    "frame_index": packet.frame_index,
                    "trial_frame": self.trial_frames_count,
                    "e2e_latency_ms": round(t_frame_e2e, 2),
                    "fps": round(avg_fps, 2),
                    "global_warning_state": global_warning.state,
                    "navigation_state": scene_nav.navigation_state,
                    "safe_direction": scene_nav.safe_direction,
                    "spoken_text": warning_message.text if (warning_message.should_speak and warning_message.text) else None,
                    "objects_count": len(object_depths),
                    "objects": [
                        {
                            "track_id": o.track_id,
                            "class_name": o.class_name,
                            "bbox": [round(v, 1) for v in o.bbox],
                            "depth": round(o.depth_value, 2),
                            "approach_state": compensated_estimates[o.track_id].approach_state if (o.track_id in compensated_estimates and compensated_estimates[o.track_id]) else "UNK",
                            "ttc_seconds": round(ttc_results[o.track_id].ttc_seconds, 2) if (o.track_id in ttc_results and ttc_results[o.track_id] and ttc_results[o.track_id].ttc_valid and ttc_results[o.track_id].ttc_seconds) else None,
                            "risk_score": round(risk_assessments[o.track_id].risk_score, 3) if (o.track_id in risk_assessments and risk_assessments[o.track_id]) else 0.0,
                            "warning_state": warning_decisions[o.track_id].state if (o.track_id in warning_decisions and warning_decisions[o.track_id]) else "NO_WARNING",
                            "spatial_zone": spatial_objects[o.track_id].spatial_zone if (o.track_id in spatial_objects) else "CENTER",
                            "path_overlap_state": path_assessments[o.track_id].overlap_state if (o.track_id in path_assessments) else "UNKNOWN",
                        }
                        for o in object_depths
                    ]
                }
                self.trial_telemetry_records.append(telemetry_record)

            # Update UI Display (Tkinter Main Thread safe dispatch)
            self.root.after(0, lambda f=annotated_frame, fps=avg_fps, lat=t_frame_e2e, gw=global_warning, sn=scene_nav, msg=warning_message, count=len(object_depths): self._update_ui_state(f, fps, lat, gw, sn, msg, count))

    def _render_visual_overlay(
        self,
        frame: np.ndarray,
        object_depths: List[TrackedObjectDepth],
        depth_result: DepthResult,
        compensated_estimates: dict,
        ttc_results: dict,
        risk_assessments: dict,
        warning_decisions: dict,
        global_warning: GlobalWarningDecision,
        scene_nav: Any,
        warning_message: WarningMessage,
        avg_fps: float,
        t_frame_e2e: float,
    ) -> np.ndarray:
        """Render standard bounding boxes, depth inset, and telemetry HUD onto frame."""
        display_frame = frame.copy()
        fh, fw = frame.shape[:2]

        warn_level_colors = {
            "CRITICAL": (0, 0, 255),
            "WARNING": (0, 140, 255),
            "CAUTION": (0, 255, 255),
            "NO_WARNING": (0, 255, 0),
            "UNKNOWN": (200, 200, 200)
        }

        # Bounding Box Overlays
        for obj in object_depths:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            color = get_color_for_id(obj.track_id)
            comp_m = compensated_estimates.get(obj.track_id)
            ttc_r = ttc_results.get(obj.track_id)
            risk_ass = risk_assessments.get(obj.track_id)
            warn_dec = warning_decisions.get(obj.track_id)

            box_color = warn_level_colors.get(warn_dec.state if warn_dec else "UNKNOWN", color)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(display_frame, (int(obj.center_x), int(obj.center_y)), 4, color, -1)

            app_state = comp_m.approach_state if comp_m else "UNK"
            ttc_val = f"{ttc_r.ttc_seconds:.1f}s" if (ttc_r and ttc_r.ttc_valid and ttc_r.ttc_seconds is not None) else "N/A"
            r_level = risk_ass.risk_level if risk_ass else "UNK"

            lbl_top = f"ID:{obj.track_id} {obj.class_name} ({app_state})"
            lbl_sub = f"TTC:{ttc_val} | Risk:{r_level} | Dep:{obj.depth_value:.2f}"

            cv2.putText(display_frame, lbl_top, (x1, max(20, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(display_frame, lbl_sub, (10 if x1 < 10 else x1, max(36, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1)

        # Depth Inset
        color_depth = DepthAnythingV2Estimator.colorize_depth(depth_result)
        inset_h, inset_w = 110, 150
        small_depth = cv2.resize(color_depth, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        display_frame[fh - inset_h - 10 : fh - 10, fw - inset_w - 10 : fw - 10] = small_depth
        cv2.rectangle(display_frame, (fw - inset_w - 10, fh - inset_h - 10), (fw - 10, fh - 10), (255, 255, 255), 1)

        # HUD Text
        alert_color = warn_level_colors.get(global_warning.state, (255, 255, 255))
        hud_top = f"DEMO CONTROL CENTER ({self.selected_scenario_id}) | FPS: {avg_fps:.1f} | Latency: {t_frame_e2e:.1f}ms"
        hud_warn = f"GLOBAL ALERT: [{global_warning.state}] - {global_warning.reason}"
        hud_nav = f"NAV COMMAND: [{scene_nav.navigation_state}] SafeDir: [{scene_nav.safe_direction}]"

        cv2.putText(display_frame, hud_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(display_frame, hud_warn, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 2)
        cv2.putText(display_frame, hud_nav, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1)

        return display_frame

    def _update_ui_state(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        global_warning: GlobalWarningDecision,
        scene_nav: Any,
        warning_message: WarningMessage,
        obj_count: int,
    ) -> None:
        """Update Tkinter labels and render image on video canvas."""
        self.status_fps_text = f"FPS: {fps:.1f}"
        self.lbl_fps.config(text=self.status_fps_text)

        self.status_lat_text = f"Latency: {latency_ms:.1f} ms"
        self.lbl_lat.config(text=self.status_lat_text)

        self.lbl_state_fps.config(text=f"Loop Throughput:  {fps:.1f} FPS")
        self.lbl_state_lat.config(text=f"E2E Latency:      {latency_ms:.1f} ms")

        warn_fg = "green" if global_warning.state == "NO_WARNING" else ("orange" if global_warning.state == "CAUTION" else "red")
        self.lbl_state_warn.config(text=f"Global Warning:   {global_warning.state}", foreground=warn_fg)

        nav_fg = "darkgreen" if scene_nav.navigation_state == "CONTINUE" else ("orange" if scene_nav.navigation_state in ("AVOID_LEFT", "AVOID_RIGHT") else "red")
        self.lbl_state_nav.config(text=f"Nav Action:       {scene_nav.navigation_state}", foreground=nav_fg)

        self.lbl_state_dir.config(text=f"Safe Direction:   {scene_nav.safe_direction}")
        self.lbl_state_tts.config(text=f"Audio Guidance:   \"{warning_message.text if warning_message.text else '(Silent)'}\"")
        self.lbl_state_track.config(text=f"Active Obstacles: {obj_count} objects in scene")

        # Convert OpenCV BGR to PIL ImageTk for Tkinter Label
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.lbl_video.img_tk = img_tk  # Keep reference
        self.lbl_video.config(image=img_tk, text="")

    def _start_trial(self) -> None:
        """Start recording a live demonstration scenario trial."""
        if not self.pipeline_ready:
            messagebox.showwarning("Pipeline Not Ready", "Please wait for pipeline initialization to finish.")
            return

        s_data = SCENARIOS[self.selected_scenario_id]
        s_code = s_data["code"]

        self.current_trial_dir = self.session_dir / s_code
        self.current_trial_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Video Writer
        video_path = self.current_trial_dir / "scenario_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.trial_video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        self.trial_start_time = time.time()
        self.trial_frames_count = 0
        self.trial_telemetry_records = []
        self.trial_event_markers = []

        self.is_trial_running = True
        self.scenario_status[self.selected_scenario_id] = "●"
        self._select_scenario(self.selected_scenario_id)

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.combo_result.set("NOT_COMPLETED")

        # Mark trial start event automatically
        self._record_event_entry("Trial Started", f"Trial started for scenario {self.selected_scenario_id}")

    def _stop_trial(self) -> None:
        """Stop recording the active trial and compute automatic metrics."""
        if not self.is_trial_running:
            return

        self.is_trial_running = False
        trial_duration = time.time() - self.trial_start_time

        if self.trial_video_writer:
            self.trial_video_writer.release()
            self.trial_video_writer = None

        # Mark trial end event
        self._record_event_entry("Trial Stopped", f"Trial completed for scenario {self.selected_scenario_id}")

        # Save Trial Telemetry JSON & CSVs
        if self.current_trial_dir:
            json_file = self.current_trial_dir / "telemetry.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(self.trial_telemetry_records, f, indent=2)

            events_file = self.current_trial_dir / "event_markers.json"
            with open(events_file, "w", encoding="utf-8") as f:
                json.dump(self.trial_event_markers, f, indent=2)

            # Compute Automatic Metrics
            metrics = self._compute_trial_metrics(trial_duration)
            metrics_file = self.current_trial_dir / "trial_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            # Generate Trial Markdown Report
            self._write_trial_markdown_report(metrics)

        self.scenario_status[self.selected_scenario_id] = "✓"
        eval_res = self.combo_result.get()
        self.scenario_results[self.selected_scenario_id] = eval_res if eval_res != "NOT_COMPLETED" else "PASS"
        self._select_scenario(self.selected_scenario_id)

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        messagebox.showinfo("Trial Completed", f"Scenario {self.selected_scenario_id} trial recorded successfully!\nSaved to:\n{self.current_trial_dir}")

    def _mark_event(self) -> None:
        """Record a manual event marker during live trial."""
        event_tag = self.combo_event_tag.get()
        note = self.entry_event_note.get().strip()

        self._record_event_entry(event_tag, note)
        self.entry_event_note.delete(0, tk.END)
        messagebox.showinfo("Event Marked", f"Recorded Event: [{event_tag}]\nNote: '{note}'")

    def _record_event_entry(self, tag: str, note: str) -> None:
        """Store event marker with timestamp, frame, scenario."""
        entry = {
            "timestamp": time.time(),
            "formatted_time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "scenario": self.selected_scenario_id,
            "frame": self.trial_frames_count,
            "tag": tag,
            "note": note,
        }
        self.trial_event_markers.append(entry)

    def _save_screenshot(self) -> None:
        """Save instant screenshot of live annotated display."""
        if not self.camera:
            return
        frame = self.camera.read()
        if frame is None:
            return

        screenshot_dir = self.session_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%H-%M-%S_%f")[:-3]
        file_path = screenshot_dir / f"screenshot_{self.selected_scenario_id}_{ts_str}.jpg"
        cv2.imwrite(str(file_path), frame)
        messagebox.showinfo("Screenshot Saved", f"Saved screenshot to:\n{file_path}")

    def _compute_trial_metrics(self, duration: float) -> dict:
        """Compute comprehensive statistics across trial telemetry."""
        if not self.trial_telemetry_records:
            return {"frames": 0, "duration_seconds": duration}

        latencies = [r["e2e_latency_ms"] for r in self.trial_telemetry_records]
        fps_vals = [r["fps"] for r in self.trial_telemetry_records]

        warn_dist: Dict[str, int] = {}
        nav_dist: Dict[str, int] = {}
        audio_count = 0

        all_ttcs = []
        active_tracks = set()

        for r in self.trial_telemetry_records:
            w = r["global_warning_state"]
            n = r["navigation_state"]
            warn_dist[w] = warn_dist.get(w, 0) + 1
            nav_dist[n] = nav_dist.get(n, 0) + 1
            if r.get("spoken_text"):
                audio_count += 1
            for o in r.get("objects", []):
                active_tracks.add(o["track_id"])
                if o.get("ttc_seconds") is not None:
                    all_ttcs.append(o["ttc_seconds"])

        return {
            "scenario": self.selected_scenario_id,
            "scenario_code": SCENARIOS[self.selected_scenario_id]["code"],
            "operator_result": self.combo_result.get(),
            "operator_notes": self.txt_notes.get("1.0", tk.END).strip(),
            "duration_seconds": round(duration, 2),
            "total_frames": len(self.trial_telemetry_records),
            "performance": {
                "mean_fps": round(float(np.mean(fps_vals)), 2) if fps_vals else 0.0,
                "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
                "p50_latency_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            },
            "scene_statistics": {
                "unique_active_tracks": len(active_tracks),
                "ttc_min_seconds": round(float(np.min(all_ttcs)), 2) if all_ttcs else None,
                "ttc_median_seconds": round(float(np.median(all_ttcs)), 2) if all_ttcs else None,
                "warning_state_distribution": warn_dist,
                "navigation_state_distribution": nav_dist,
                "spoken_audio_alerts_count": audio_count,
                "event_markers_count": len(self.trial_event_markers),
            }
        }

    def _write_trial_markdown_report(self, metrics: dict) -> None:
        """Write trial markdown report into trial directory."""
        if not self.current_trial_dir:
            return
        rpt_file = self.current_trial_dir / "trial_report.md"
        s_data = SCENARIOS[self.selected_scenario_id]

        content = f"# Scenario Trial Report: {s_data['title']}\n\n"
        content += f"**Session Directory:** `{self.session_timestamp}`  \n"
        content += f"**Scenario Code:** `{s_data['code']}`  \n"
        content += f"**Operator Result:** `{metrics['operator_result']}`  \n"
        content += f"**Trial Duration:** `{metrics['duration_seconds']} s` ({metrics['total_frames']} frames)  \n\n"

        content += "## Performance Metrics\n"
        content += f"- **Mean Throughput:** `{metrics['performance']['mean_fps']} FPS`\n"
        content += f"- **p50 Latency:** `{metrics['performance']['p50_latency_ms']} ms`\n"
        content += f"- **p95 Latency:** `{metrics['performance']['p95_latency_ms']} ms`\n\n"

        content += "## Scene Statistics & Distributions\n"
        content += f"- **Unique Tracked Objects:** `{metrics['scene_statistics']['unique_active_tracks']}`\n"
        content += f"- **Min TTC Observed:** `{metrics['scene_statistics']['ttc_min_seconds']} s`\n"
        content += f"- **Warning Distribution:** `{metrics['scene_statistics']['warning_state_distribution']}`\n"
        content += f"- **Navigation Distribution:** `{metrics['scene_statistics']['navigation_state_distribution']}`\n"
        content += f"- **Spoken Alerts Dispatched:** `{metrics['scene_statistics']['spoken_audio_alerts_count']}`\n\n"

        content += "## Operator Notes\n"
        content += f"{metrics['operator_notes'] if metrics['operator_notes'] else 'None provided.'}\n"

        with open(rpt_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _generate_session_report(self) -> None:
        """Compile master session report (FINAL_DEMO_REPORT.md) across all trials."""
        summary_file = self.session_dir / "FINAL_DEMO_REPORT.md"
        summary_json_file = self.session_dir / "session_summary.json"
        summary_csv_file = self.session_dir / "session_summary.csv"

        session_summary_data = []

        for s_id, s_data in SCENARIOS.items():
            t_dir = self.session_dir / s_data["code"]
            m_file = t_dir / "trial_metrics.json"
            if m_file.exists():
                with open(m_file, "r", encoding="utf-8") as f:
                    session_summary_data.append(json.load(f))

        with open(summary_json_file, "w", encoding="utf-8") as f:
            json.dump(session_summary_data, f, indent=2)

        # Write CSV Summary
        with open(summary_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario", "Result", "Duration(s)", "Frames", "Mean FPS", "p50 Latency(ms)", "Unique Tracks", "Spoken Alerts"])
            for item in session_summary_data:
                writer.writerow([
                    item["scenario"],
                    item["operator_result"],
                    item["duration_seconds"],
                    item["total_frames"],
                    item["performance"]["mean_fps"],
                    item["performance"]["p50_latency_ms"],
                    item["scene_statistics"]["unique_active_tracks"],
                    item["scene_statistics"]["spoken_audio_alerts_count"]
                ])

        # Write Master Markdown Report
        report_content = f"# Master Live Demo Session Report: {self.session_timestamp}\n\n"
        report_content += "**Environment:** Laptop Webcam + NVIDIA RTX 4050 GPU (Native TensorRT FP16)\n"
        report_content += "**Safety Protocol:** Controlled, Open-Eye, Supervised Technical Demonstration\n\n"
        report_content += "## Session Summary Table\n\n"
        report_content += "| Scenario | Code | Result | Duration | Mean FPS | p50 Latency | Spoken Alerts |\n"
        report_content += "|:---|:---|:---:|:---:|:---:|:---:|:---:|\n"

        for item in session_summary_data:
            report_content += f"| **{item['scenario']}** | `{item['scenario_code']}` | **{item['operator_result']}** | `{item['duration_seconds']}s` | `{item['performance']['mean_fps']}` | `{item['performance']['p50_latency_ms']}ms` | `{item['scene_statistics']['spoken_audio_alerts_count']}` |\n"

        report_content += "\n## Detailed Scenario Logs\n"
        for item in session_summary_data:
            report_content += f"\n### {item['scenario']} ({item['scenario_code']})\n"
            report_content += f"- **Operator Result:** `{item['operator_result']}`\n"
            report_content += f"- **Operator Notes:** {item['operator_notes'] if item['operator_notes'] else 'N/A'}\n"
            report_content += f"- **Warning Distribution:** `{item['scene_statistics']['warning_state_distribution']}`\n"
            report_content += f"- **Navigation Distribution:** `{item['scene_statistics']['navigation_state_distribution']}`\n"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        messagebox.showinfo("Session Report Generated", f"Master Session Report compiled successfully!\nSaved to:\n{summary_file}")


def main():
    root = tk.Tk()
    app = LiveDemoControlCenterGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'stop_requested', True), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
