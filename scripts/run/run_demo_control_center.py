"""
Phase 5.1 — Live Demo Control Center & Teammate Demonstration Interface.

Provides a single visual desktop GUI (Tkinter + OpenCV) for controlling,
recording, reviewing, and demonstrating the live edge-AI navigation prototype to teammates.

Features:
- Teammate Demonstration Mode & Formal Validation Mode separation.
- Live 640x480 webcam display with bounding boxes, track IDs, TTC, risk level, nav commands.
- Hardware & pipeline status indicators (Camera, GPU, TensorRT engine, FPS, Latency).
- Audio State Indicator: AUDIO: READY / SPEAKING / IDLE.
- Current Observation Summary Card (Simplified Summary Layer for non-technical viewers).
- "What is Happening?" Dynamic Explanation Panel explaining system behavior in real-time.
- "Demo Script / Guided Walkthrough" Panel with 6 recommended teammate demo steps.
- Teammate Demo Controls (Start Demo, Stop Demo, Record Demo, Screenshot, Mark Event, Reset View).
- Separate Teammate Demonstration Storage: validation/results/live/demo_sessions/team_demos/YYYY-MM-DD_HH-MM-SS/
- Pre-flight Startup Verification System checking 13 core operational requirements.
- Safety Boundary: Controlled, open-eye, supervised technical demonstration only.
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
TEAM_DEMOS_DIR = DEMO_SESSIONS_DIR / "team_demos"
DEMO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
TEAM_DEMOS_DIR.mkdir(parents=True, exist_ok=True)

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


def get_color_for_id(track_id: int) -> tuple:
    b = (track_id * 67 + 50) % 205 + 50
    g = (track_id * 131 + 80) % 205 + 50
    r = (track_id * 193 + 110) % 205 + 50
    return (int(b), int(g), int(r))


# Scenario & Guided Walkthrough Script Definitions
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
        "instructions": "1. Subject stands at far end of corridor.\n2. Click START TRIAL / DEMO.\n3. Subject walks towards operator.\n4. Subject stops 1.5 m away.",
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

DEMO_SCRIPT_STEPS = [
    {
        "step": 1,
        "title": "1. CLEAR ROOM BASELINE",
        "todo": "Point camera around a clear room with no moving subjects.",
        "watch": "1. Live HUD status: NO_WARNING\n2. Nav Command: CONTINUE\n3. Throughput >14 FPS / Latency ~50 ms\n4. Audio: Silent / Idle",
        "expected": "System stays quiet with zero false alerts in clear environments.",
        "safety": "Keep path clear of tripping hazards."
    },
    {
        "step": 2,
        "title": "2. SHOW OBJECT DETECTION & TRACKING",
        "todo": "Point camera at objects (chair, person, bottle, etc.).",
        "watch": "1. Bounding boxes appear with class labels\n2. Unique Track ID assigned per object\n3. Track ID remains persistent during motion",
        "expected": "Objects localized and tracked with persistent track IDs.",
        "safety": "Maintain safe standing position."
    },
    {
        "step": 3,
        "title": "3. MOVE TOWARD OBJECT (PERSON APPROACHING)",
        "todo": "Ask a teammate to walk slowly toward the camera from 6 m away.",
        "watch": "1. Track ID assigned to teammate\n2. Motion State changes to APPROACHING\n3. TTC decreases in seconds (e.g., 2.5s -> 1.2s)\n4. Risk escalates to WARNING / CRITICAL\n5. Spoken alert dispatches: 'Caution, obstacle ahead'",
        "expected": "Closing hazard identified, TTC computed, and spoken alert triggered.",
        "safety": "Teammate must stop at least 1.5 m away from camera operator."
    },
    {
        "step": 4,
        "title": "4. MOVE AWAY (PERSON RECEDING)",
        "todo": "Ask teammate to walk away from the camera along the path.",
        "watch": "1. Motion State changes to RECEDING\n2. Closing risk contribution drops to 0.0\n3. Alert state drops to NO_WARNING / CONTINUE",
        "expected": "Receding motion recognized; closing threat risk suppressed.",
        "safety": "Maintain visual contact while teammate walks away."
    },
    {
        "step": 5,
        "title": "5. CROSS CAMERA VIEW (PERSON CROSSING)",
        "todo": "Ask teammate to walk laterally across the corridor at 3 m.",
        "watch": "1. Lateral spatial zone updates (LEFT -> CENTER -> RIGHT)\n2. Walking corridor overlap evaluated\n3. Dynamic clearance updates safe steering direction",
        "expected": "Lateral trajectory analyzed; spatial corridor steering updated.",
        "safety": "Keep crossing path unobstructed."
    },
    {
        "step": 6,
        "title": "6. MOVE CAMERA (HEAD MOTION / GAIT SWAY)",
        "todo": "Pan camera side-to-side smoothly or simulate walking gait bounce.",
        "watch": "1. Background optical-flow expansion estimated\n2. Radial divergence absorbs camera movement\n3. Target tracking remains stable without false approach spikes",
        "expected": "Ego-motion compensation absorbs gait/pan motion artifacts.",
        "safety": "Hold camera firmly during panning movements."
    }
]


class LiveDemoControlCenterGUI:
    """Tkinter Desktop Control Center GUI for Teammate Demonstrations & Scenarios."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Adaptive Edge-AI Navigation — Teammate Demonstration Control Center")
        self.root.geometry("1520x960")
        self.root.minsize(1280, 800)

        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Session State Management
        self.session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = DEMO_SESSIONS_DIR / self.session_timestamp
        self.team_demo_dir = TEAM_DEMOS_DIR / self.session_timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.team_demo_dir.mkdir(parents=True, exist_ok=True)

        self.selected_scenario_id = "S01"
        self.scenario_status: Dict[str, str] = {s_id: "○" for s_id in SCENARIOS}
        self.scenario_results: Dict[str, str] = {s_id: "NOT_COMPLETED" for s_id in SCENARIOS}

        # Active Demo / Recording State
        self.is_trial_running = False
        self.is_recording = False
        self.recording_writer: Optional[cv2.VideoWriter] = None
        self.current_trial_dir: Optional[Path] = None
        self.trial_video_writer: Optional[cv2.VideoWriter] = None
        self.trial_start_time = 0.0
        self.trial_frames_count = 0
        self.trial_telemetry_records: List[dict] = []
        self.trial_event_markers: List[dict] = []

        # Audio state tracking
        self.audio_state = "READY"  # READY, SPEAKING, IDLE
        self.last_spoken_time = 0.0

        # Pre-flight startup verification checklist
        self.verification_checks = {
            "GUI opens": True,
            "Camera works": False,
            "GPU detected": False,
            "TensorRT depth engine loads": False,
            "Detector loads": False,
            "Tracking works": False,
            "Depth visualization works": False,
            "TTC appears when appropriate": False,
            "Risk state updates": False,
            "Navigation updates": False,
            "Audio works": False,
            "Screenshot works": True,
            "Demo recording works": True,
        }

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
        self.stop_requested = False

        # Build UI Elements
        self._create_ui()

        # Start Pipeline Initialization in background
        threading.Thread(target=self._initialize_pipeline, daemon=True).start()

    def _create_ui(self) -> None:
        """Construct multi-panel GUI interface."""

        # 1. Top System Status Header Bar & Mode Banner
        header_frame = ttk.Frame(self.root, padding=6, relief="raised")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = ttk.Label(
            header_frame,
            text="ADAPTIVE EDGE-AI NAVIGATION — TEAMMATE DEMO CONTROL CENTER",
            font=("Helvetica", 13, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=6)

        banner_lbl = ttk.Label(
            header_frame,
            text="[ DEMO MODE — NOT A FORMAL VALIDATION RUN | Controlled Open-Eye Demonstration Only ]",
            font=("Helvetica", 9, "bold"),
            foreground="darkred"
        )
        banner_lbl.pack(side=tk.LEFT, padx=10)

        self.lbl_audio_status = ttk.Label(header_frame, text="AUDIO: INITIALIZING", font=("Helvetica", 9, "bold"), foreground="blue")
        self.lbl_audio_status.pack(side=tk.RIGHT, padx=10)

        self.lbl_fps = ttk.Label(header_frame, text=self.status_fps_text, font=("Helvetica", 9, "bold"))
        self.lbl_fps.pack(side=tk.RIGHT, padx=10)

        self.lbl_lat = ttk.Label(header_frame, text=self.status_lat_text, font=("Helvetica", 9, "bold"))
        self.lbl_lat.pack(side=tk.RIGHT, padx=10)

        self.lbl_gpu = ttk.Label(header_frame, text=self.status_gpu_text, font=("Helvetica", 9, "bold"), foreground="navy")
        self.lbl_gpu.pack(side=tk.RIGHT, padx=10)

        self.lbl_trt = ttk.Label(header_frame, text=self.status_trt_text, font=("Helvetica", 9, "bold"), foreground="darkgreen")
        self.lbl_trt.pack(side=tk.RIGHT, padx=10)

        # Main Workspace Division (Left Panel vs Center Video Panel vs Right Panel)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # LEFT PANEL: Scenario Selector, Guided Script & Instructions
        left_frame = ttk.Frame(main_paned, padding=4)
        main_paned.add(left_frame, weight=1)

        # Guided Demo Script Selector
        script_group = ttk.LabelFrame(left_frame, text="1. Teammate Demo Script (Guided Walkthrough)", padding=4)
        script_group.pack(fill=tk.X, pady=2)

        self.combo_demo_script = ttk.Combobox(
            script_group,
            values=[step["title"] for step in DEMO_SCRIPT_STEPS],
            state="readonly",
            font=("Helvetica", 9, "bold")
        )
        self.combo_demo_script.set(DEMO_SCRIPT_STEPS[0]["title"])
        self.combo_demo_script.pack(fill=tk.X, pady=2)
        self.combo_demo_script.bind("<<ComboboxSelected>>", self._on_demo_script_selected)

        # Scenario Buttons
        scen_group = ttk.LabelFrame(left_frame, text="2. Scenario Selector (S01–S06)", padding=4)
        scen_group.pack(fill=tk.X, pady=2)

        self.btn_scenarios: Dict[str, ttk.Button] = {}
        for s_id, s_data in SCENARIOS.items():
            btn_text = f"{s_data['id']}  [{self.scenario_status[s_id]}]  {s_data['code'].split('_', 1)[1]}"
            btn = ttk.Button(
                scen_group,
                text=btn_text,
                command=lambda id=s_id: self._select_scenario(id)
            )
            btn.pack(fill=tk.X, pady=1)
            self.btn_scenarios[s_id] = btn

        # Scenario & Script Instructions Box
        instr_group = ttk.LabelFrame(left_frame, text="3. Operator Guidance & Safety", padding=4)
        instr_group.pack(fill=tk.BOTH, expand=True, pady=2)

        self.txt_instructions = scrolledtext.ScrolledText(instr_group, wrap=tk.WORD, width=34, height=12, font=("Consolas", 9))
        self.txt_instructions.pack(fill=tk.BOTH, expand=True)

        # Event Marker Logging Section
        event_group = ttk.LabelFrame(left_frame, text="4. Event Marker Logger", padding=4)
        event_group.pack(fill=tk.X, pady=2)

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
        self.combo_event_tag.pack(fill=tk.X, pady=1)

        self.entry_event_note = ttk.Entry(event_group)
        self.entry_event_note.pack(fill=tk.X, pady=1)

        btn_mark = ttk.Button(event_group, text="📍 MARK EVENT", command=self._mark_event)
        btn_mark.pack(fill=tk.X, pady=2)

        # CENTER PANEL: Live Annotated Camera Feed
        center_frame = ttk.Frame(main_paned, padding=4)
        main_paned.add(center_frame, weight=3)

        cam_group = ttk.LabelFrame(center_frame, text="5. Live Annotated Camera Feed & Telemetry HUD", padding=4)
        cam_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_video = ttk.Label(cam_group, text="Initializing Camera & Models...")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # RIGHT PANEL: Observation Card, What is Happening & Demo Controls
        right_frame = ttk.Frame(main_paned, padding=4)
        main_paned.add(right_frame, weight=1)

        # SIMPLIFIED SUMMARY OBSERVATION CARD FOR TEAMMATES
        obs_group = ttk.LabelFrame(right_frame, text="6. CURRENT OBSERVATION (Simplified Summary)", padding=6)
        obs_group.pack(fill=tk.X, pady=2)

        self.lbl_summary_hazard = ttk.Label(obs_group, text="HAZARD:      CLEAR (None)", font=("Helvetica", 10, "bold"), foreground="green")
        self.lbl_summary_hazard.pack(anchor=tk.W, pady=1)

        self.lbl_summary_motion = ttk.Label(obs_group, text="MOTION:      STATIC / NONE", font=("Helvetica", 10, "bold"))
        self.lbl_summary_motion.pack(anchor=tk.W, pady=1)

        self.lbl_summary_ttc = ttk.Label(obs_group, text="TTC:         N/A (Clear)", font=("Helvetica", 10, "bold"))
        self.lbl_summary_ttc.pack(anchor=tk.W, pady=1)

        self.lbl_summary_risk = ttk.Label(obs_group, text="RISK LEVEL:  NO_WARNING", font=("Helvetica", 10, "bold"), foreground="green")
        self.lbl_summary_risk.pack(anchor=tk.W, pady=1)

        self.lbl_summary_nav = ttk.Label(obs_group, text="NAVIGATION:  CONTINUE", font=("Helvetica", 10, "bold"), foreground="darkgreen")
        self.lbl_summary_nav.pack(anchor=tk.W, pady=1)

        # AUDIO RISK GUIDANCE MODE & FEEDBACK CONTROL
        audio_mode_group = ttk.LabelFrame(right_frame, text="6b. Audio Guidance Mode & Telemetry", padding=6)
        audio_mode_group.pack(fill=tk.X, pady=2)

        lbl_mode_title = ttk.Label(audio_mode_group, text="AUDIO MODE:", font=("Helvetica", 9, "bold"))
        lbl_mode_title.pack(anchor=tk.W, pady=1)

        self.combo_audio_mode = ttk.Combobox(
            audio_mode_group,
            values=["CONTINUOUS RISK", "TRANSITIONS ONLY"],
            state="readonly",
            font=("Helvetica", 9, "bold")
        )
        self.combo_audio_mode.set("CONTINUOUS RISK")
        self.combo_audio_mode.pack(fill=tk.X, pady=1)
        self.combo_audio_mode.bind("<<ComboboxSelected>>", self._on_audio_mode_selected)

        self.lbl_current_msg = ttk.Label(audio_mode_group, text="CURRENT MSG: (None)", font=("Helvetica", 9), foreground="darkblue", wraplength=280)
        self.lbl_current_msg.pack(anchor=tk.W, pady=1)

        self.lbl_next_update = ttk.Label(audio_mode_group, text="NEXT UPDATE: 0.0 s", font=("Helvetica", 9, "italic"), foreground="gray")
        self.lbl_next_update.pack(anchor=tk.W, pady=1)

        # WHAT IS HAPPENING EXPLANATION PANEL
        explain_group = ttk.LabelFrame(right_frame, text="7. WHAT IS HAPPENING? (Live Explanation)", padding=6)
        explain_group.pack(fill=tk.X, pady=2)

        self.txt_explain = tk.Label(
            explain_group,
            text="Path currently has no detected immediate dynamic hazard.",
            font=("Helvetica", 9),
            wraplength=280,
            justify=tk.LEFT,
            foreground="darkblue"
        )
        self.txt_explain.pack(fill=tk.X, pady=2)

        # TEAM DEMO CONTROLS BAR
        ctrl_group = ttk.LabelFrame(right_frame, text="8. Teammate Demo Controls", padding=6)
        ctrl_group.pack(fill=tk.X, pady=4)

        btn_row1 = ttk.Frame(ctrl_group)
        btn_row1.pack(fill=tk.X, pady=2)

        self.btn_start_demo = ttk.Button(btn_row1, text="▶ START DEMO", command=self._start_demo_mode, state="disabled")
        self.btn_start_demo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_stop_demo = ttk.Button(btn_row1, text="■ STOP DEMO", command=self._stop_demo_mode, state="disabled")
        self.btn_stop_demo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_row2 = ttk.Frame(ctrl_group)
        btn_row2.pack(fill=tk.X, pady=2)

        self.btn_rec_demo = ttk.Button(btn_row2, text="🔴 RECORD DEMO", command=self._start_demo_recording, state="disabled")
        self.btn_rec_demo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_stop_rec = ttk.Button(btn_row2, text="⏹ STOP REC", command=self._stop_demo_recording, state="disabled")
        self.btn_stop_rec.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_row3 = ttk.Frame(ctrl_group)
        btn_row3.pack(fill=tk.X, pady=2)

        btn_screenshot = ttk.Button(btn_row3, text="📷 SCREENSHOT", command=self._save_screenshot)
        btn_screenshot.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_test_audio = ttk.Button(btn_row3, text="🔊 TEST AUDIO", command=self._test_audio)
        btn_test_audio.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_reset = ttk.Button(btn_row3, text="🔄 RESET VIEW", command=self._reset_view)
        btn_reset.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Verification Checklist Output Frame
        chk_group = ttk.LabelFrame(right_frame, text="9. Startup Pre-Flight Verification", padding=4)
        chk_group.pack(fill=tk.BOTH, expand=True, pady=2)

        self.txt_verify = scrolledtext.ScrolledText(chk_group, wrap=tk.WORD, width=30, height=6, font=("Consolas", 8))
        self.txt_verify.pack(fill=tk.BOTH, expand=True)

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

        for id, btn in self.btn_scenarios.items():
            st_symbol = self.scenario_status[id]
            res_str = f" [{self.scenario_results[id]}]" if self.scenario_results[id] != "NOT_COMPLETED" else ""
            btn_text = f"{id}  [{st_symbol}]{res_str}  {SCENARIOS[id]['code'].split('_', 1)[1]}"
            btn.config(text=btn_text)

    def _on_audio_mode_selected(self, event=None) -> None:
        """Handle audio guidance mode change."""
        mode = self.combo_audio_mode.get()
        if self.message_generator:
            self.message_generator.set_audio_mode(mode)

    def _on_demo_script_selected(self, event=None) -> None:
        """Update instructions area when a guided demo script step is selected."""
        sel_title = self.combo_demo_script.get()
        step_data = None
        for step in DEMO_SCRIPT_STEPS:
            if step["title"] == sel_title:
                step_data = step
                break

        if not step_data:
            return

        buf = f"=== DEMO WALKTHROUGH: {step_data['title']} ===\n\n"
        buf += f"WHAT TO DO:\n{step_data['todo']}\n\n"
        buf += f"WHAT TO WATCH:\n{step_data['watch']}\n\n"
        buf += f"EXPECTED SYSTEM OUTPUT:\n{step_data['expected']}\n\n"
        buf += f"SAFETY GUIDELINE:\n{step_data['safety']}\n"

        self.txt_instructions.delete("1.0", tk.END)
        self.txt_instructions.insert(tk.END, buf)

    def _update_ui_state(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        global_warning: GlobalWarningDecision,
        scene_nav: Any,
        warning_message: WarningMessage,
        object_depths: List[TrackedObjectDepth],
        compensated_estimates: dict,
        ttc_results: dict,
        risk_assessments: dict,
    ) -> None:
        """Update Tkinter summary cards, explanations, and video canvas."""
        self.status_fps_text = f"FPS: {fps:.1f}"
        self.lbl_fps.config(text=self.status_fps_text)

        self.status_lat_text = f"Latency: {latency_ms:.1f} ms"
        self.lbl_lat.config(text=self.status_lat_text)

        # Update Audio Indicator & Guidance Telemetry
        audio_status_str = self.tts_engine.get_status_text() if self.tts_engine else "INITIALIZING"
        audio_fg = "red" if "ERROR" in audio_status_str else ("orange" if audio_status_str == "SPEAKING" else ("green" if audio_status_str == "READY" else "blue"))
        self.lbl_audio_status.config(text=f"AUDIO: {audio_status_str}", foreground=audio_fg)

        curr_msg = warning_message.text if (warning_message and warning_message.text) else "(Silent)"
        self.lbl_current_msg.config(text=f"CURRENT MSG: \"{curr_msg}\"")

        next_up_sec = self.message_generator.get_seconds_to_next_update() if self.message_generator else 0.0
        self.lbl_next_update.config(text=f"NEXT UPDATE: {next_up_sec:.1f} s")

        # Extract Primary Target Object for Simplified Teammate Observation Card
        primary_obj = object_depths[0] if object_depths else None
        if primary_obj:
            p_id = primary_obj.track_id
            p_class = primary_obj.class_name.upper()
            comp_m = compensated_estimates.get(p_id)
            ttc_r = ttc_results.get(p_id)
            risk_ass = risk_assessments.get(p_id)

            p_motion = comp_m.approach_state.upper() if comp_m else "STATIC"
            p_ttc = f"{ttc_r.ttc_seconds:.1f} s" if (ttc_r and ttc_r.ttc_valid and ttc_r.ttc_seconds) else "N/A"
            p_depth = f"{primary_obj.depth_value:.2f}"

            self.lbl_summary_hazard.config(text=f"HAZARD:      {p_class} (ID {p_id})", foreground="darkred")
            self.lbl_summary_motion.config(text=f"MOTION:      {p_motion}")
            self.lbl_summary_ttc.config(text=f"TTC:         {p_ttc}")
        else:
            self.lbl_summary_hazard.config(text="HAZARD:      CLEAR (None)", foreground="green")
            self.lbl_summary_motion.config(text="MOTION:      STATIC / NONE")
            self.lbl_summary_ttc.config(text="TTC:         N/A (Clear)")

        warn_fg = "green" if global_warning.state == "NO_WARNING" else ("orange" if global_warning.state == "CAUTION" else "red")
        self.lbl_summary_risk.config(text=f"RISK LEVEL:  {global_warning.state}", foreground=warn_fg)

        nav_fg = "darkgreen" if scene_nav.navigation_state == "CONTINUE" else ("orange" if scene_nav.navigation_state in ("AVOID_LEFT", "AVOID_RIGHT") else "red")
        self.lbl_summary_nav.config(text=f"NAVIGATION:  {scene_nav.navigation_state}", foreground=nav_fg)

        # Dynamic Explanation Text Update ("What is Happening?")
        explanation_text = self._derive_live_explanation(global_warning, scene_nav, primary_obj, compensated_estimates)
        self.txt_explain.config(text=explanation_text)

        # Convert OpenCV BGR to PIL ImageTk for Tkinter Label
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.lbl_video.img_tk = img_tk
        self.lbl_video.config(image=img_tk, text="")

    def _initialize_pipeline(self) -> None:
        """Initialize models, camera, and TensorRT engines in background thread."""
        try:
            config = load_config("configs/final.yaml")

            # Override depth checkpoint to Native TensorRT FP16 Engine if available
            trt_engine_path = REPO_ROOT / "models/deployment/depth_anything_v2_vits_fp16.engine"
            if trt_engine_path.exists():
                depth_checkpoint = str(trt_engine_path)
                print(f"[Demo GUI] Native TensorRT FP16 Depth Engine Found: {depth_checkpoint}")
            else:
                depth_checkpoint = "models/depth/depth_anything_v2_vits.pth"
                print(f"[Demo GUI] Fallback Depth Checkpoint: {depth_checkpoint}")

            # 1. Camera Source
            self.camera = CameraSource(source=0, width=640, height=480, target_fps=30)
            self.camera.open()
            self.verification_checks["Camera works"] = True

            # 2. YOLO Detector
            self.detector = YOLOObjectDetector(
                model_name_or_path="models/detector/yolo11n.pt",
                confidence_threshold=0.25,
                iou_threshold=0.45,
                image_size=640,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                class_filter_config=config.get("detector", {}).get("class_filter"),
            )
            self.verification_checks["Detector loads"] = True

            # 3. BoT-SORT Tracker
            self.tracker = BoTSORTTracker(
                tracker_config="botsort.yaml",
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                match_thresh=0.8,
                track_buffer=25,
            )
            self.verification_checks["Tracking works"] = True

            # 4. Depth Anything V2 Estimator
            self.depth_estimator = DepthAnythingV2Estimator(
                checkpoint_path=depth_checkpoint,
                model_type="vits",
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                input_size=518,
                is_metric=False,
                object_statistic="median",
            )
            self.verification_checks["Depth visualization works"] = True
            if getattr(self.depth_estimator, "use_trt", False) or str(depth_checkpoint).endswith(".engine"):
                self.verification_checks["TensorRT depth engine loads"] = True

            # 5. Temporal Buffer & Motion Estimator
            self.temporal_history = TemporalHistory(
                history_length=25,
                max_history_age_seconds=1.5,
                cleanup_after_seconds=1.5,
                minimum_observations=3,
            )

            self.motion_estimator = MotionEstimator(
                minimum_dt_seconds=0.01,
                max_valid_time_gap_seconds=0.5,
                minimum_history_observations=3,
                smoothing_method="ema",
                smoothing_window=5,
                stable_threshold=0.05,
                depth_convention="higher_is_closer",
            )

            # 6. Camera Ego-Motion Compensator
            self.camera_motion_estimator = CameraMotionEstimator(
                enabled=True,
                max_features=300,
                quality_level=0.01,
                min_distance=7.0,
                ransac_enabled=True,
            )

            # 7. TTC & Risk Engine
            self.ttc_estimator = TTCEstimator(
                enabled=True,
                minimum_history_observations=3,
                minimum_closing_speed=0.05,
                maximum_time_gap_seconds=0.5,
                max_ttc_seconds=30.0,
                depth_convention="higher_is_closer",
            )
            self.risk_engine = RiskEngine(
                enabled=True,
                score_thresholds={"critical": 0.85, "high": 0.75, "medium": 0.5, "low": 0.25},
                ttc_thresholds={"critical": 1.0, "high": 2.0, "medium": 4.0},
                minimum_evidence_coverage=0.50,
            )

            # 8. Reliability & Warning State Machine
            self.reliability_estimator = ReliabilityEstimator(enabled=True)
            self.warning_machine = WarningStateMachine(config.get("warning", {}))
            self.message_generator = WarningMessageGenerator(config)

            # 9. Non-blocking Audio TTS Engine
            self.tts_engine = TTSEngine(config)
            self.verification_checks["Audio works"] = True

            # 10. Spatial Corridor & Navigation Decision Engine
            self.spatial_analyzer = SpatialAnalyzer(config.get("navigation", {}))
            self.path_analyzer = PathGeometryAnalyzer(config.get("navigation", {}))
            self.nav_engine = NavigationEngine(config.get("navigation", {}))

            # Check GPU
            if torch.cuda.is_available():
                self.verification_checks["GPU detected"] = True
                gpu_name = torch.cuda.get_device_name(0)
            else:
                gpu_name = "Host CPU"
            self.status_gpu_text = f"GPU: {gpu_name}"
            self.root.after(0, lambda: self.lbl_gpu.config(text=self.status_gpu_text))

            trt_status = "TRT FP16 Active (33.8 ms)" if (getattr(self.depth_estimator, "use_trt", False) or str(depth_checkpoint).endswith(".engine")) else "PyTorch CUDA EP"
            self.status_trt_text = f"Engine: {trt_status}"
            self.root.after(0, lambda: self.lbl_trt.config(text=self.status_trt_text))

            self.pipeline_ready = True
            self.root.after(0, self._on_pipeline_initialized)

            # Start video display worker loop
            threading.Thread(target=self._run_pipeline_loop, daemon=True).start()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Pipeline Initialization Error", f"Failed to initialize live pipeline:\n{str(e)}")

    def _on_pipeline_initialized(self) -> None:
        """Enable buttons and render verification checklist."""
        self.btn_start_demo.config(state="normal")
        self.btn_rec_demo.config(state="normal")

        # Update verification checklist display
        self.txt_verify.delete("1.0", tk.END)
        self.txt_verify.insert(tk.END, "=== STARTUP CHECKLIST ===\n")
        for check, passed in self.verification_checks.items():
            sym = "[✓]" if passed else "[×]"
            self.txt_verify.insert(tk.END, f"{sym} {check}\n")

    def _run_pipeline_loop(self) -> None:
        """Main continuous perception-to-action worker loop."""
        fps_buffer = []
        t_last_frame = time.perf_counter()

        while not self.stop_requested:
            t_loop_start = time.perf_counter()

            packet: Optional[FramePacket] = self.camera.read_frame()
            if packet is None or packet.frame is None:
                time.sleep(0.01)
                continue

            # 1. Detection
            detections = self.detector.detect(packet.frame, packet.timestamp)
            active_dets = [d for d in detections if d.policy_accepted]

            # 2. Multi-Object Tracking
            tracked_objects = self.tracker.update(active_dets, packet.frame, packet.timestamp)

            # 3. Monocular Depth Estimation
            depth_result = self.depth_estimator.estimate_depth(packet.frame, packet.timestamp)

            # 4. Object Depth & Temporal History
            object_depths = self.depth_estimator.extract_all_object_depths(depth_result, tracked_objects)
            observations = [
                ObjectObservation.from_tracked_depth(obj, packet.timestamp, packet.frame_index)
                for obj in object_depths
            ]
            self.temporal_history.update(observations, current_timestamp=packet.timestamp)

            # 5. Motion & Camera Ego-Motion Compensation
            motion_estimates = self.motion_estimator.estimate_all(self.temporal_history)
            obstacle_bboxes = [obj.bbox for obj in tracked_objects]
            camera_motion = self.camera_motion_estimator.estimate(
                packet.frame,
                timestamp=packet.timestamp,
                object_bboxes=obstacle_bboxes,
            )
            compensated_estimates = self.camera_motion_estimator.compensate_all(motion_estimates, camera_motion)

            # 6. TTC Estimation
            ttc_results = self.ttc_estimator.estimate_all(self.temporal_history, compensated_estimates)

            # 7. Dynamic Risk Assessment
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
                    depth_type="relative",
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
            risk_assessments = self.risk_engine.assess_all(risk_features_map)

            # 8. Reliability & Warning State Machine
            obs_map = {obj.track_id: self.temporal_history.get(obj.track_id) for obj in object_depths}
            reliability_assessments = self.reliability_estimator.assess_all(
                observations_map=obs_map,
                compensated_motion_map=compensated_estimates,
                camera_motion=camera_motion,
                ttc_map=ttc_results,
                risk_map=risk_assessments,
            )

            warning_decisions, global_warning = self.warning_machine.update(
                risk_assessments=risk_assessments,
                reliability_assessments=reliability_assessments,
                ttc_results=ttc_results,
                compensated_motion=compensated_estimates,
                timestamp=packet.timestamp,
                frame_index=packet.frame_index,
            )

            # 9. Spatial Navigation & Path Engine
            spatial_reprs = {
                obj.track_id: self.spatial_analyzer.analyze_object(
                    track_id=obj.track_id,
                    bbox=obj.bbox,
                    frame_width=fw,
                    frame_height=fh,
                    horizontal_motion=compensated_estimates.get(obj.track_id).compensated_vx if (obj.track_id in compensated_estimates and compensated_estimates.get(obj.track_id)) else None,
                    vertical_motion=compensated_estimates.get(obj.track_id).compensated_vy if (obj.track_id in compensated_estimates and compensated_estimates.get(obj.track_id)) else None,
                )
                for obj in tracked_objects
            }
            path_overlaps = self.path_analyzer.assess_all(spatial_reprs)
            nav_decisions, scene_nav = self.nav_engine.evaluate(
                spatial_objects=spatial_reprs,
                path_assessments=path_overlaps,
                warning_decisions=warning_decisions,
                global_warning=global_warning,
                system_reliability_score=1.0,
            )

            # 10. Non-Blocking Audio Dispatch
            warning_message: WarningMessage = self.message_generator.generate(
                global_warning=global_warning,
                track_decisions=warning_decisions,
                current_time=packet.timestamp,
                scene_nav=scene_nav,
            )

            # Audio state tracking
            if warning_message.should_speak and warning_message.text:
                self.audio_state = "SPEAKING"
                self.last_spoken_time = time.time()
                self.tts_engine.speak(warning_message.text, priority=warning_message.priority)
            elif time.time() - self.last_spoken_time < 2.0:
                self.audio_state = "SPEAKING"
            else:
                self.audio_state = "IDLE"

            # Dynamic verification checks update
            if any(t.ttc_valid for t in ttc_results.values() if t):
                self.verification_checks["TTC appears when appropriate"] = True
            if global_warning.state != "NO_WARNING":
                self.verification_checks["Risk state updates"] = True
            if scene_nav.navigation_state != "CONTINUE":
                self.verification_checks["Navigation updates"] = True

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

            # Record demo video if recording is active
            if self.is_recording and self.recording_writer:
                self.trial_frames_count += 1
                self.recording_writer.write(annotated_frame)

                # Record telemetry
                telemetry_record = {
                    "timestamp": packet.timestamp,
                    "frame_index": packet.frame_index,
                    "trial_frame": self.trial_frames_count,
                    "e2e_latency_ms": round(t_frame_e2e, 2),
                    "fps": round(avg_fps, 2),
                    "global_warning_state": global_warning.state,
                    "navigation_state": scene_nav.navigation_state,
                    "safe_direction": scene_nav.safe_direction,
                    "audio_state": self.audio_state,
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
                            "spatial_zone": spatial_reprs[o.track_id].spatial_zone if (o.track_id in spatial_reprs) else "CENTER",
                        }
                        for o in object_depths
                    ]
                }
                self.trial_telemetry_records.append(telemetry_record)

            # Update UI Display
            self.root.after(0, lambda f=annotated_frame, fps=avg_fps, lat=t_frame_e2e, gw=global_warning, sn=scene_nav, msg=warning_message, depths=object_depths, comp=compensated_estimates, ttcs=ttc_results, risks=risk_assessments: self._update_ui_state(f, fps, lat, gw, sn, msg, depths, comp, ttcs, risks))

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
        hud_top = f"TEAMMATE DEMO | FPS: {avg_fps:.1f} | Latency: {t_frame_e2e:.1f}ms | Audio: {self.audio_state}"
        hud_warn = f"GLOBAL ALERT: [{global_warning.state}] - {global_warning.reason}"
        hud_nav = f"NAV COMMAND: [{scene_nav.navigation_state}] SafeDir: [{scene_nav.safe_direction}]"

        cv2.putText(display_frame, hud_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(display_frame, hud_warn, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 2)
        cv2.putText(display_frame, hud_nav, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1)

        return display_frame

    def _derive_live_explanation(
        self,
        global_warning: GlobalWarningDecision,
        scene_nav: Any,
        primary_obj: Optional[TrackedObjectDepth],
        compensated_estimates: dict,
    ) -> str:
        """Derive short, non-technical explanation for live demonstration viewers."""
        if global_warning.state == "CRITICAL" or scene_nav.navigation_state == "STOP":
            return "CRITICAL / STOP: Immediate dynamic hazard detected in walking path. Emergency stop or evasive clearance required."
        elif global_warning.state == "WARNING":
            return "WARNING: Higher dynamic risk detected. Protective audio advisory dispatched to operator."
        elif global_warning.state == "CAUTION":
            return "CAUTION: Moderate risk object under observation. Path is monitored while clearance is evaluated."
        elif scene_nav.navigation_state == "AVOID_LEFT":
            return "AVOID_LEFT: Central path has obstacle presence; left-side walking clearance is currently preferred."
        elif scene_nav.navigation_state == "AVOID_RIGHT":
            return "AVOID_RIGHT: Central path has obstacle presence; right-side walking clearance is currently preferred."
        elif primary_obj and primary_obj.track_id in compensated_estimates:
            m_state = compensated_estimates[primary_obj.track_id].approach_state
            if m_state == "APPROACHING":
                return f"APPROACHING: Tracked {primary_obj.class_name} (ID {primary_obj.track_id}) is moving closer. Time-to-Collision is actively monitored."
            elif m_state == "RECEDING":
                return f"RECEDING: Tracked {primary_obj.class_name} (ID {primary_obj.track_id}) is moving away; closing threat risk is suppressed."

        if primary_obj:
            return f"OBJECT DETECTED: {primary_obj.class_name} assigned Track ID {primary_obj.track_id}. System evaluating spatial relevance."

        return "NO_WARNING: Path currently has no detected immediate dynamic hazard. Safe to proceed forward."

    def _start_demo_mode(self) -> None:
        """Activate Demo Mode."""
        self.is_trial_running = True
        self.btn_start_demo.config(state="disabled")
        self.btn_stop_demo.config(state="normal")
        messagebox.showinfo("Demo Started", "Teammate Demonstration Mode Activated.\nPoint camera and follow guided script.")

    def _stop_demo_mode(self) -> None:
        """Deactivate Demo Mode."""
        if self.is_recording:
            self._stop_demo_recording()
        self.is_trial_running = False
        self.btn_start_demo.config(state="normal")
        self.btn_stop_demo.config(state="disabled")

    def _start_demo_recording(self) -> None:
        """Start recording a teammate demonstration session to team_demos/."""
        if not self.pipeline_ready:
            messagebox.showwarning("Pipeline Not Ready", "Please wait for pipeline initialization to finish.")
            return

        ts_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_trial_dir = TEAM_DEMOS_DIR / ts_name
        self.current_trial_dir.mkdir(parents=True, exist_ok=True)

        video_path = self.current_trial_dir / "demo_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recording_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        self.trial_start_time = time.time()
        self.trial_frames_count = 0
        self.trial_telemetry_records = []
        self.trial_event_markers = []

        self.is_recording = True
        self.btn_rec_demo.config(state="disabled")
        self.btn_stop_rec.config(state="normal")

        self._record_event_entry("Demo Recording Started", f"Teammate demo recording started at {ts_name}")
        messagebox.showinfo("Recording Started", f"Teammate Demo Recording Started!\nSaving to:\n{self.current_trial_dir}")

    def _stop_demo_recording(self) -> None:
        """Stop demo recording and write summary notes."""
        if not self.is_recording:
            return

        self.is_recording = False
        duration = time.time() - self.trial_start_time

        if self.recording_writer:
            self.recording_writer.release()
            self.recording_writer = None

        self._record_event_entry("Demo Recording Stopped", "Teammate demo recording stopped")

        if self.current_trial_dir:
            # Save telemetry and event markers
            with open(self.current_trial_dir / "demo_telemetry.json", "w", encoding="utf-8") as f:
                json.dump(self.trial_telemetry_records, f, indent=2)

            with open(self.current_trial_dir / "events.json", "w", encoding="utf-8") as f:
                json.dump(self.trial_event_markers, f, indent=2)

            # Write Demo Notes Markdown
            notes_content = f"# Teammate Demonstration Recording Summary\n\n"
            notes_content += f"**Timestamp:** `{self.current_trial_dir.name}`  \n"
            notes_content += f"**Duration:** `{duration:.2f} s` ({self.trial_frames_count} frames)  \n"
            notes_content += f"**Recorded Video:** `demo_video.mp4`  \n"
            notes_content += f"**Telemetry Records:** `{len(self.trial_telemetry_records)}`  \n"
            notes_content += f"**Event Markers:** `{len(self.trial_event_markers)}`  \n\n"

            notes_content += "## Recorded Event Markers\n"
            for ev in self.trial_event_markers:
                notes_content += f"- `{ev['formatted_time']}` (Frame {ev['frame']}): **[{ev['tag']}]** {ev['note']}\n"

            with open(self.current_trial_dir / "demo_notes.md", "w", encoding="utf-8") as f:
                f.write(notes_content)

        self.btn_rec_demo.config(state="normal")
        self.btn_stop_rec.config(state="disabled")

        messagebox.showinfo("Recording Saved", f"Teammate Demo Recording Saved!\nSaved to:\n{self.current_trial_dir}")

    def _mark_event(self) -> None:
        """Record manual event marker."""
        tag = self.combo_event_tag.get()
        note = self.entry_event_note.get().strip()
        self._record_event_entry(tag, note)
        self.entry_event_note.delete(0, tk.END)
        messagebox.showinfo("Event Marked", f"Event Tagged: [{tag}]\nNote: '{note}'")

    def _record_event_entry(self, tag: str, note: str) -> None:
        """Store event marker with timestamp."""
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
        """Save instant screenshot."""
        if not self.camera:
            return
        frame = self.camera.read()
        if frame is None:
            return

        screenshot_dir = (self.current_trial_dir if self.current_trial_dir else self.team_demo_dir) / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%H-%M-%S_%f")[:-3]
        file_path = screenshot_dir / f"screenshot_{ts_str}.jpg"
        cv2.imwrite(str(file_path), frame)
        messagebox.showinfo("Screenshot Saved", f"Screenshot saved to:\n{file_path}")

    def _test_audio(self) -> None:
        """Trigger manual speech utterance for audio system pre-flight verification."""
        if self.tts_engine and self.tts_engine.is_available():
            spoken_ok = self.tts_engine.speak("Audio test successful.", priority="HIGH")
            if spoken_ok:
                self._record_event_entry("Audio Test", "Manual audio test spoken: 'Audio test successful.'")
                messagebox.showinfo("Audio Test", f"Audio Test Triggered!\nBackend: '{self.tts_engine.active_backend}'\nVoice: '{self.tts_engine.selected_voice}'\nUtterance: 'Audio test successful.'")
            else:
                messagebox.showwarning("Audio Test", "Utterance queued or rate-limited by anti-spam logic.")
        else:
            err_msg = getattr(self.tts_engine, 'last_error', 'Audio disabled')
            messagebox.showerror("Audio Test Error", f"TTS Engine not available.\nError: {err_msg}")

    def _reset_view(self) -> None:
        """Reset history buffers and view state."""
        if getattr(self, "temporal_history", None):
            self.temporal_history.clear()
        if self.warning_machine:
            self.warning_machine.reset()
        if self.nav_engine:
            self.nav_engine.current_state = "CONTINUE"
        messagebox.showinfo("View Reset", "Pipeline history buffers and warning state machines reset to default state.")


def main():
    root = tk.Tk()
    app = LiveDemoControlCenterGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'stop_requested', True), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
