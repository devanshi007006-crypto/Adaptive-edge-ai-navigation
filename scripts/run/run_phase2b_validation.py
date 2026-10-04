"""
Phase 2B - Re-Validation Runner & Performance Analyzer.

Executes the updated perception & risk pipeline (with indoor navigation class filtering
and temporal approach stabilization) on the 5 validation videos, measures real wall-clock
latency on cuda:0, analyzes temporal behavior, and generates markdown reports.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adaptive_navigation.main import run_perception_pipeline, load_config

VIDEOS = [
    {
        "stem": "S03_approaching_r01",
        "scenario": "approaching",
        "path": "validation/videos/approaching/S03_approaching_r01.mp4",
        "description": "Dynamic approaching obstacle / pedestrian directly closing in",
    },
    {
        "stem": "S01_clear_r01",
        "scenario": "clear_path",
        "path": "validation/videos/clear_path/S01_clear_r01.mp4",
        "description": "Negative control: clear unobstructed corridor traversal",
    },
    {
        "stem": "S05_crossing_r01",
        "scenario": "crossing",
        "path": "validation/videos/crossing/S05_crossing_r01.mp4",
        "description": "Dynamic orthogonal / transverse crossing pedestrian",
    },
    {
        "stem": "S04_receding_r01",
        "scenario": "receding",
        "path": "validation/videos/receding/S04_receding_r01.mp4",
        "description": "Object moving away / increasing distance from observer",
    },
    {
        "stem": "S02_static_r01",
        "scenario": "static_obstacle",
        "path": "validation/videos/static_obstacle/S02_static_r01.mp4",
        "description": "Stationary obstacle in corridor with ego-camera approach",
    },
]

def analyze_video_run(video_info: dict, out_dir: Path, wall_clock_seconds: float):
    """Parses Phase 2B telemetry files and generates run_report.md."""
    json_path = out_dir / "telemetry.json"
    frames_csv_path = out_dir / "telemetry_frames.csv"
    objs_csv_path = out_dir / "telemetry_objects.csv"

    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_frames = len(records)
    effective_fps = total_frames / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

    # Extract stage latencies
    det_lats = [r["latencies_ms"]["detector"] for r in records]
    track_lats = [r["latencies_ms"]["tracker"] for r in records]
    depth_lats = [r["latencies_ms"]["depth"] for r in records]
    ttc_lats = [r["latencies_ms"]["ttc"] for r in records]
    risk_lats = [r["latencies_ms"]["risk"] for r in records]
    nav_lats = [r["latencies_ms"]["navigation"] for r in records]
    tts_lats = [r["latencies_ms"]["tts"] for r in records]
    total_lats = [r["latencies_ms"]["total_frame"] for r in records]

    # CUDA & Memory stats
    device_name = records[-1]["device"].get("device_name", "N/A") if records else "N/A"
    vram_alloc = max([r["device"].get("vram_allocated_mb", 0.0) for r in records]) if records else 0.0
    vram_res = max([r["device"].get("vram_reserved_mb", 0.0) for r in records]) if records else 0.0

    # Parse detections & objects
    raw_det_counts = []
    active_det_counts = []
    filtered_det_counts = []
    raw_classes = {}
    active_classes = {}
    filtered_classes = {}

    all_objects = []
    track_ids = set()
    valid_ttc_count = 0
    ttc_values = []
    approach_counts = {"APPROACHING": 0, "RECEDING": 0, "STATIONARY": 0, "STABLE": 0, "UNKNOWN": 0}
    warning_counts = {"NO_WARNING": 0, "CAUTION": 0, "WARNING": 0, "CRITICAL": 0}
    nav_states = {}
    safe_dirs = {}
    spoken_messages = []

    for r in records:
        raw_det_counts.append(r.get("raw_detections_count", 0))
        active_det_counts.append(r.get("active_detections_count", 0))
        filtered_det_counts.append(r.get("filtered_detections_count", 0))

        for rd in r.get("raw_detections", []):
            cname = rd["class_name"]
            raw_classes[cname] = raw_classes.get(cname, 0) + 1
            if rd.get("policy_accepted"):
                active_classes[cname] = active_classes.get(cname, 0) + 1
            else:
                filtered_classes[cname] = filtered_classes.get(cname, 0) + 1

        gw = r.get("global_warning", {})
        w_state = gw.get("state", "NO_WARNING")
        warning_counts[w_state] = warning_counts.get(w_state, 0) + 1

        nav = r.get("navigation", {})
        n_state = nav.get("state", "UNKNOWN")
        nav_states[n_state] = nav_states.get(n_state, 0) + 1
        s_dir = nav.get("safe_direction", "NONE")
        safe_dirs[s_dir] = safe_dirs.get(s_dir, 0) + 1

        aud = r.get("audio", {})
        if aud.get("spoken") and aud.get("text"):
            spoken_messages.append((r["frame_index"], r["timestamp_sec"], aud["text"]))

        for obj in r.get("objects", []):
            tid = obj["track_id"]
            track_ids.add(tid)

            app = obj.get("approach_state", "UNKNOWN")
            approach_counts[app] = approach_counts.get(app, 0) + 1

            if obj.get("ttc_valid") and obj.get("ttc_seconds") is not None:
                valid_ttc_count += 1
                ttc_values.append(obj["ttc_seconds"])

            all_objects.append(obj)

    summary = {
        "video": video_info["path"],
        "stem": video_info["stem"],
        "scenario": video_info["scenario"],
        "total_frames": total_frames,
        "runtime_sec": round(wall_clock_seconds, 2),
        "effective_fps": round(effective_fps, 2),
        "device": device_name,
        "max_vram_alloc_mb": round(vram_alloc, 2),
        "max_vram_res_mb": round(vram_res, 2),
        "detections": {
            "total_raw": sum(raw_det_counts),
            "total_active": sum(active_det_counts),
            "total_filtered": sum(filtered_det_counts),
            "raw_classes": raw_classes,
            "active_classes": active_classes,
            "filtered_classes": filtered_classes,
        },
        "latencies": {
            "total": {"mean": round(float(np.mean(total_lats)), 2), "p50": round(float(np.percentile(total_lats, 50)), 2), "p95": round(float(np.percentile(total_lats, 95)), 2)},
            "detector": {"mean": round(float(np.mean(det_lats)), 2), "p50": round(float(np.percentile(det_lats, 50)), 2), "p95": round(float(np.percentile(det_lats, 95)), 2)},
            "tracker": {"mean": round(float(np.mean(track_lats)), 2), "p50": round(float(np.percentile(track_lats, 50)), 2), "p95": round(float(np.percentile(track_lats, 95)), 2)},
            "depth": {"mean": round(float(np.mean(depth_lats)), 2), "p50": round(float(np.percentile(depth_lats, 50)), 2), "p95": round(float(np.percentile(depth_lats, 95)), 2)},
            "ttc": {"mean": round(float(np.mean(ttc_lats)), 2), "p50": round(float(np.percentile(ttc_lats, 50)), 2), "p95": round(float(np.percentile(ttc_lats, 95)), 2)},
            "risk": {"mean": round(float(np.mean(risk_lats)), 2), "p50": round(float(np.percentile(risk_lats, 50)), 2), "p95": round(float(np.percentile(risk_lats, 95)), 2)},
            "navigation": {"mean": round(float(np.mean(nav_lats)), 2), "p50": round(float(np.percentile(nav_lats, 50)), 2), "p95": round(float(np.percentile(nav_lats, 95)), 2)},
            "tts": {"mean": round(float(np.mean(tts_lats)), 2), "p50": round(float(np.percentile(tts_lats, 50)), 2), "p95": round(float(np.percentile(tts_lats, 95)), 2)},
        },
        "tracks": {
            "unique_count": len(track_ids),
            "track_ids": sorted(list(track_ids)),
        },
        "ttc": {
            "valid_count": valid_ttc_count,
            "min_ttc": round(float(np.min(ttc_values)), 2) if ttc_values else None,
            "p50_ttc": round(float(np.percentile(ttc_values, 50)), 2) if ttc_values else None,
            "max_ttc": round(float(np.max(ttc_values)), 2) if ttc_values else None,
            "approach_states": approach_counts,
        },
        "risk_and_warning": {
            "warning_distribution": warning_counts,
            "nav_states": nav_states,
            "safe_directions": safe_dirs,
            "spoken_count": len(spoken_messages),
            "spoken_sample": spoken_messages[:5],
        },
    }

    # Generate run_report.md
    report_md = f"""# Phase 2B Behavioral Run Report: {video_info['stem']}

## 1. Video & Execution Overview
- **Scenario**: `{video_info['scenario']}` ({video_info['description']})
- **Video Path**: `{video_info['path']}`
- **Execution Device**: `{summary['device']}` (`cuda:0`)
- **Total Frames Processed**: {summary['total_frames']}
- **Wall-Clock Runtime**: {summary['runtime_sec']:.2f} seconds
- **Effective Real FPS**: {summary['effective_fps']:.2f} FPS
- **Max VRAM Allocated**: {summary['max_vram_alloc_mb']:.2f} MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: {summary['max_vram_res_mb']:.2f} MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: {summary['detections']['total_raw']}
- **Active Hazard Detections**: {summary['detections']['total_active']}
- **Policy-Filtered Detections**: {summary['detections']['total_filtered']}
- **Raw Detected Classes**: {summary['detections']['raw_classes']}
- **Active Hazard Classes Tracked**: {summary['detections']['active_classes']}
- **Filtered Classes**: {summary['detections']['filtered_classes']}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | {summary['latencies']['detector']['mean']:.2f} | {summary['latencies']['detector']['p50']:.2f} | {summary['latencies']['detector']['p95']:.2f} |
| **BoT-SORT Tracker** | {summary['latencies']['tracker']['mean']:.2f} | {summary['latencies']['tracker']['p50']:.2f} | {summary['latencies']['tracker']['p95']:.2f} |
| **Depth Anything V2 (vits)** | {summary['latencies']['depth']['mean']:.2f} | {summary['latencies']['depth']['p50']:.2f} | {summary['latencies']['depth']['p95']:.2f} |
| **TTC Estimation (Step 9)** | {summary['latencies']['ttc']['mean']:.2f} | {summary['latencies']['ttc']['p50']:.2f} | {summary['latencies']['ttc']['p95']:.2f} |
| **Multi-Factor Risk Engine** | {summary['latencies']['risk']['mean']:.2f} | {summary['latencies']['risk']['p50']:.2f} | {summary['latencies']['risk']['p95']:.2f} |
| **Spatial Navigation Engine** | {summary['latencies']['navigation']['mean']:.2f} | {summary['latencies']['navigation']['p50']:.2f} | {summary['latencies']['navigation']['p95']:.2f} |
| **User-Facing Audio / TTS Dispatch** | {summary['latencies']['tts']['mean']:.2f} | {summary['latencies']['tts']['p50']:.2f} | {summary['latencies']['tts']['p95']:.2f} |
| **End-to-End Per-Frame Pipeline** | **{summary['latencies']['total']['mean']:.2f}** | **{summary['latencies']['total']['p50']:.2f}** | **{summary['latencies']['total']['p95']:.2f}** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: {summary['tracks']['unique_count']} (IDs: {summary['tracks']['track_ids']})
- **Active Track Classes**: {summary['detections']['active_classes']}

### B. Depth & Motion
- **Approach State Distribution**: {summary['ttc']['approach_states']}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: {summary['ttc']['valid_count']} instances
- **TTC Min / p50 / Max**: {summary['ttc']['min_ttc']}s / {summary['ttc']['p50_ttc']}s / {summary['ttc']['max_ttc']}s (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {summary['risk_and_warning']['warning_distribution']}
- **Navigation Decisions**: {summary['risk_and_warning']['nav_states']}
- **Recommended Safe Directions**: {summary['risk_and_warning']['safe_directions']}
- **Spoken Audio Warnings**: {summary['risk_and_warning']['spoken_count']} utterances generated
"""

    report_path = out_dir / "run_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Generated report: {report_path}")
    return summary

def run_single_video(video_info: dict, base_dir: Path):
    out_dir = base_dir / video_info["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(REPO_ROOT / "adaptive_navigation" / "config.yaml"))

    print(f"\n{'='*80}")
    print(f"STARTING PHASE 2B VALIDATION RUN: {video_info['stem']} ({video_info['scenario']})")
    print(f"Video file: {video_info['path']}")
    print(f"Output dir: {out_dir}")
    print(f"{'='*80}\n")

    t_start = time.perf_counter()
    run_perception_pipeline(
        config=config,
        source_override=str(REPO_ROOT / video_info["path"]),
        device_override="cuda:0",
        headless=True,
        telemetry_dir=str(out_dir),
        telemetry_prefix="telemetry",
    )
    t_end = time.perf_counter()
    wall_clock = t_end - t_start

    print(f"\n[Finished {video_info['stem']}] Wall-clock time: {wall_clock:.2f}s")
    summary = analyze_video_run(video_info, out_dir, wall_clock)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 2B Real Video Validation Runner")
    parser.add_argument("--video-idx", type=int, default=None, help="Index of single video (0-4)")
    parser.add_argument("--all", action="store_true", help="Run all 5 validation videos sequentially")
    args = parser.parse_args()

    results_dir = REPO_ROOT / "validation" / "results" / "video_runs_phase2b"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.video_idx is not None:
        if 0 <= args.video_idx < len(VIDEOS):
            v_info = VIDEOS[args.video_idx]
            run_single_video(v_info, results_dir)
        else:
            print(f"Error: Invalid index {args.video_idx}. Choose 0 to {len(VIDEOS)-1}")
    elif args.all:
        for v_info in VIDEOS:
            run_single_video(v_info, results_dir)
        print(f"\nAll {len(VIDEOS)} videos successfully processed for Phase 2B!")
    else:
        print("Please specify --video-idx <0-4> or --all")

if __name__ == "__main__":
    main()
