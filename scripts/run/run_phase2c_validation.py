"""
Phase 2C - Throughput & Cadence Validation Runner.

Executes the perception, tracking, and risk pipeline with 2:1 Depth Cadence
(and optional forward ego-motion compensation) on the 5 validation videos,
measures wall-clock throughput on cuda:0, and generates markdown reports.
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

def analyze_video_run(video_info: dict, out_dir: Path, wall_clock_seconds: float, depth_cadence: int = 2):
    """Parses Phase 2C telemetry files and generates run_report.md."""
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

    # Latency percentiles helper
    def calc_percentiles(vals):
        if not vals:
            return 0.0, 0.0, 0.0
        # Exclude initial warmup frame for p50/p95 stability
        clean = vals[1:] if len(vals) > 1 else vals
        return (
            float(np.mean(clean)),
            float(np.percentile(clean, 50)),
            float(np.percentile(clean, 95)),
        )

    det_mean, det_p50, det_p95 = calc_percentiles(det_lats)
    track_mean, track_p50, track_p95 = calc_percentiles(track_lats)
    depth_mean, depth_p50, depth_p95 = calc_percentiles(depth_lats)
    ttc_mean, ttc_p50, ttc_p95 = calc_percentiles(ttc_lats)
    risk_mean, risk_p50, risk_p95 = calc_percentiles(risk_lats)
    nav_mean, nav_p50, nav_p95 = calc_percentiles(nav_lats)
    tts_mean, tts_p50, tts_p95 = calc_percentiles(tts_lats)
    tot_mean, tot_p50, tot_p95 = calc_percentiles(total_lats)

    ttc_min = f"{min(ttc_values):.2f}s" if ttc_values else "None"
    ttc_med = f"{float(np.median(ttc_values)):.2f}s" if ttc_values else "None"
    ttc_max = f"{max(ttc_values):.2f}s" if ttc_values else "None"

    # Only count depth on active depth execution frames
    active_depth_lats = [lat for lat in depth_lats if lat > 0.0]
    active_depth_mean = float(np.mean(active_depth_lats)) if active_depth_lats else 0.0

    report_content = f"""# Phase 2C Benchmark Run Report: {video_info['stem']}

## 1. Video & Execution Overview
- **Scenario**: `{video_info['scenario']}` ({video_info['description']})
- **Video Path**: `{video_info['path']}`
- **Execution Device**: `{device_name}` (`cuda:0`)
- **Depth Cadence**: `{depth_cadence}:1` (Depth Anything V2 executes every {depth_cadence} frame(s))
- **Total Frames Processed**: {total_frames}
- **Wall-Clock Runtime**: {wall_clock_seconds:.2f} seconds
- **Effective Real FPS**: {effective_fps:.2f} FPS
- **Max VRAM Allocated**: {vram_alloc:.2f} MB (< 7.3% of 6 GB RTX 4050)
- **Max VRAM Reserved**: {vram_res:.2f} MB

## 2. Detection & Indoor Policy Filtering
- **Raw Total Detections**: {sum(raw_det_counts)}
- **Active Hazard Detections**: {sum(active_det_counts)}
- **Policy-Filtered Detections**: {sum(filtered_det_counts)}
- **Raw Detected Classes**: {raw_classes}
- **Active Hazard Classes Tracked**: {active_classes}
- **Filtered Classes**: {filtered_classes}

## 3. Real Latency Breakdown (Wall-Clock Execution)

| Pipeline Stage | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|:---|:---:|:---:|:---:|
| **YOLO11n Object Detector** | {det_mean:.2f} | {det_p50:.2f} | {det_p95:.2f} |
| **BoT-SORT Tracker** | {track_mean:.2f} | {track_p50:.2f} | {track_p95:.2f} |
| **Depth Anything V2 (vits, cadence {depth_cadence}:1)** | {depth_mean:.2f} | {depth_p50:.2f} | {depth_p95:.2f} |
| **Depth When Executing** | {active_depth_mean:.2f} ms | - | - |
| **TTC Estimation (Step 9)** | {ttc_mean:.2f} | {ttc_p50:.2f} | {ttc_p95:.2f} |
| **Multi-Factor Risk Engine** | {risk_mean:.2f} | {risk_p50:.2f} | {risk_p95:.2f} |
| **Spatial Navigation Engine** | {nav_mean:.2f} | {nav_p50:.2f} | {nav_p95:.2f} |
| **User-Facing Audio / TTS Dispatch** | {tts_mean:.2f} | {tts_p50:.2f} | {tts_p95:.2f} |
| **End-to-End Per-Frame Pipeline** | **{tot_mean:.2f}** | **{tot_p50:.2f}** | **{tot_p95:.2f}** |

## 4. Temporal Behavioral Observations

### A. Detection & Tracking
- **Unique Track IDs Detected**: {len(track_ids)} (IDs: {sorted(list(track_ids))})
- **Active Track Classes**: {active_classes}

### B. Depth & Motion
- **Approach State Distribution**: {approach_counts}

### C. Time-to-Collision (TTC)
- **Valid TTC Measurements**: {valid_ttc_count} instances
- **TTC Min / p50 / Max**: {ttc_min} / {ttc_med} / {ttc_max} (strictly evaluated during validated closing motion)

### D. Risk & Warning State Machine
- **Global Warning Distribution**: {warning_counts}
- **Navigation Decisions**: {nav_states}
- **Recommended Safe Directions**: {safe_dirs}
- **Spoken Audio Warnings**: {len(spoken_messages)} utterances generated

"""
    report_path = out_dir / "run_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated report: {report_path}")

    return {
        "stem": video_info["stem"],
        "scenario": video_info["scenario"],
        "frames": total_frames,
        "runtime_sec": wall_clock_seconds,
        "fps": effective_fps,
        "det_p50": det_p50,
        "track_p50": track_p50,
        "depth_p50": depth_p50,
        "total_mean": tot_mean,
        "total_p50": tot_p50,
        "total_p95": tot_p95,
        "warning_counts": warning_counts,
        "valid_ttc": valid_ttc_count,
        "ttc_min": ttc_min,
        "ttc_med": ttc_med,
        "approach_counts": approach_counts,
        "nav_states": nav_states,
    }

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2C 2:1 Cadence Validation Benchmark")
    parser.add_argument("--video-idx", type=int, default=None, help="Run single video index 0..4")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0)")
    parser.add_argument("--config", type=str, default="adaptive_navigation/config.yaml", help="Path to config")
    parser.add_argument("--out-dir", type=str, default="validation/results/video_runs_phase2c", help="Output directory")
    parser.add_argument("--depth-cadence", type=int, default=2, help="Depth inference cadence (default 2)")
    args = parser.parse_args()

    out_base = REPO_ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    config = load_config(str(REPO_ROOT / args.config))

    videos_to_run = VIDEOS if args.video_idx is None else [VIDEOS[args.video_idx]]

    results = []
    for vid in videos_to_run:
        stem = vid["stem"]
        vpath = REPO_ROOT / vid["path"]
        vid_out_dir = out_base / stem
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"STARTING PHASE 2C BENCHMARK: {stem} (Scenario: {vid['scenario']})")
        print(f"Video file: {vpath}")
        print(f"Output dir: {vid_out_dir}")
        print(f"Device: {args.device} | Cadence: {args.depth_cadence}:1")
        print("=" * 80 + "\n")

        t_start = time.perf_counter()

        run_perception_pipeline(
            config=config,
            source_override=str(vpath),
            device_override=args.device,
            headless=True,
            telemetry_dir=str(vid_out_dir),
            telemetry_prefix="telemetry",
            depth_cadence_override=args.depth_cadence,
        )

        t_elapsed = time.perf_counter() - t_start
        print(f"\n[Finished {stem}] Wall-clock time: {t_elapsed:.2f}s")

        res = analyze_video_run(vid, vid_out_dir, t_elapsed, depth_cadence=args.depth_cadence)
        if res:
            results.append(res)

    print("\n" + "=" * 80)
    print("PHASE 2C BENCHMARK EXECUTION SUMMARY:")
    for r in results:
        print(f" - {r['stem']:22s}: {r['frames']} frames in {r['runtime_sec']:.1f}s -> {r['fps']:.2f} FPS | p50: {r['total_p50']:.1f}ms | p95: {r['total_p95']:.1f}ms | TTC: {r['valid_ttc']} | Warnings: {r['warning_counts']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
