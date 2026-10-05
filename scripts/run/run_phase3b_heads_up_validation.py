"""
Phase 3B — Systematic HEADS-UP External Validation Runner.

Executes the validated Phase 2C perception and risk pipeline on the 8-sequence
stratified HEADS-UP egocentric suite (~1,010 real frames), measures real wall-clock
throughput on cuda:0 (RTX 4050), logs real telemetry, and generates comprehensive
metric files and per-sequence run reports.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (cuda:0)
Pipeline: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Cadence) + Optical-Flow Ego-Motion Compensation
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

HEADS_UP_PHASE3B_SEQUENCES = [
    {
        "id": "HU_U01_multiped",
        "name": "HU_U01 Multi-Pedestrian",
        "scenario": "single & multiple pedestrians, cluttered scene",
        "desc": "Multi-pedestrian egocentric scene with 3 simultaneous active pedestrians (Agents 1, 2, 3) in camera FOV.",
        "start_frame": 0,
        "end_frame": 120,
    },
    {
        "id": "HU_U02_approach",
        "name": "HU_U02 Steady Approach",
        "scenario": "approaching pedestrian, normal walking, sparse scene",
        "desc": "Steady forward walking with closing oncoming pedestrian (Agent 5, closing from 10.5m to 6.5m).",
        "start_frame": 198,
        "end_frame": 348,
    },
    {
        "id": "HU_U03_headmotion",
        "name": "HU_U03 Strong Head Motion",
        "scenario": "strong head motion, camera translation, close hazard",
        "desc": "Strong head scanning / saccadic rotation (>60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m to 4.4m).",
        "start_frame": 3170,
        "end_frame": 3290,
    },
    {
        "id": "HU_U04_dense_crowd",
        "name": "HU_U04 Dense Plaza Crowd",
        "scenario": "dense object scene, multiple pedestrians, partial occlusion",
        "desc": "Dense public plaza crowd with 5 simultaneous active pedestrians (Agents 12, 15, 16, 17, 18), cluttered background, occlusions.",
        "start_frame": 1230,
        "end_frame": 1450,
    },
    {
        "id": "HU_U05_crossing",
        "name": "HU_U05 Lateral Crossing",
        "scenario": "lateral/crossing motion, camera translation",
        "desc": "Pedestrian crossing diagonally across user's forward walking corridor (Agent 21, distance ~28.7m -> 32.8m).",
        "start_frame": 1550,
        "end_frame": 1750,
    },
    {
        "id": "HU_U06_close_following",
        "name": "HU_U06 Close Following Cluster",
        "scenario": "multi-pedestrian cluster, camera translation, dynamic depth",
        "desc": "Walking forward in public plaza trailing a dynamic pedestrian cluster (Agents 50, 51, 53, 55).",
        "start_frame": 2110,
        "end_frame": 2310,
    },
    {
        "id": "HU_U07_receding",
        "name": "HU_U07 Receding Pedestrians",
        "scenario": "receding pedestrians, sparse object scene, open space",
        "desc": "Monotonically receding pedestrians walking ahead in open plaza (Agents 102 & 104, distance expanding 21.4m -> 39.5m).",
        "start_frame": 4180,
        "end_frame": 4380,
    },
    {
        "id": "HU_U08_disappearance_reappearance",
        "name": "HU_U08 Track Disappearance / Reappearance",
        "scenario": "temporary track disappearance/reappearance, camera translation",
        "desc": "Camera translation through architectural boundary with peripheral pedestrian disappearing behind boundary and re-emerging (Agent 179).",
        "start_frame": 8050,
        "end_frame": 8250,
    }
]


def analyze_sequence_telemetry(seq_info: dict, out_dir: Path, wall_clock_seconds: float):
    """
    Parses Phase 3B telemetry files (JSON and CSV) and calculates all required
    throughput, tracking, motion stability, TTC, risk, navigation, and policy metrics.
    """
    json_path = out_dir / "telemetry.json"
    frames_csv_path = out_dir / "telemetry_frames.csv"
    objects_csv_path = out_dir / "telemetry_objects.csv"

    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_frames = len(records)
    effective_fps = total_frames / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

    # 1. Latency Metrics
    det_lats = [r["latencies_ms"]["detector"] for r in records]
    track_lats = [r["latencies_ms"]["tracker"] for r in records]
    depth_lats = [r["latencies_ms"]["depth"] for r in records]
    cam_lats = [r["latencies_ms"].get("camera_motion", 0.0) for r in records]
    ttc_lats = [r["latencies_ms"]["ttc"] for r in records]
    risk_lats = [r["latencies_ms"]["risk"] for r in records]
    nav_lats = [r["latencies_ms"]["navigation"] for r in records]
    total_lats = [r["latencies_ms"]["total_frame"] for r in records]

    # Frame-by-frame FPS
    instant_fps = [1000.0 / lat for lat in total_lats if lat > 0]
    mean_fps = np.mean(instant_fps) if instant_fps else effective_fps
    median_fps = np.median(instant_fps) if instant_fps else effective_fps
    p50_lat = np.percentile(total_lats, 50) if total_lats else 0.0
    p95_lat = np.percentile(total_lats, 95) if total_lats else 0.0

    # Hardware stats
    device_name = records[-1].get("device", {}).get("device_name", "N/A") if records else "N/A"
    vram_alloc = max([r.get("device", {}).get("vram_allocated_mb", 0.0) for r in records]) if records else 0.0

    # 2. Policy & Detection Filtering Metrics
    raw_det_counts = []
    active_det_counts = []
    filtered_det_counts = []
    active_track_counts = []

    if frames_csv_path.exists():
        with open(frames_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_det_counts.append(int(row["raw_detections"]))
                active_det_counts.append(int(row["active_detections"]))
                filtered_det_counts.append(int(row["filtered_detections"]))
                active_track_counts.append(int(row["active_tracks"]))
    else:
        # Fallback from json
        for r in records:
            act = len(r.get("objects", []))
            active_det_counts.append(act)
            raw_det_counts.append(act)
            filtered_det_counts.append(0)
            active_track_counts.append(act)

    total_raw_proposals = sum(raw_det_counts)
    total_active_detections = sum(active_det_counts)
    total_policy_suppressed = sum(filtered_det_counts)
    suppression_pct = (total_policy_suppressed / total_raw_proposals * 100.0) if total_raw_proposals > 0 else 0.0

    # 3. Tracking Metrics
    all_tracks = set()
    person_detections = 0
    non_person_detections = 0
    track_lifetimes = {}
    track_classes = {}
    track_motion_histories = {}

    valid_ttc_values = []
    invalid_ttc_count = 0
    approach_counts = {"APPROACHING": 0, "RECEDING": 0, "STATIONARY": 0, "STABLE": 0, "UNKNOWN": 0}

    for r in records:
        for obj in r.get("objects", []):
            tid = obj["track_id"]
            cname = obj.get("class_name", "person")
            all_tracks.add(tid)
            track_lifetimes[tid] = track_lifetimes.get(tid, 0) + 1
            track_classes[tid] = cname

            if cname == "person":
                person_detections += 1
            else:
                non_person_detections += 1

            m_state = obj.get("approach_state", "UNKNOWN")
            approach_counts[m_state] = approach_counts.get(m_state, 0) + 1

            if tid not in track_motion_histories:
                track_motion_histories[tid] = []
            track_motion_histories[tid].append(m_state)

            ttc = obj.get("ttc_seconds")
            if ttc is not None and not np.isnan(ttc) and not np.isinf(ttc):
                valid_ttc_values.append(ttc)
            else:
                invalid_ttc_count += 1

    total_track_obs = sum(approach_counts.values())
    frac_approaching = (approach_counts["APPROACHING"] / total_track_obs) if total_track_obs > 0 else 0.0
    frac_receding = (approach_counts["RECEDING"] / total_track_obs) if total_track_obs > 0 else 0.0
    frac_static = ((approach_counts["STATIONARY"] + approach_counts["STABLE"]) / total_track_obs) if total_track_obs > 0 else 0.0
    frac_unknown = (approach_counts["UNKNOWN"] / total_track_obs) if total_track_obs > 0 else 0.0

    # Approach/Recede transitions
    motion_transitions = 0
    for tid, m_hist in track_motion_histories.items():
        for i in range(1, len(m_hist)):
            prev_s = m_hist[i-1]
            curr_s = m_hist[i]
            if prev_s in ("APPROACHING", "RECEDING") and curr_s in ("APPROACHING", "RECEDING") and prev_s != curr_s:
                motion_transitions += 1

    # TTC Stats
    min_ttc = float(np.min(valid_ttc_values)) if valid_ttc_values else None
    median_ttc = float(np.median(valid_ttc_values)) if valid_ttc_values else None
    p95_ttc = float(np.percentile(valid_ttc_values, 95)) if valid_ttc_values else None
    ttc_std = float(np.std(valid_ttc_values)) if len(valid_ttc_values) > 1 else 0.0

    # 4. Risk & Warning Metrics
    warning_counts = {"NO_WARNING": 0, "CAUTION": 0, "WARNING": 0}
    warning_sequence = []
    for r in records:
        w_state = r.get("global_warning", {}).get("state", "NO_WARNING")
        warning_counts[w_state] = warning_counts.get(w_state, 0) + 1
        warning_sequence.append(w_state)

    risk_transitions = 0
    for i in range(1, len(warning_sequence)):
        if warning_sequence[i] != warning_sequence[i-1]:
            risk_transitions += 1

    # 5. Navigation Metrics
    nav_counts = {"CONTINUE": 0, "CAUTION": 0, "AVOID_LEFT": 0, "AVOID_RIGHT": 0, "STOP": 0}
    nav_sequence = []
    nav_dirs = {"NONE": 0, "LEFT": 0, "RIGHT": 0}

    for r in records:
        n_state = r.get("navigation", {}).get("state", "CONTINUE")
        nav_counts[n_state] = nav_counts.get(n_state, 0) + 1
        nav_sequence.append(n_state)

        n_dir = r.get("navigation", {}).get("safe_direction", "NONE")
        nav_dirs[n_dir] = nav_dirs.get(n_dir, 0) + 1

    nav_transitions = 0
    for i in range(1, len(nav_sequence)):
        if nav_sequence[i] != nav_sequence[i-1]:
            nav_transitions += 1

    # Navigation consistency: % frames where decision matches previous frame
    nav_consecutive_matches = sum(1 for i in range(1, len(nav_sequence)) if nav_sequence[i] == nav_sequence[i-1])
    nav_consistency_pct = (nav_consecutive_matches / (len(nav_sequence) - 1) * 100.0) if len(nav_sequence) > 1 else 100.0

    # Write per-sequence run report
    report_md = f"""# HEADS-UP External Validation Run Report — {seq_info['name']}

**Sequence ID**: `{seq_info['id']}`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `{device_name}` on `cuda:0` | Python `{sys.version.split()[0]}`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Optical-Flow Ego-Motion Compensation)  

---

## 1. Scenario Description & Coverage
- **Description**: {seq_info['desc']}
- **Covered Scenarios**: {seq_info['scenario']}
- **Total Valid Frames**: {total_frames}
- **Resolution**: 1280 × 720 @ 30 FPS
- **Duration**: {total_frames / 30.0:.2f} seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | {wall_clock_seconds:.2f} | s |
| **Real Effective Throughput** | **{effective_fps:.2f}** | **FPS** |
| **Instantaneous FPS (Mean / Median)** | {mean_fps:.2f} / {median_fps:.2f} | FPS |
| **Total Frame Latency (p50 / Median)** | **{p50_lat:.2f}** | **ms** |
| **Total Frame Latency (p95)** | **{p95_lat:.2f}** | **ms** |
| **Total Frame Latency (Mean)** | {np.mean(total_lats):.2f} | ms |
| **Detector Latency (Mean)** | {np.mean(det_lats):.2f} | ms |
| **Tracker Latency (Mean)** | {np.mean(track_lats):.2f} | ms |
| **Depth Latency (Mean over all frames)** | {np.mean(depth_lats):.2f} | ms |
| **Optical-Flow / Ego-Motion Latency (Mean)** | {np.mean(cam_lats):.2f} | ms |
| **TTC Step Latency (Mean)** | {np.mean(ttc_lats):.2f} | ms |
| **Risk Engine Latency (Mean)** | {np.mean(risk_lats):.2f} | ms |
| **Navigation Latency (Mean)** | {np.mean(nav_lats):.2f} | ms |
| **Peak GPU VRAM Allocated** | {vram_alloc:.2f} | MB |

---

## 3. Perception, Policy & Tracking Metrics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Raw Detector Proposals** | {total_raw_proposals} | COCO-80 raw detections before policy |
| **Active Detections** | {total_active_detections} | Detections retained after indoor class policy |
| **Policy Suppressed Detections** | {total_policy_suppressed} ({suppression_pct:.1f}%) | Non-navigation objects filtered out |
| **Person Detections** | {person_detections} | Retained dynamic pedestrian proposals |
| **Relevant Non-Person Objects** | {non_person_detections} | Retained indoor navigation obstacles (chairs, etc.) |
| **Total Unique Tracks** | {len(all_tracks)} | Unique persistent IDs across sequence |
| **Longest Track Persistence** | {max(track_lifetimes.values()) if track_lifetimes else 0} frames | Max continuous track lifetime ({max(track_lifetimes.values()) / total_frames * 100:.1f}% of sequence) |

---

## 4. Kinematic Motion Stability & TTC

| Metric | Count / Value | Proportion |
| :--- | :---: | :---: |
| **Approaching Observations** | {approach_counts['APPROACHING']} | {frac_approaching * 100:.1f}% |
| **Receding Observations** | {approach_counts['RECEDING']} | {frac_receding * 100:.1f}% |
| **Static / Stable Observations** | {approach_counts['STATIONARY'] + approach_counts['STABLE']} | {frac_static * 100:.1f}% |
| **Unknown / Insufficient Observations** | {approach_counts['UNKNOWN']} | {frac_unknown * 100:.1f}% |
| **Kinematic State Transitions (Approach <-> Recede)** | {motion_transitions} | Motion direction reversals |
| **Valid TTC Calculations** | {len(valid_ttc_values)} | Finite, positive time horizons |
| **Invalid TTC Observations** | {invalid_ttc_count} | Divergence <= 0 or insufficient history |
| **Minimum TTC Observed** | {f'{min_ttc:.2f} s' if min_ttc else 'N/A'} | Closest collision hazard horizon |
| **Median TTC Observed** | {f'{median_ttc:.2f} s' if median_ttc else 'N/A'} | Typical collision horizon |
| **95th Percentile TTC** | {f'{p95_ttc:.2f} s' if p95_ttc else 'N/A'} | Long-range hazard horizon |
| **TTC Standard Deviation** | {f'{ttc_std:.2f} s' if valid_ttc_values else 'N/A'} | Temporal TTC dispersion |

---

## 5. Risk Engine & Navigation Decision States

### Warning State Distribution
- **NO_WARNING**: {warning_counts.get('NO_WARNING', 0)} frames ({warning_counts.get('NO_WARNING', 0) / total_frames * 100:.1f}%)
- **CAUTION**: {warning_counts.get('CAUTION', 0)} frames ({warning_counts.get('CAUTION', 0) / total_frames * 100:.1f}%)
- **WARNING**: {warning_counts.get('WARNING', 0)} frames ({warning_counts.get('WARNING', 0) / total_frames * 100:.1f}%)
- **Risk State Transitions**: {risk_transitions} switches

### Navigation Decisions
- **CONTINUE**: {nav_counts.get('CONTINUE', 0)} frames ({nav_counts.get('CONTINUE', 0) / total_frames * 100:.1f}%)
- **CAUTION**: {nav_counts.get('CAUTION', 0)} frames ({nav_counts.get('CAUTION', 0) / total_frames * 100:.1f}%)
- **AVOID_LEFT**: {nav_counts.get('AVOID_LEFT', 0)} frames ({nav_counts.get('AVOID_LEFT', 0) / total_frames * 100:.1f}%)
- **AVOID_RIGHT**: {nav_counts.get('AVOID_RIGHT', 0)} frames ({nav_counts.get('AVOID_RIGHT', 0) / total_frames * 100:.1f}%)
- **STOP**: {nav_counts.get('STOP', 0)} frames ({nav_counts.get('STOP', 0) / total_frames * 100:.1f}%)
- **Navigation State Transitions**: {nav_transitions} switches
- **Consecutive Frame Consistency**: {nav_consistency_pct:.1f}%

### Safe Steering Recommendations
- **NONE (Path unblocked / Stopped)**: {nav_dirs.get('NONE', 0)} frames ({nav_dirs.get('NONE', 0) / total_frames * 100:.1f}%)
- **RIGHT**: {nav_dirs.get('RIGHT', 0)} frames ({nav_dirs.get('RIGHT', 0) / total_frames * 100:.1f}%)
- **LEFT**: {nav_dirs.get('LEFT', 0)} frames ({nav_dirs.get('LEFT', 0) / total_frames * 100:.1f}%)
"""
    with open(out_dir / "run_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return {
        "sequence_id": seq_info["id"],
        "name": seq_info["name"],
        "scenario": seq_info["scenario"],
        "frames": total_frames,
        "wall_clock_s": wall_clock_seconds,
        "fps_effective": effective_fps,
        "fps_mean": mean_fps,
        "fps_median": median_fps,
        "p50_ms": p50_lat,
        "p95_ms": p95_lat,
        "det_lat_ms": float(np.mean(det_lats)),
        "track_lat_ms": float(np.mean(track_lats)),
        "depth_lat_ms": float(np.mean(depth_lats)),
        "cam_lat_ms": float(np.mean(cam_lats)),
        "ttc_lat_ms": float(np.mean(ttc_lats)),
        "risk_lat_ms": float(np.mean(risk_lats)),
        "nav_lat_ms": float(np.mean(nav_lats)),
        "vram_mb": vram_alloc,
        "raw_proposals": total_raw_proposals,
        "active_detections": total_active_detections,
        "policy_suppressed": total_policy_suppressed,
        "suppression_pct": suppression_pct,
        "person_detections": person_detections,
        "non_person_detections": non_person_detections,
        "total_tracks": len(all_tracks),
        "longest_track_frames": max(track_lifetimes.values()) if track_lifetimes else 0,
        "approaching_obs": approach_counts["APPROACHING"],
        "receding_obs": approach_counts["RECEDING"],
        "static_obs": approach_counts["STATIONARY"] + approach_counts["STABLE"],
        "unknown_obs": approach_counts["UNKNOWN"],
        "motion_transitions": motion_transitions,
        "valid_ttc_count": len(valid_ttc_values),
        "invalid_ttc_count": invalid_ttc_count,
        "min_ttc": min_ttc,
        "median_ttc": median_ttc,
        "p95_ttc": p95_ttc,
        "ttc_std": ttc_std,
        "no_warning_frames": warning_counts.get("NO_WARNING", 0),
        "caution_frames": warning_counts.get("CAUTION", 0),
        "warning_frames": warning_counts.get("WARNING", 0),
        "risk_transitions": risk_transitions,
        "nav_continue_frames": nav_counts.get("CONTINUE", 0),
        "nav_caution_frames": nav_counts.get("CAUTION", 0),
        "nav_avoid_left_frames": nav_counts.get("AVOID_LEFT", 0),
        "nav_avoid_right_frames": nav_counts.get("AVOID_RIGHT", 0),
        "nav_stop_frames": nav_counts.get("STOP", 0),
        "nav_transitions": nav_transitions,
        "nav_consistency_pct": nav_consistency_pct
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3B Systematic HEADS-UP External Validation.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth-cadence", type=int, default=2)
    parser.add_argument("--compensate-ego-motion", action="store_true", default=True)
    parser.add_argument("--results-dir", type=str, default="validation/results/heads_up")
    args = parser.parse_args()

    results_dir = REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = REPO_ROOT / "validation/datasets/heads_up/sequences"

    print("=" * 80)
    print("PHASE 3B — SYSTEMATIC HEADS-UP EXTERNAL VALIDATION (8 SEQUENCES)")
    print(f"Device: {args.device} | Cadence: {args.depth_cadence}:1 | Ego-Motion Compensation: {args.compensate_ego_motion}")
    print("=" * 80)

    summary_results = []

    for seq in HEADS_UP_PHASE3B_SEQUENCES:
        s_id = seq["id"]
        seq_folder = sequences_dir / s_id
        video_path = seq_folder / f"{s_id}.mp4"

        if not video_path.exists():
            print(f"Error: Sequence video {video_path} does not exist. Skipping.")
            continue

        out_seq_dir = results_dir / s_id
        out_seq_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running Sequence: {seq['name']} ({s_id})")
        print(f"    Video: {video_path}")
        print(f"    Output: {out_seq_dir}")

        cfg = load_config()
        cfg.setdefault("depth", {})["cadence"] = args.depth_cadence
        cfg.setdefault("camera_motion", {}).setdefault("forward_compensation", {})["enabled"] = args.compensate_ego_motion

        t0 = time.time()
        run_perception_pipeline(
            config=cfg,
            source_override=str(video_path),
            device_override=args.device,
            max_frames=None,  # Run full video
            headless=True,
            telemetry_dir=str(out_seq_dir),
            telemetry_prefix="telemetry",
            depth_cadence_override=args.depth_cadence
        )
        wall_time = time.time() - t0

        res = analyze_sequence_telemetry(seq, out_seq_dir, wall_time)
        if res:
            summary_results.append(res)
            print(f"    Done in {wall_time:.2f}s | {res['fps_effective']:.2f} FPS | p50: {res['p50_ms']:.1f}ms | Active Tracks: {res['total_tracks']}")

    # Save summary tables
    if summary_results:
        # 1. Sequence Results CSV
        seq_csv_path = results_dir / "phase3b_sequence_results.csv"
        with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_results[0].keys()))
            writer.writeheader()
            for r in summary_results:
                writer.writerow(r)
        print(f"\nSaved sequence results to {seq_csv_path}")

        # 2. Overall Summary CSV
        total_all_frames = sum(r["frames"] for r in summary_results)
        total_all_time = sum(r["wall_clock_s"] for r in summary_results)
        aggregate_fps = total_all_frames / total_all_time if total_all_time > 0 else 0.0

        all_p50 = [r["p50_ms"] for r in summary_results]
        all_p95 = [r["p95_ms"] for r in summary_results]

        summary_csv_path = results_dir / "phase3b_summary.csv"
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value", "Unit", "Notes"])
            writer.writerow(["Total Sequences", len(summary_results), "sequences", "Stratified behavioral sample"])
            writer.writerow(["Total Valid Frames", total_all_frames, "frames", "Real egocentric frames"])
            writer.writerow(["Total Wall-Clock Time", f"{total_all_time:.2f}", "seconds", f"cuda:0 ({summary_results[0]['vram_mb']:.1f} MB VRAM)"])
            writer.writerow(["Aggregate Effective Throughput", f"{aggregate_fps:.2f}", "FPS", "Real pipeline throughput"])
            writer.writerow(["Mean p50 Frame Latency", f"{np.mean(all_p50):.2f}", "ms", "Median per sequence averaged"])
            writer.writerow(["Mean p95 Frame Latency", f"{np.mean(all_p95):.2f}", "ms", "95th percentile averaged"])
            writer.writerow(["Total Raw Proposals", sum(r["raw_proposals"] for r in summary_results), "proposals", "COCO-80 raw detections"])
            writer.writerow(["Total Active Detections", sum(r["active_detections"] for r in summary_results), "detections", "Retained after class policy"])
            writer.writerow(["Total Policy Suppressed", sum(r["policy_suppressed"] for r in summary_results), "detections", "Filtered background objects"])
            writer.writerow(["Mean Suppression Rate", f"{np.mean([r['suppression_pct'] for r in summary_results]):.1f}", "%", "Policy suppression percentage"])
            writer.writerow(["Total Unique Tracks", sum(r["total_tracks"] for r in summary_results), "tracks", "BoT-SORT persistent tracks"])
            writer.writerow(["Total Approaching Observations", sum(r["approaching_obs"] for r in summary_results), "obs", "Closing hazards"])
            writer.writerow(["Total Receding Observations", sum(r["receding_obs"] for r in summary_results), "obs", "Expanding objects"])
            writer.writerow(["Total Valid TTC Count", sum(r["valid_ttc_count"] for r in summary_results), "calculations", "Finite positive TTC"])
            writer.writerow(["Mean Navigation Consistency", f"{np.mean([r['nav_consistency_pct'] for r in summary_results]):.1f}", "%", "Consecutive frame stability"])
        print(f"Saved aggregate summary to {summary_csv_path}")

        # 3. Metrics JSON
        metrics_json_path = results_dir / "phase3b_metrics.json"
        metrics_dict = {
            "metadata": {
                "benchmark": "HEADS-UP Phase 3B Systematic External Validation",
                "hardware": summary_results[0].get("name", "cuda:0"),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_sequences": len(summary_results),
                "total_frames": total_all_frames,
                "total_wall_clock_seconds": total_all_time,
                "aggregate_fps": aggregate_fps,
            },
            "aggregate_metrics": {
                "fps": aggregate_fps,
                "mean_p50_ms": float(np.mean(all_p50)),
                "mean_p95_ms": float(np.mean(all_p95)),
                "total_raw_proposals": sum(r["raw_proposals"] for r in summary_results),
                "total_active_detections": sum(r["active_detections"] for r in summary_results),
                "total_policy_suppressed": sum(r["policy_suppressed"] for r in summary_results),
                "mean_suppression_pct": float(np.mean([r['suppression_pct'] for r in summary_results])),
                "total_tracks": sum(r["total_tracks"] for r in summary_results),
                "total_valid_ttc": sum(r["valid_ttc_count"] for r in summary_results),
                "mean_nav_consistency_pct": float(np.mean([r['nav_consistency_pct'] for r in summary_results]))
            },
            "sequences": summary_results
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics JSON to {metrics_json_path}")

    print("\n" + "=" * 80)
    print("PHASE 3B EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
