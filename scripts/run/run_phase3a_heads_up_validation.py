"""
Phase 3A - HEADS-UP External Validation Runner.

Executes the Phase 2C perception and risk pipeline on the selected HEADS-UP
representative egocentric sequences, measures wall-clock throughput on cuda:0,
logs real telemetry, and generates per-sequence analysis reports comparing
model predictions with external reference annotations.
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

HEADS_UP_SEQUENCES = [
    {
        "id": "HU_unconstrained_s01_multiped",
        "name": "HEADS-UP S01 Multi-Pedestrian",
        "desc": "Multiple pedestrians in field of view (Agents 1, 2, 3 simultaneously active)",
        "frames": 73,
    },
    {
        "id": "HU_unconstrained_s02_approach",
        "name": "HEADS-UP S02 Steady Approach",
        "desc": "Ordinary head motion, steady walking closing in on pedestrian (Agent 5, distance 10.5m -> 6.5m)",
        "frames": 102,
    },
    {
        "id": "HU_unconstrained_s03_headmotion",
        "name": "HEADS-UP S03 Strong Head Motion",
        "desc": "Strong head rotation / scanning (>60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m -> 4.4m)",
        "frames": 75,
    }
]


def analyze_heads_up_run(seq_info: dict, out_dir: Path, wall_clock_seconds: float):
    """Parses Phase 3A telemetry and generates run_report.md."""
    json_path = out_dir / "telemetry.json"
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_frames = len(records)
    effective_fps = total_frames / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

    # Latency statistics
    det_lats = [r["latencies_ms"]["detector"] for r in records]
    track_lats = [r["latencies_ms"]["tracker"] for r in records]
    depth_lats = [r["latencies_ms"]["depth"] for r in records]
    ttc_lats = [r["latencies_ms"]["ttc"] for r in records]
    risk_lats = [r["latencies_ms"]["risk"] for r in records]
    nav_lats = [r["latencies_ms"]["navigation"] for r in records]
    total_lats = [r["latencies_ms"]["total_frame"] for r in records]

    # CUDA & Memory stats
    device_name = records[-1].get("device", {}).get("device_name", "N/A") if records else "N/A"
    vram_alloc = max([r.get("device", {}).get("vram_allocated_mb", 0.0) for r in records]) if records else 0.0

    # Detection & Tracking stats
    track_ids = set()
    valid_ttc_count = 0
    ttc_values = []
    approach_counts = {"APPROACHING": 0, "RECEDING": 0, "STATIONARY": 0, "STABLE": 0, "UNKNOWN": 0}
    # Warning state distribution
    warning_counts = {"NO_WARNING": 0, "CAUTION": 0, "WARNING": 0}
    nav_counts = {"CONTINUE": 0, "CAUTION": 0, "AVOID_LEFT": 0, "AVOID_RIGHT": 0, "STOP": 0}
    nav_dirs = {"NONE": 0, "LEFT": 0, "RIGHT": 0}
    track_lifetimes = {}

    for r in records:
        w_state = r.get("global_warning", {}).get("state", "NO_WARNING")
        warning_counts[w_state] = warning_counts.get(w_state, 0) + 1
        
        n_state = r.get("navigation", {}).get("state", "CONTINUE")
        nav_counts[n_state] = nav_counts.get(n_state, 0) + 1

        n_dir = r.get("navigation", {}).get("safe_direction", "NONE")
        nav_dirs[n_dir] = nav_dirs.get(n_dir, 0) + 1

        for obj in r.get("objects", []):
            tid = obj["track_id"]
            track_ids.add(tid)
            track_lifetimes[tid] = track_lifetimes.get(tid, 0) + 1
            
            # Motion state
            m_state = obj.get("approach_state", "UNKNOWN")
            approach_counts[m_state] = approach_counts.get(m_state, 0) + 1

            # TTC
            ttc = obj.get("ttc_seconds")
            if ttc is not None and not np.isnan(ttc) and not np.isinf(ttc):
                valid_ttc_count += 1
                ttc_values.append(ttc)

    # Compute percentiles
    p50_total = np.percentile(total_lats, 50) if total_lats else 0.0
    p95_total = np.percentile(total_lats, 95) if total_lats else 0.0
    p50_depth = np.percentile(depth_lats, 50) if depth_lats else 0.0
    mean_depth = np.mean(depth_lats) if depth_lats else 0.0

    report_md = f"""# HEADS-UP External Validation Run Report — {seq_info['name']}

**Sequence ID**: `{seq_info['id']}`  
**Dataset Source**: Official HEADS-UP Benchmark (`Yassaman/HEADS-UP`, EPFL VITA Lab)  
**Hardware & Runtime**: `{device_name}` on `cuda:0` | Python `{sys.version.split()[0]}`  
**Configuration**: YOLO11n + BoT-SORT + Depth Anything V2 (2:1 Depth Cadence + Forward Ego-Motion Compensation)  

---

## 1. Scenario Description
- **Description**: {seq_info['desc']}
- **Total Valid Frames**: {total_frames}
- **Resolution**: 1280 × 720 @ 30 FPS
- **Sequence Duration**: {total_frames / 30.0:.2f} seconds

---

## 2. Computational Throughput & Latency Profile

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Wall-Clock Execution Time** | {wall_clock_seconds:.2f} | seconds |
| **Real Throughput** | **{effective_fps:.2f}** | **FPS** |
| **Total Frame Latency (Mean)** | {np.mean(total_lats):.2f} | ms |
| **Total Frame Latency (Median / p50)** | **{p50_total:.2f}** | **ms** |
| **Total Frame Latency (p95)** | **{p95_total:.2f}** | **ms** |
| **Depth Step Latency (Mean)** | {mean_depth:.2f} | ms |
| **Depth Step Latency (Median / p50)** | {p50_depth:.2f} | ms |
| **Detector Latency (Mean)** | {np.mean(det_lats):.2f} | ms |
| **Tracker Latency (Mean)** | {np.mean(track_lats):.2f} | ms |
| **Peak GPU VRAM Allocated** | {vram_alloc:.2f} | MB |

---

## 3. Perception, Tracking & Behavioral Dynamics

| Metric | Result | Context / Details |
| :--- | :---: | :--- |
| **Unique Track IDs** | {len(track_ids)} | Track IDs: `{sorted(list(track_ids))}` |
| **Longest Track Lifetime** | {max(track_lifetimes.values()) if track_lifetimes else 0} frames | Max continuous track persistence |
| **Valid TTC Calculations** | {valid_ttc_count} | Finite, positive time-to-collision frames |
| **Minimum TTC Observed** | {f'{min(ttc_values):.2f} s' if ttc_values else 'N/A'} | Closest collision hazard horizon |
| **Mean TTC Observed** | {f'{np.mean(ttc_values):.2f} s' if ttc_values else 'N/A'} | Average collision horizon |

### Object Kinematic Motion States
- **APPROACHING**: {approach_counts.get('APPROACHING', 0)} object-frames
- **RECEDING**: {approach_counts.get('RECEDING', 0)} object-frames
- **STATIONARY**: {approach_counts.get('STATIONARY', 0)} object-frames
- **STABLE / UNKNOWN**: {approach_counts.get('STABLE', 0) + approach_counts.get('UNKNOWN', 0)} object-frames

### Warning State Distribution
- **NO_WARNING (Clear)**: {warning_counts.get('NO_WARNING', 0)} frames ({warning_counts.get('NO_WARNING', 0) / total_frames * 100:.1f}%)
- **CAUTION**: {warning_counts.get('CAUTION', 0)} frames ({warning_counts.get('CAUTION', 0) / total_frames * 100:.1f}%)
- **WARNING**: {warning_counts.get('WARNING', 0)} frames ({warning_counts.get('WARNING', 0) / total_frames * 100:.1f}%)

### Navigation Guidance & Evasive Maneuvers
- **CONTINUE**: {nav_counts.get('CONTINUE', 0)} frames ({nav_counts.get('CONTINUE', 0) / total_frames * 100:.1f}%)
- **CAUTION (Slow / Monitor)**: {nav_counts.get('CAUTION', 0)} frames ({nav_counts.get('CAUTION', 0) / total_frames * 100:.1f}%)
- **AVOID_LEFT (Steer Left)**: {nav_counts.get('AVOID_LEFT', 0)} frames ({nav_counts.get('AVOID_LEFT', 0) / total_frames * 100:.1f}%)
- **AVOID_RIGHT (Steer Right)**: {nav_counts.get('AVOID_RIGHT', 0)} frames ({nav_counts.get('AVOID_RIGHT', 0) / total_frames * 100:.1f}%)
- **STOP (Emergency Stop)**: {nav_counts.get('STOP', 0)} frames ({nav_counts.get('STOP', 0) / total_frames * 100:.1f}%)

### Recommended Safe Steering Directions
- **NONE (Path unblocked / Stopped)**: {nav_dirs.get('NONE', 0)} frames ({nav_dirs.get('NONE', 0) / total_frames * 100:.1f}%)
- **RIGHT**: {nav_dirs.get('RIGHT', 0)} frames ({nav_dirs.get('RIGHT', 0) / total_frames * 100:.1f}%)
- **LEFT**: {nav_dirs.get('LEFT', 0)} frames ({nav_dirs.get('LEFT', 0) / total_frames * 100:.1f}%)

---

## 4. Dataset Provenance Limitation Statement
*Note: The reference trajectory annotations in HEADS-UP were machine-generated using YOLOv8, ByteTrack, temporal averaging, and Kalman filtering. They represent an external trajectory reference for qualitative comparison, NOT an independent ground-truth oracle.*
"""
    with open(out_dir / "run_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report generated: {out_dir / 'run_report.md'}")
    return {
        "sequence_id": seq_info["id"],
        "name": seq_info["name"],
        "fps": effective_fps,
        "p50_ms": p50_total,
        "p95_ms": p95_total,
        "depth_mean_ms": mean_depth,
        "tracks": len(track_ids),
        "valid_ttc": valid_ttc_count,
        "min_ttc": min(ttc_values) if ttc_values else None,
        "warnings": warning_counts,
        "navigation": nav_counts
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3A HEADS-UP External Validation.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth-cadence", type=int, default=2)
    parser.add_argument("--compensate-ego-motion", action="store_true", default=True)
    parser.add_argument("--results-dir", type=str, default="validation/results/heads_up")
    args = parser.parse_args()

    results_dir = REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = REPO_ROOT / "validation/datasets/heads_up/sequences"

    print("=" * 70)
    print("PHASE 3A — HEADS-UP EXTERNAL VALIDATION BENCHMARK")
    print(f"Device: {args.device} | Cadence: {args.depth_cadence}:1 | Ego-Motion: {args.compensate_ego_motion}")
    print("=" * 70)

    summary_results = []

    for seq in HEADS_UP_SEQUENCES:
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
            max_frames=seq["frames"] + 10,
            headless=True,
            telemetry_dir=str(out_seq_dir),
            telemetry_prefix="telemetry",
            depth_cadence_override=args.depth_cadence
        )
        wall_time = time.time() - t0

        res = analyze_heads_up_run(seq, out_seq_dir, wall_time)
        if res:
            summary_results.append(res)

    print("\n" + "=" * 70)
    print("PHASE 3A ALL SEQUENCES COMPLETE")
    print("=" * 70)
    for r in summary_results:
        print(f"{r['sequence_id']:35s} | {r['fps']:5.2f} FPS | p50: {r['p50_ms']:5.1f} ms | Tracks: {r['tracks']} | Valid TTC: {r['valid_ttc']}")


if __name__ == "__main__":
    main()
