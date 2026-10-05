import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_PATH = REPO_ROOT / "validation/results/live/logs/live_telemetry.json"
BASELINE_PATH = REPO_ROOT / "validation/results/live/live_risk_navigation_baseline.json"

def audit_baseline_telemetry():
    if not TELEMETRY_PATH.exists():
        print(f"Error: Telemetry file {TELEMETRY_PATH} does not exist.")
        return

    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} frames from {TELEMETRY_PATH.name}")

    state_counts = {}
    nav_counts = {}
    obj_counts = {}
    stop_caution_frames = []

    for rec in records:
        w_state = rec.get("global_warning_state", "UNKNOWN")
        n_state = rec.get("navigation_state", "UNKNOWN")
        
        state_counts[w_state] = state_counts.get(w_state, 0) + 1
        nav_counts[n_state] = nav_counts.get(n_state, 0) + 1

        objs = rec.get("objects", [])
        for obj in objs:
            cls_name = obj.get("class_name", "unknown")
            obj_counts[cls_name] = obj_counts.get(cls_name, 0) + 1

        if w_state == "CAUTION" and n_state == "STOP":
            stop_caution_frames.append({
                "frame_index": rec["frame_index"],
                "timestamp_sec": rec["timestamp_sec"],
                "active_tracks_count": rec["active_tracks_count"],
                "objects": objs
            })

    print("\n--- Telemetry Summary ---")
    print("Global Warning States:", state_counts)
    print("Navigation States:", nav_counts)
    print("Detected Object Counts:", obj_counts)
    print(f"Frames with CAUTION + STOP: {len(stop_caution_frames)} / {len(records)}")

    # Print detailed trace of first 10 CAUTION + STOP frames
    print("\n--- Sample Trace of CAUTION + STOP Frames ---")
    for item in stop_caution_frames[:10]:
        print(f"\n[Frame {item['frame_index']:04d}] Objects ({len(item['objects'])}):")
        for o in item["objects"]:
            print(f"  Track ID {o['track_id']} ({o['class_name']}): bbox={o['bbox']}, depth={o['depth_value']}, approach={o['approach_state']}, ttc={o['ttc_seconds']}, risk_level={o['risk_level']}, warn_state={o['warning_state']}")

    # Save baseline JSON
    baseline_data = {
        "total_frames": len(records),
        "warning_state_counts": state_counts,
        "navigation_state_counts": nav_counts,
        "object_counts": obj_counts,
        "caution_stop_frames_count": len(stop_caution_frames),
        "sample_caution_stop_frames": stop_caution_frames[:20]
    }

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\nSaved baseline analysis to {BASELINE_PATH}")

if __name__ == "__main__":
    audit_baseline_telemetry()
