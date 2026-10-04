"""
Test script to run the real YOLO detector + BoT-SORT tracker on data/test_clip.mp4.
Verifies track ID persistence and continuity across frames.
"""

import sys
import os
from pathlib import Path
import cv2

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from adaptive_navigation.perception.detector import YOLOObjectDetector
from adaptive_navigation.perception.tracker import BoTSORTTracker


def main():
    print("=" * 65)
    print("STEP 6: VERIFY BOT-SORT TRACKING AND PERSISTENT TRACK IDS")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")

    detector = YOLOObjectDetector(
        model_name_or_path="models/detector/yolo11n.pt",
        confidence_threshold=0.25,
        device=device,
    )

    tracker = BoTSORTTracker(
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        match_thresh=0.8,
        track_buffer=30,
    )

    clip_path = str(_repo_root / "data" / "test_clip.mp4")
    cap = cv2.VideoCapture(clip_path)

    frame_idx = 0
    track_history = {}  # track_id -> list of frame indices

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / 30.0
        dets = detector.detect(frame, timestamp)
        # Filter for person to track main hazard
        person_dets = [d for d in dets if d.class_name == "person"]
        tracked_objs = tracker.update(person_dets, frame, timestamp)

        for obj in tracked_objs:
            if obj.track_id not in track_history:
                track_history[obj.track_id] = []
            track_history[obj.track_id].append(frame_idx)

        if frame_idx < 5 or frame_idx % 15 == 0:
            summary_str = ", ".join(f"TID:{o.track_id} {o.class_name}(conf={o.confidence:.2f}, bbox=[{o.bbox[0]:.0f},{o.bbox[1]:.0f},{o.bbox[2]:.0f},{o.bbox[3]:.0f}])" for o in tracked_objs)
            print(f"Frame {frame_idx:02d}: {len(tracked_objs)} tracks -> [{summary_str}]")

        frame_idx += 1

    cap.release()
    print("-" * 65)
    print(f"Total Frames Processed: {frame_idx}")
    print("Track ID Persistence Summary:")
    for tid, frames in track_history.items():
        duration = len(frames)
        span = f"Frame {frames[0]} to {frames[-1]}"
        continuity = duration / (frames[-1] - frames[0] + 1) * 100.0
        print(f"  Track ID {tid}: active for {duration} frames ({span}, continuity={continuity:.1f}%)")

    # Validate tracking continuity
    main_track = max(track_history.values(), key=len) if track_history else []
    print(f"Longest continuous track duration: {len(main_track)} / {frame_idx} frames")
    if len(main_track) >= 50:
        print("BoT-SORT Tracking Persistence: PASS (High ID stability)")
    else:
        print("BoT-SORT Tracking Persistence: WARNING (Track fragmented)")
    print("=" * 65)


if __name__ == "__main__":
    main()
