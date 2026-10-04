"""
Inspect Depth Anything V2 output convention and numerical scale on test_clip.mp4.
Determines whether Depth Anything V2 outputs inverse depth (disparity: higher = closer)
or direct distance (lower = closer).
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from adaptive_navigation.perception.detector import YOLOObjectDetector
from adaptive_navigation.perception.tracker import BoTSORTTracker
from adaptive_navigation.perception.depth import DepthAnythingV2Estimator


def main():
    print("=" * 65)
    print("STEP 7: INSPECT DEPTH ANYTHING V2 OUTPUT CONVENTION")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")

    detector = YOLOObjectDetector(
        model_name_or_path="models/detector/yolo11n.pt",
        confidence_threshold=0.25,
        device=device,
    )
    tracker = BoTSORTTracker()
    depth_estimator = DepthAnythingV2Estimator(
        checkpoint_path="models/depth/depth_anything_v2_vits.pth",
        model_type="vits",
        device=device,
        is_metric=False,  # Uncalibrated relative depth
    )

    cap = cv2.VideoCapture(str(_repo_root / "data" / "test_clip.mp4"))
    frame_idx = 0
    records = []

    while cap.isOpened() and frame_idx < 60:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / 30.0
        dets = detector.detect(frame, timestamp)
        person_dets = [d for d in dets if d.class_name == "person"]
        tracks = tracker.update(person_dets, frame, timestamp)
        depth_res = depth_estimator.estimate_depth(frame, timestamp)
        obj_depths = depth_estimator.extract_all_object_depths(depth_res, tracks)

        if obj_depths:
            d_val = obj_depths[0].depth_value
            bbox = obj_depths[0].bbox
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            records.append((frame_idx, timestamp, d_val, box_area))
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx:02d} (t={timestamp:.2f}s): Object Depth Value = {d_val:.3f}, BBox Area = {box_area:.0f} px^2, Map Range = [{depth_res.min_depth:.2f}, {depth_res.max_depth:.2f}]")

        frame_idx += 1

    cap.release()
    print("-" * 65)
    first = records[0]
    last = records[-1]
    print(f"First Frame (t={first[1]:.2f}s): Depth Value = {first[2]:.3f}, BBox Area = {first[3]:.0f}")
    print(f"Last Frame  (t={last[1]:.2f}s): Depth Value = {last[2]:.3f}, BBox Area = {last[3]:.0f}")

    delta_depth = last[2] - first[2]
    delta_area = last[3] - first[3]

    print(f"Delta Depth: {delta_depth:+.3f}")
    print(f"Delta Area:  {delta_area:+.0f} px^2")

    if delta_depth > 0 and delta_area > 0:
        print("CONVENTION CONFIRMED: Depth Anything V2 outputs RELATIVE INVERSE DEPTH / DISPARITY.")
        print("  --> HIGHER VALUE = CLOSER (increases as obstacle approaches).")
    elif delta_depth < 0 and delta_area > 0:
        print("CONVENTION CONFIRMED: Depth outputs DIRECT DISTANCE.")
        print("  --> LOWER VALUE = CLOSER (decreases as obstacle approaches).")
    else:
        print("CONVENTION INDETERMINATE: inspect frame sequence.")
    print("=" * 65)


if __name__ == "__main__":
    main()
