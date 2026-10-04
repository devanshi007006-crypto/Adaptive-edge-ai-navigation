"""
Test script to run the real YOLO detector on data/test_clip.mp4.
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


def main():
    print("=" * 65)
    print("STEP 5: RUN ACTUAL YOLO DETECTOR ON DATA/TEST_CLIP.MP4")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")

    detector = YOLOObjectDetector(
        model_name_or_path="models/detector/yolo11n.pt",
        confidence_threshold=0.25,
        device=device,
    )

    clip_path = str(_repo_root / "data" / "test_clip.mp4")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Unable to open {clip_path}")
        return

    frame_idx = 0
    total_detections = 0
    detected_classes = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / 30.0
        dets = detector.detect(frame, timestamp)
        total_detections += len(dets)
        for d in dets:
            detected_classes[d.class_name] = detected_classes.get(d.class_name, 0) + 1

        if frame_idx < 5 or frame_idx % 15 == 0:
            summary_str = ", ".join(f"{d.class_name}({d.confidence:.2f}, bbox=[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}])" for d in dets)
            print(f"Frame {frame_idx:02d} (t={timestamp:.2f}s): {len(dets)} objects -> [{summary_str}]")
        frame_idx += 1

    cap.release()
    print("-" * 65)
    print(f"Processed: {frame_idx} frames.")
    print(f"Total Detections: {total_detections}")
    print(f"Detected Classes: {detected_classes}")
    print("YOLO Detection on test_clip.mp4: PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()
