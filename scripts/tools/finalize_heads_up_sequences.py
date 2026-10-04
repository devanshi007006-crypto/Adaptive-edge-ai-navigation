"""
Finalize HEADS-UP sequences by pruning undecodable placeholder frames,
synchronizing metadata, poses, and trajectories, and compiling clean MP4 videos.
"""

import os
import cv2
import json
import csv

base = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\sequences'
meta_dir = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\metadata'

poses_path = os.path.join(meta_dir, "unconstrained_camera_poses.csv")
trajs_path = os.path.join(meta_dir, "unconstrained_trajectory_raw.csv")

poses_data = {}
with open(poses_path, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    header_poses = next(r)
    for row in r:
        try:
            fid = int(row[1].strip('"'))
            poses_data[fid] = row
        except Exception:
            pass

trajs_data = {}
with open(trajs_path, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    header_trajs = next(r)
    for row in r:
        try:
            fid = int(row[1].strip('"'))
            trajs_data[fid] = row
        except Exception:
            pass

print(f"Loaded {len(poses_data)} poses, {len(trajs_data)} trajectory entries.")

descriptions = {
    "HU_unconstrained_s01_multiped": "Multi-pedestrian egocentric scene with multiple active pedestrians (Agents 1, 2, 3) in camera field of view.",
    "HU_unconstrained_s02_approach": "Ordinary head motion, steady walking with closing oncoming pedestrian (Agent 5, distance 10.5m -> 6.5m).",
    "HU_unconstrained_s03_headmotion": "Strong head motion (angular velocity >60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m to 4.4m)."
}

for s_id in sorted(os.listdir(base)):
    s_dir = os.path.join(base, s_id)
    if not os.path.isdir(s_dir):
        continue
    f_dir = os.path.join(s_dir, "frames")
    if not os.path.isdir(f_dir):
        continue

    print(f"\n--- Processing {s_id} ---")
    files = sorted(os.listdir(f_dir))
    valid_frames = []

    for f in files:
        fpath = os.path.join(f_dir, f)
        img = cv2.imread(fpath)
        if img is None or img.shape[0] == 0:
            # Delete corrupted / placeholder frame
            os.remove(fpath)
            print(f"Removed placeholder/corrupt file: {f}")
        else:
            fid = int(f.replace("left_", "").replace(".png", ""))
            valid_frames.append((fid, fpath, img))

    print(f"Valid frames retained: {len(valid_frames)}")
    if not valid_frames:
        print(f"Warning: No valid frames for {s_id}")
        continue

    valid_fids = [item[0] for item in valid_frames]
    h, w = valid_frames[0][2].shape[:2]

    # 1. Update metadata.json
    meta = {
        "sequence_id": s_id,
        "subset": "unconstrained",
        "start_frame_id": min(valid_fids),
        "end_frame_id": max(valid_fids),
        "num_frames": len(valid_frames),
        "fps": 30.0,
        "width": w,
        "height": h,
        "scenario_description": descriptions.get(s_id, ""),
        "camera_model": "Stereolabs ZED Mini",
        "mount_type": "Cap-mounted head rig"
    }
    with open(os.path.join(s_dir, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    # 2. Sliced poses CSV
    with open(os.path.join(s_dir, "camera_poses.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(header_poses)
        for fid in valid_fids:
            if fid in poses_data:
                wr.writerow(poses_data[fid])

    # 3. Sliced trajectories CSV
    with open(os.path.join(s_dir, "trajectories.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(header_trajs)
        for fid in valid_fids:
            if fid in trajs_data:
                wr.writerow(trajs_data[fid])

    # 4. Compile MP4 video
    mp4_path = os.path.join(s_dir, f"{s_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (w, h))
    for fid, fpath, img in valid_frames:
        # Convert RGBA to BGR if needed
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        writer.write(img)
    writer.release()
    print(f"Compiled video {mp4_path}: {len(valid_frames)} frames @ {w}x{h}, {os.path.getsize(mp4_path)/1024/1024:.2f} MB")

print("\nAll sequences finalized successfully!")
