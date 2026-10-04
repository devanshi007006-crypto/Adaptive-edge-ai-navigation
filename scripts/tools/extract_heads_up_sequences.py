"""
Extract selected representative sequences from HEADS-UP archive.

Extracts three targeted episodes covering:
1. HU_unconstrained_s01_multiped: Frames 0–120 (Multi-pedestrian dynamic scene: Agents 1, 2, 3)
2. HU_unconstrained_s02_approach: Frames 198–348 (Steady walking, approaching pedestrian: Agent 5, 10.5m -> 6.5m)
3. HU_unconstrained_s03_headmotion: Frames 3170–3290 (Severe head motion, >60 deg/s, close hazard: Agent 74, 1.7m -> 4.4m)

For each sequence:
- Saves raw unresized PNG frames to frames/
- Generates synchronized camera_poses.csv from unconstrained_camera_poses.csv
- Generates synchronized trajectories.csv from unconstrained_trajectory_raw.csv
- Writes metadata.json
- Compiles a high-quality lossless MP4 video (1920x1080 @ 30 FPS)
"""

import os
import tarfile
import re
import csv
import json
import ast
import cv2

tar_path = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\rgb_unconstrained.tar.gz'
meta_dir = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\metadata'
output_base = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\sequences'

sequences_config = [
    {
        "id": "HU_unconstrained_s01_multiped",
        "subset": "unconstrained",
        "start_frame": 0,
        "end_frame": 120,
        "desc": "Multi-pedestrian egocentric scene with 3 active pedestrians (Agents 1, 2, 3) in camera field of view."
    },
    {
        "id": "HU_unconstrained_s02_approach",
        "subset": "unconstrained",
        "start_frame": 198,
        "end_frame": 348,
        "desc": "Ordinary head motion, steady walking with closing oncoming pedestrian (Agent 5, distance 10.5m -> 6.5m)."
    },
    {
        "id": "HU_unconstrained_s03_headmotion",
        "subset": "unconstrained",
        "start_frame": 3170,
        "end_frame": 3290,
        "desc": "Strong head motion (angular velocity >60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m to 4.4m)."
    }
]

# Build set of all required frame numbers
all_target_frames = {}
for seq in sequences_config:
    s_id = seq["id"]
    for f in range(seq["start_frame"], seq["end_frame"] + 1):
        all_target_frames[f] = s_id

print(f"Total target frames to extract across all 3 sequences: {len(all_target_frames)}")

# Create directories
for seq in sequences_config:
    seq_dir = os.path.join(output_base, seq["id"])
    frames_dir = os.path.join(seq_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

# 1. Read camera poses and raw trajectories into memory for slicing
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

print(f"Loaded {len(poses_data)} poses, {len(trajs_data)} trajectory frames.")

# 2. Extract frames from tar.gz
print("Opening archive for sequence extraction...")
extracted_frames = {seq["id"]: {} for seq in sequences_config}

with tarfile.open(tar_path, "r:gz") as tar:
    count = 0
    found = 0
    for member in tar:
        count += 1
        mat = re.search(r'left_(\d+)\.png', member.name)
        if mat:
            fid = int(mat.group(1))
            if fid in all_target_frames:
                s_id = all_target_frames[fid]
                f_obj = tar.extractfile(member)
                img_data = f_obj.read()
                
                # Save frame
                seq_dir = os.path.join(output_base, s_id)
                out_path = os.path.join(seq_dir, "frames", f"left_{fid:08d}.png")
                with open(out_path, "wb") as out_fp:
                    out_fp.write(img_data)
                
                extracted_frames[s_id][fid] = out_path
                found += 1
                if found % 50 == 0 or found == len(all_target_frames):
                    print(f"Extracted {found}/{len(all_target_frames)} target frames (scanned {count} archive members)...")
                
                if found >= len(all_target_frames):
                    print("All target frames extracted!")
                    break

# 3. Write metadata, sliced CSVs, and videos for each sequence
for seq in sequences_config:
    s_id = seq["id"]
    seq_dir = os.path.join(output_base, s_id)
    frames_dir = os.path.join(seq_dir, "frames")
    
    extracted = sorted(extracted_frames[s_id].keys())
    print(f"\nFinalizing {s_id}: {len(extracted)} frames ({min(extracted)} to {max(extracted)})")
    
    # Metadata JSON
    meta = {
        "sequence_id": s_id,
        "subset": seq["subset"],
        "start_frame_id": min(extracted),
        "end_frame_id": max(extracted),
        "num_frames": len(extracted),
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "scenario_description": seq["desc"],
        "camera_model": "Stereolabs ZED Mini",
        "mount_type": "Cap-mounted head rig"
    }
    with open(os.path.join(seq_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    # Sliced poses CSV
    with open(os.path.join(seq_dir, "camera_poses.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_poses)
        for fid in extracted:
            if fid in poses_data:
                w.writerow(poses_data[fid])
                
    # Sliced trajectories CSV
    with open(os.path.join(seq_dir, "trajectories.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_trajs)
        for fid in extracted:
            if fid in trajs_data:
                w.writerow(trajs_data[fid])
                
    # Compile MP4 video
    mp4_path = os.path.join(seq_dir, f"{s_id}.mp4")
    print(f"Compiling video {mp4_path}...")
    sample_img = cv2.imread(extracted_frames[s_id][extracted[0]])
    h, w = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (w, h))
    for fid in extracted:
        img = cv2.imread(extracted_frames[s_id][fid])
        writer.write(img)
    writer.release()
    print(f"Video created: {mp4_path} ({os.path.getsize(mp4_path)/1024/1024:.2f} MB)")

print("\nAll HEADS-UP representative sequences prepared successfully.")
