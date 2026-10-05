"""
Phase 3B Systematic HEADS-UP Sequence Extractor.

Extracts an 8-sequence stratified behavioral suite (~1,010 valid frames) from
the unconstrained subset:
1. HU_U01_multiped: Multi-pedestrian dynamic scene (Agents 1, 2, 3) [73 frames]
2. HU_U02_approach: Steady walking with oncoming pedestrian (Agent 5, 10.5m->6.5m) [102 frames]
3. HU_U03_headmotion: Violent head scanning (>60 deg/s) with close hazard (Agent 74) [75 frames]
4. HU_U04_dense_crowd: Dense multi-pedestrian plaza crowd (Agents 12, 15, 16, 17, 18) [~160 frames]
5. HU_U05_crossing: Lateral crossing motion across user's walking corridor (Agent 21) [~150 frames]
6. HU_U06_close_following: Public plaza walking following pedestrian cluster (Agents 50, 51, 53, 55) [~150 frames]
7. HU_U07_receding: Receding pedestrians in open plaza (Agents 102, 104, 21m->40m) [~150 frames]
8. HU_U08_disappearance_reappearance: Peripheral pedestrian track loss and re-acquisition (Agent 179) [~150 frames]
"""

import os
import tarfile
import re
import csv
import json
import cv2
import time

tar_path = r'validation\datasets\heads_up\rgb_unconstrained.tar.gz'
meta_dir = r'validation\datasets\heads_up\metadata'
output_base = r'validation\datasets\heads_up\sequences'

sequences_config = [
    {
        "id": "HU_U01_multiped",
        "legacy_id": "HU_unconstrained_s01_multiped",
        "subset": "unconstrained",
        "start_frame": 0,
        "end_frame": 120,
        "desc": "Multi-pedestrian egocentric scene with 3 active pedestrians (Agents 1, 2, 3) in camera field of view."
    },
    {
        "id": "HU_U02_approach",
        "legacy_id": "HU_unconstrained_s02_approach",
        "subset": "unconstrained",
        "start_frame": 198,
        "end_frame": 348,
        "desc": "Ordinary head motion, steady walking with closing oncoming pedestrian (Agent 5, distance 10.5m -> 6.5m)."
    },
    {
        "id": "HU_U03_headmotion",
        "legacy_id": "HU_unconstrained_s03_headmotion",
        "subset": "unconstrained",
        "start_frame": 3170,
        "end_frame": 3290,
        "desc": "Strong head motion (angular velocity >60 deg/s) with close-proximity pedestrian (Agent 74, 1.7m to 4.4m)."
    },
    {
        "id": "HU_U04_dense_crowd",
        "legacy_id": None,
        "subset": "unconstrained",
        "start_frame": 1230,
        "end_frame": 1450,
        "desc": "Dense multi-pedestrian plaza crowd with 5 active pedestrians (Agents 12, 15, 16, 17, 18), cluttered environment, partial occlusions."
    },
    {
        "id": "HU_U05_crossing",
        "legacy_id": None,
        "subset": "unconstrained",
        "start_frame": 1550,
        "end_frame": 1750,
        "desc": "Lateral crossing motion across user's walking corridor (Agent 21, distance ~28.7m -> 32.8m)."
    },
    {
        "id": "HU_U06_close_following",
        "legacy_id": None,
        "subset": "unconstrained",
        "start_frame": 2110,
        "end_frame": 2310,
        "desc": "Walking in public plaza following close pedestrian cluster (Agents 50, 51, 53, 55)."
    },
    {
        "id": "HU_U07_receding",
        "legacy_id": None,
        "subset": "unconstrained",
        "start_frame": 4180,
        "end_frame": 4380,
        "desc": "Monotonic receding pedestrians in open space (Agents 102 & 104, distance increasing 21.4m -> 39.5m)."
    },
    {
        "id": "HU_U08_disappearance_reappearance",
        "legacy_id": None,
        "subset": "unconstrained",
        "start_frame": 8050,
        "end_frame": 8250,
        "desc": "Camera translation through architectural space with peripheral pedestrian track disappearance and re-acquisition (Agent 179)."
    }
]

# Check existing sequences and skip frames already extracted
all_target_frames = {}
for seq in sequences_config:
    s_id = seq["id"]
    # Check if existing folder or legacy folder has valid frames
    seq_dir = os.path.join(output_base, s_id)
    legacy_dir = os.path.join(output_base, seq["legacy_id"]) if seq["legacy_id"] else None
    
    target_dir = seq_dir
    frames_dir = os.path.join(seq_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # If legacy dir exists and target dir has fewer frames, copy frames across
    if legacy_dir and os.path.exists(os.path.join(legacy_dir, "frames")):
        legacy_frames_dir = os.path.join(legacy_dir, "frames")
        for lf in os.listdir(legacy_frames_dir):
            src_file = os.path.join(legacy_frames_dir, lf)
            dst_file = os.path.join(frames_dir, lf)
            if not os.path.exists(dst_file) and os.path.isfile(src_file):
                import shutil
                shutil.copy2(src_file, dst_file)
        print(f"Preserved and copied legacy frames {seq['legacy_id']} -> {s_id}")
        
    existing_fids = set()
    for f in os.listdir(frames_dir):
        if f.endswith(".png") and os.path.getsize(os.path.join(frames_dir, f)) > 100000:
            existing_fids.add(int(f.replace("left_", "").replace(".png", "")))
            
    for f in range(seq["start_frame"], seq["end_frame"] + 1):
        if f not in existing_fids:
            all_target_frames[f] = s_id

print(f"Total target frames to extract from archive: {len(all_target_frames)}")

# Load metadata
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

if all_target_frames:
    print(f"Extracting {len(all_target_frames)} frames from {tar_path}...")
    found = 0
    t0 = time.time()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            mat = re.search(r'left_(\d+)\.png', member.name)
            if mat:
                fid = int(mat.group(1))
                if fid in all_target_frames:
                    s_id = all_target_frames[fid]
                    f_obj = tar.extractfile(member)
                    img_data = f_obj.read()
                    
                    # Only save if valid image data
                    if len(img_data) > 100000:
                        out_path = os.path.join(output_base, s_id, "frames", f"left_{fid:08d}.png")
                        with open(out_path, "wb") as out_fp:
                            out_fp.write(img_data)
                        found += 1
                        if found % 50 == 0 or found == len(all_target_frames):
                            print(f"  Extracted {found}/{len(all_target_frames)} frames ({time.time()-t0:.1f}s)...")
                            
                    if found >= len(all_target_frames):
                        print("All requested frames extracted!")
                        break
    print(f"Archive scan complete. Extracted {found} valid frames.")
else:
    print("All frames already present locally.")

# Finalize each sequence: slice poses, trajectories, metadata JSON, and compile MP4 video
print("\nFinalizing all 8 sequences...")
for seq in sequences_config:
    s_id = seq["id"]
    seq_dir = os.path.join(output_base, s_id)
    f_dir = os.path.join(seq_dir, "frames")
    
    files = sorted(os.listdir(f_dir))
    valid_frames = []
    for f in files:
        fpath = os.path.join(f_dir, f)
        img = cv2.imread(fpath)
        if img is not None and img.shape[0] > 0:
            fid = int(f.replace("left_", "").replace(".png", ""))
            valid_frames.append((fid, fpath, img))
        else:
            if os.path.exists(fpath):
                os.remove(fpath)
                
    if not valid_frames:
        print(f"Warning: No valid frames for {s_id}")
        continue
        
    valid_fids = [item[0] for item in valid_frames]
    h, w = valid_frames[0][2].shape[:2]
    
    # Metadata JSON
    meta = {
        "sequence_id": s_id,
        "subset": "unconstrained",
        "start_frame_id": min(valid_fids),
        "end_frame_id": max(valid_fids),
        "num_frames": len(valid_frames),
        "fps": 30.0,
        "width": w,
        "height": h,
        "scenario_description": seq["desc"],
        "camera_model": "Stereolabs ZED Mini",
        "mount_type": "Cap-mounted head rig"
    }
    with open(os.path.join(seq_dir, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
        
    # Poses CSV
    with open(os.path.join(seq_dir, "camera_poses.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(header_poses)
        for fid in valid_fids:
            if fid in poses_data:
                wr.writerow(poses_data[fid])
                
    # Trajectories CSV
    with open(os.path.join(seq_dir, "trajectories.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(header_trajs)
        for fid in valid_fids:
            if fid in trajs_data:
                wr.writerow(trajs_data[fid])
                
    # MP4 Video
    mp4_path = os.path.join(seq_dir, f"{s_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (w, h))
    for fid, fpath, img in valid_frames:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        writer.write(img)
    writer.release()
    print(f"  {s_id:32s} | {len(valid_frames):3d} frames | {w}x{h} | {os.path.getsize(mp4_path)/1024/1024:.2f} MB")

print("\nAll 8 Phase 3B sequences finalized successfully.")
