import csv
import json
import os
from collections import defaultdict
import numpy as np

meta_dir = r"validation\datasets\heads_up\metadata"

splits = ["easy", "hard", "unconstrained"]

for split in splits:
    poses_path = os.path.join(meta_dir, f"{split}_camera_poses.csv")
    trajs_path = os.path.join(meta_dir, f"{split}_trajectory_raw.csv")
    
    print(f"\n==================== {split.upper()} ====================")
    
    # 1. Poses
    poses = []
    if os.path.exists(poses_path):
        with open(poses_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                if len(row) > 8:
                    try:
                        fid = int(row[1].strip('"'))
                        x, y, z = float(row[2]), float(row[3]), float(row[4])
                        q1, q2, q3, q4 = float(row[5]), float(row[6]), float(row[7]), float(row[8])
                        poses.append((fid, x, y, z, q1, q2, q3, q4))
                    except Exception:
                        pass
        print(f"Total poses: {len(poses)} (frame {poses[0][0]} to {poses[-1][0]})")
    
    # 2. Trajectories
    trajs_by_frame = defaultdict(list)
    agent_frames = defaultdict(list)
    agent_depths = defaultdict(list)
    
    if os.path.exists(trajs_path):
        with open(trajs_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            # Find column indices
            # e.g., frame, agent_id, x, y, z or similar
            # Let's inspect header
            print("Trajectory header:", header)
            for row in r:
                if len(row) > 1:
                    try:
                        fid = int(row[1].strip('"'))
                        # parse row
                        trajs_by_frame[fid].append(row)
                    except Exception:
                        pass
        print(f"Frames with trajectory labels: {len(trajs_by_frame)}")
