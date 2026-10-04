import csv
import json
import os
import ast
from collections import defaultdict
import numpy as np

meta_dir = r"validation\datasets\heads_up\metadata"

def analyze_split(split):
    poses_path = os.path.join(meta_dir, f"{split}_camera_poses.csv")
    trajs_path = os.path.join(meta_dir, f"{split}_trajectory_raw.csv")
    
    # Load poses
    poses = {}
    with open(poses_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                fid = int(row[1].strip('"'))
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                q1, q2, q3, q4 = float(row[5]), float(row[6]), float(row[7]), float(row[8])
                poses[fid] = (x, y, z, q1, q2, q3, q4)
            except Exception:
                pass
                
    # Load trajectories
    # row[2] is coordinates string: e.g. [[agent_id, x, y, z], ...]
    agent_tracks = defaultdict(list) # agent_id -> [(fid, x, y, z)]
    frame_agents = defaultdict(list) # fid -> [agent_id]
    
    with open(trajs_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                fid = int(row[1].strip('"'))
                coords = ast.literal_eval(row[2])
                for item in coords:
                    aid = int(item[0])
                    x, y, z = float(item[1]), float(item[2]), float(item[3])
                    agent_tracks[aid].append((fid, x, y, z))
                    frame_agents[fid].append(aid)
            except Exception:
                pass
                
    print(f"\n=== {split.upper()} STATS ===")
    print(f"Total agents: {len(agent_tracks)}, frames with pedestrians: {len(frame_agents)}")
    
    # Sort agents by track length
    long_tracks = []
    for aid, track in agent_tracks.items():
        if len(track) >= 50:
            fids = [p[0] for p in track]
            # check continuity: are fids mostly contiguous?
            span = max(fids) - min(fids) + 1
            z_start = track[0][3]
            z_end = track[-1][3]
            dz = z_end - z_start
            long_tracks.append((aid, len(track), min(fids), max(fids), span, z_start, z_end, dz))
            
    long_tracks.sort(key=lambda x: x[1], reverse=True)
    print(f"Tracks >= 50 frames: {len(long_tracks)}")
    for t in long_tracks[:15]:
        aid, length, f_min, f_max, span, z_start, z_end, dz = t
        motion = "APPROACHING" if dz < -1.0 else ("RECEDING" if dz > 1.0 else "STATIONARY/LATERAL")
        print(f"  Agent {aid:3d}: len={length:4d}, frames={f_min:5d}-{f_max:5d} (span={span:4d}), z={z_start:4.1f}->{z_end:4.1f} ({motion})")

for s in ["easy", "hard", "unconstrained"]:
    analyze_split(s)
