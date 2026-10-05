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
    agent_tracks = defaultdict(list) # agent_id -> [(fid, x, y, z)]
    frame_agents = defaultdict(list) # fid -> [agent_id]
    
    with open(trajs_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 2 and row[2].strip() not in ('[]', ''):
                try:
                    fid = int(row[1].strip('"'))
                    coords = ast.literal_eval(row[2])
                    for item in coords:
                        aid = int(item['id'])
                        x, y, z = float(item['x']), float(item['y']), float(item['z'])
                        agent_tracks[aid].append((fid, x, y, z))
                        frame_agents[fid].append(aid)
                except Exception as e:
                    pass
                
    print(f"\n=== {split.upper()} STATS ===")
    print(f"Total distinct agents: {len(agent_tracks)}, frames with pedestrians: {len(frame_agents)}")
    
    # Sort agents by track length
    long_tracks = []
    for aid, track in agent_tracks.items():
        if len(track) >= 30:
            fids = [p[0] for p in track]
            span = max(fids) - min(fids) + 1
            z_start = track[0][3]
            z_end = track[-1][3]
            x_start = track[0][1]
            x_end = track[-1][1]
            y_start = track[0][2]
            y_end = track[-1][2]
            # In camera frame, distance is sqrt(x^2 + y^2 + z^2)
            d_start = np.sqrt(x_start**2 + y_start**2 + z_start**2)
            d_end = np.sqrt(x_end**2 + y_end**2 + z_end**2)
            dd = d_end - d_start
            dx = x_end - x_start
            long_tracks.append((aid, len(track), min(fids), max(fids), span, d_start, d_end, dd, dx))
            
    long_tracks.sort(key=lambda x: x[1], reverse=True)
    print(f"Tracks >= 30 frames: {len(long_tracks)}")
    for t in long_tracks[:20]:
        aid, length, f_min, f_max, span, d_start, d_end, dd, dx = t
        if dd < -1.5:
            motion = "APPROACHING"
        elif dd > 1.5:
            motion = "RECEDING"
        elif abs(dx) > 1.5:
            motion = "CROSSING/LATERAL"
        else:
            motion = "STATIC/STABLE"
        print(f"  Agent {aid:3d}: len={length:4d}, frames={f_min:5d}-{f_max:5d} (span={span:4d}), dist={d_start:4.1f}->{d_end:4.1f}m ({motion})")

for s in ["easy", "hard", "unconstrained"]:
    analyze_split(s)
