import csv
import ast
from collections import defaultdict

for split in ["easy", "hard"]:
    trajs_path = f"validation/datasets/heads_up/metadata/{split}_trajectory_raw.csv"
    print(f"\n=== Inspecting {split.upper()} frames 0 to 2500 ===")
    tracks = defaultdict(list)
    with open(trajs_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 2 and row[2].strip() not in ('[]', ''):
                try:
                    fid = int(row[1].strip('"'))
                    if fid <= 2500:
                        coords = ast.literal_eval(row[2])
                        for item in coords:
                            tracks[int(item['id'])].append((fid, float(item['x']), float(item['y']), float(item['z'])))
                except Exception:
                    pass
    print(f"Agents active in frames 0-2500: {len(tracks)}")
    for aid, pts in sorted(tracks.items(), key=lambda x: len(x[1]), reverse=True):
        fids = [p[0] for p in pts]
        d_start = (pts[0][1]**2 + pts[0][2]**2 + pts[0][3]**2)**0.5
        d_end = (pts[-1][1]**2 + pts[-1][2]**2 + pts[-1][3]**2)**0.5
        print(f"  Agent {aid:2d}: {len(pts)} frames ({min(fids)} to {max(fids)}), dist: {d_start:.1f}m -> {d_end:.1f}m")
