import csv
import os

meta_dir = r"validation/datasets/heads_up/metadata"
poses_path = os.path.join(meta_dir, "unconstrained_camera_poses.csv")

frame_ids_in_metadata = set()
with open(poses_path, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        try:
            fid = int(row[1].strip('"'))
            frame_ids_in_metadata.add(fid)
        except Exception:
            pass

print(f"Total frame IDs in unconstrained_camera_poses.csv: {len(frame_ids_in_metadata)}")

ranges = [
    ("HU_U01_multiped", 0, 120, "Legacy 3A: 73 frames"),
    ("HU_U02_approach", 198, 348, "Legacy 3A: 102 frames"),
    ("HU_U03_headmotion", 3170, 3290, "Legacy 3A: 75 frames"),
    ("HU_U04_dense_crowd", 1230, 1450, "New"),
    ("HU_U05_crossing", 1550, 1750, "New"),
    ("HU_U06_close_following", 2110, 2310, "New"),
    ("HU_U07_receding", 4180, 4380, "New"),
    ("HU_U08_disappearance_reappearance", 8050, 8250, "New")
]

print("-" * 100)
print(f"{'Seq ID':35s} | {'Start':5s} | {'End':5s} | {'Raw Inclusive':13s} | {'Frames in Metadata':18s} | Notes")
print("-" * 100)

total_raw_inclusive = 0
total_meta_frames = 0

for name, s, e, note in ranges:
    meta_count = sum(1 for f in range(s, e + 1) if f in frame_ids_in_metadata)
    raw_range_inclusive = e - s + 1
    total_raw_inclusive += raw_range_inclusive
    total_meta_frames += meta_count
    print(f"{name:35s} | {s:5d} | {e:5d} | {raw_range_inclusive:13d} | {meta_count:18d} | {note}")

print("-" * 100)
print(f"{'TOTAL':35s} | {'-':5s} | {'-':5s} | {total_raw_inclusive:13d} | {total_meta_frames:18d} |")
