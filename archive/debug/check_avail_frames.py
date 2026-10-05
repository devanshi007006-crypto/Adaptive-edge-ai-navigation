import os
import tarfile
import re

tar_path = r"validation/datasets/heads_up/rgb_unconstrained.tar.gz"

target_ranges = {
    "HU_U01_multiped": (0, 120),
    "HU_U02_approach": (198, 348),
    "HU_U03_headmotion": (3170, 3290),
    "HU_U04_dense_crowd": (1230, 1450),
    "HU_U05_crossing": (1550, 1750),
    "HU_U06_close_following": (2110, 2310),
    "HU_U07_receding": (4180, 4380),
    "HU_U08_disappearance_reappearance": (8050, 8250)
}

# Collect target frame IDs set for fast check
target_fids = {}
for k, (s, e) in target_ranges.items():
    for f in range(s, e + 1):
        target_fids[f] = k

found_frames = {k: set() for k in target_ranges}

print("Scanning tar archive member by member...")
try:
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith(".png") and "left_" in member.name:
                fname = os.path.basename(member.name)
                fid_match = re.search(r'left_(\d+)\.png', fname)
                if fid_match:
                    fid = int(fid_match.group(1))
                    if fid in target_fids:
                        seq_id = target_fids[fid]
                        found_frames[seq_id].add(fid)
except Exception as e:
    print(f"Extraction stopped at error: {e}")

print("\n" + "=" * 90)
print(f"{'Sequence ID':35s} | {'Target Range':15s} | {'Target FIDs':15s} | {'Available in Archive'}")
print("=" * 90)

for k, (s, e) in target_ranges.items():
    raw_span = e - s + 1
    avail = len(found_frames[k])
    print(f"{k:35s} | [{s:5d}, {e:5d}] | {raw_span:15d} | {avail:15d}")

print("=" * 90)
