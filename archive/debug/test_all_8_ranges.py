import os
import tarfile
import re

tar_path = r"validation/datasets/heads_up/rgb_unconstrained.tar.gz"

target_ranges = {
    "HU_U01_multiped": (0, 120),
    "HU_U02_approach": (198, 348),
    "HU_U04_dense_crowd": (1230, 1450),
    "HU_U05_crossing": (1550, 1750),
    "HU_U06_close_following": (2110, 2310),
    "HU_U03_headmotion": (3170, 3290),
    "HU_U07_receding": (4180, 4380),
    "HU_U08_disappearance_reappearance": (8050, 8250)
}

# Collect target frame IDs set for fast check
target_fids = {}
for k, (s, e) in target_ranges.items():
    for f in range(s, e + 1):
        target_fids[f] = k

found_frames = {k: [] for k in target_ranges}

print("Testing tarfile reading member by member with try/except...")
with tarfile.open(tar_path, "r:gz") as tar:
    for member in tar:
        if member.name.endswith(".png") and "left_" in member.name:
            fname = os.path.basename(member.name)
            fid_match = re.search(r'left_(\d+)\.png', fname)
            if fid_match:
                fid = int(fid_match.group(1))
                if fid in target_fids:
                    seq_id = target_fids[fid]
                    found_frames[seq_id].append(fid)
                    print(f"Found frame {fid} for {seq_id} (total {len(found_frames[seq_id])})")

print("\n" + "=" * 90)
print(f"{'Sequence ID':35s} | {'Range':15s} | {'Raw Span [s,e]':15s} | {'Actual Extracted Frames'}")
print("=" * 90)

for k, (s, e) in target_ranges.items():
    raw_span = e - s + 1
    actual_frames = len(found_frames[k])
    print(f"{k:35s} | [{s:5d}, {e:5d}] | {raw_span:15d} | {actual_frames:15d}")
