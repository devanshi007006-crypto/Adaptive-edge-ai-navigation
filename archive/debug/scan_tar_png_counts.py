import os
import tarfile

tar_path = r"validation/datasets/heads_up/rgb_unconstrained.tar.gz"

print("Checking tar members for ranges...")
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

counts = {k: 0 for k in target_ranges}
fid_lists = {k: [] for k in target_ranges}

with tarfile.open(tar_path, "r:gz") as tar:
    for member in tar:
        if member.name.endswith(".png") and "left_" in member.name:
            fname = os.path.basename(member.name)
            fid = int(fname.replace("left_", "").replace(".png", ""))
            for k, (s, e) in target_ranges.items():
                if s <= fid <= e:
                    counts[k] += 1
                    fid_lists[k].append(fid)

print("-" * 80)
print(f"{'Sequence ID':35s} | {'Range':15s} | {'Tar PNG Count':15s} | {'Raw Inclusive Span'}")
print("-" * 80)
for k, (s, e) in target_ranges.items():
    print(f"{k:35s} | [{s:5d}, {e:5d}] | {counts[k]:15d} | {e - s + 1:3d}")
print("-" * 80)
print(f"{'TOTAL':35s} | {'-':15s} | {sum(counts.values()):15d} | {sum(e-s+1 for s,e in target_ranges.values()):3d}")
