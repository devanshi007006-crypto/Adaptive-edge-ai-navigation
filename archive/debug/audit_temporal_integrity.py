import os
import json
import csv
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation")
SEQ_DIR = REPO_ROOT / "validation/datasets/heads_up/sequences"
META_DIR = REPO_ROOT / "validation/datasets/heads_up/metadata"

sequences_config = [
    {"id": "HU_U01_multiped", "start": 0, "end": 120},
    {"id": "HU_U02_approach", "start": 198, "end": 348},
    {"id": "HU_U03_headmotion", "start": 3170, "end": 3290},
    {"id": "HU_U04_dense_crowd", "start": 1230, "end": 1450},
    {"id": "HU_U05_crossing", "start": 1550, "end": 1750},
    {"id": "HU_U06_close_following", "start": 2110, "end": 2310},
    {"id": "HU_U07_receding", "start": 4180, "end": 4380},
    {"id": "HU_U08_disappearance_reappearance", "start": 8050, "end": 8250}
]

poses_path = META_DIR / "unconstrained_camera_poses.csv"
with open(poses_path, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    poses_header = next(r)

print(f"Camera Poses Header: {poses_header}")

results = []

for seq in sequences_config:
    s_id = seq["id"]
    s_range = (seq["start"], seq["end"])
    exp_slots = seq["end"] - seq["start"] + 1
    
    f_dir = SEQ_DIR / s_id / "frames"
    png_files = sorted(os.listdir(f_dir)) if f_dir.exists() else []
    extracted_fids = sorted([int(f.replace("left_", "").replace(".png", "")) for f in png_files if f.endswith(".png")])
    
    valid_count = len(extracted_fids)
    missing_count = exp_slots - valid_count
    
    all_slots = list(range(seq["start"], seq["end"] + 1))
    missing_fids = [f for f in all_slots if f not in set(extracted_fids)]
    
    is_contiguous = (valid_count == exp_slots)
    
    deltas = np.diff(extracted_fids).tolist() if len(extracted_fids) > 1 else []
    delta_counts = {}
    for d in deltas:
        delta_counts[d] = delta_counts.get(d, 0) + 1

    results.append({
        "sequence_id": s_id,
        "range_start": seq["start"],
        "range_end": seq["end"],
        "expected_slots": exp_slots,
        "valid_images": valid_count,
        "missing_images": missing_count,
        "is_contiguous": is_contiguous,
        "extracted_fids_min": min(extracted_fids) if extracted_fids else None,
        "extracted_fids_max": max(extracted_fids) if extracted_fids else None,
        "deltas_summary": delta_counts,
        "missing_pattern": f"{missing_count} missing slots ({missing_count/exp_slots*100:.1f}%)"
    })

print("\n" + "=" * 110)
print(f"{'Seq ID':35s} | Range | Slots | Images | Missing | Contiguous? | Delta Distribution")
print("=" * 110)
for r in results:
    s_str = f"[{r['range_start']}, {r['range_end']}]"
    cont_str = "YES" if r["is_contiguous"] else "NO"
    d_str = ", ".join([f"d={k}:{v}" for k, v in sorted(r["deltas_summary"].items())])
    print(f"{r['sequence_id']:35s} | {s_str:13s} | {r['expected_slots']:5d} | {r['valid_images']:6d} | {r['missing_images']:7d} | {cont_str:11s} | {d_str}")

print("=" * 110)
total_slots = sum(r["expected_slots"] for r in results)
total_images = sum(r["valid_images"] for r in results)
total_missing = sum(r["missing_images"] for r in results)
print(f"TOTAL: {total_slots} slots, {total_images} valid images, {total_missing} missing images.")
