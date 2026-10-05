import os
import gzip
import tarfile
import time

tar_path = r"validation/datasets/heads_up/rgb_unconstrained.tar.gz"

print("=" * 80)
print("HEADS-UP ARCHIVE DEEP INTEGRITY & READABILITY TEST")
print("=" * 80)

# 1. Local size vs Expected remote size
local_size = os.path.getsize(tar_path)
expected_remote_size = 12357276545  # 11.51 GB

print(f"Local file path: {tar_path}")
print(f"Local file size: {local_size} bytes ({local_size / (1024**3):.4f} GB)")
print(f"Expected size:   {expected_remote_size} bytes ({expected_remote_size / (1024**3):.4f} GB)")

size_match = (local_size == expected_remote_size)
print(f"Size Match:      {'PASS' if size_match else 'FAIL'}")

# 2. GZIP Header & Continuous Stream Integrity Test
print("\nPerforming GZIP stream integrity scan from start to EOF...")
t0 = time.time()
total_decompressed_bytes = 0
chunk_size = 32 * 1024 * 1024  # 32 MB chunks

gzip_valid = False
try:
    with gzip.open(tar_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_decompressed_bytes += len(chunk)
            if total_decompressed_bytes % (1024 * 1024 * 1024) < chunk_size:
                elapsed = time.time() - t0
                speed = (total_decompressed_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"  Decompressed {total_decompressed_bytes / (1024**3):.2f} GB ({speed:.1f} MB/s)...")
    gzip_valid = True
    print(f"GZIP stream scan complete! Total decompressed tar bytes: {total_decompressed_bytes} ({total_decompressed_bytes / (1024**3):.2f} GB) in {time.time() - t0:.1f}s.")
except Exception as e:
    print(f"GZIP stream scan FAILED at {total_decompressed_bytes} bytes: {e}")

# 3. TAR Member Scan Test
print("\nPerforming full TAR structure scan...")
t1 = time.time()
tar_valid = False
total_members = 0
left_png_members = 0

try:
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            total_members += 1
            if member.name.endswith(".png") and "left_" in member.name:
                left_png_members += 1
            if total_members % 5000 == 0:
                print(f"  Scanned {total_members} TAR members ({left_png_members} left PNGs)...")
    tar_valid = True
    print(f"TAR structure scan complete! Total members: {total_members}, Left PNGs: {left_png_members} in {time.time() - t1:.1f}s.")
except Exception as e:
    print(f"TAR structure scan FAILED: {e}")

print("\n" + "=" * 80)
print("ARCHIVE INTEGRITY CHECKPOINT REPORT")
print("=" * 80)
print(f"1. File Size Check:         {'PASS' if size_match else 'FAIL'}")
print(f"2. Full GZIP Stream Test:   {'PASS' if gzip_valid else 'FAIL'}")
print(f"3. Full TAR Structure Test: {'PASS' if tar_valid else 'FAIL'}")
print("=" * 80)
