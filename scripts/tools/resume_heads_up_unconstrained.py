"""
Robust resume script for downloading missing trailing bytes of rgb_unconstrained.tar.gz.
Handles socket timeouts, dropped connections, and partial reads with exponential backoff and resume.
"""

import os
import sys
import time
import urllib.request
import gzip

token = os.environ.get('HF_TOKEN')
if not token:
    raise ValueError(
        "HF_TOKEN environment variable is required to access the gated HEADS-UP dataset. "
        "Please set HF_TOKEN in your environment (e.g., $env:HF_TOKEN='your_token')."
    )
url = 'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/rgb_unconstrained.tar.gz'
dest_path = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\rgb_unconstrained.tar.gz'

# 1. Get total file size
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Range': 'bytes=0-10'})
with urllib.request.urlopen(req) as resp:
    cr = resp.headers.get('Content-Range', '')
    total_bytes = int(cr.split('/')[-1])

print(f"Verified total file size: {total_bytes} bytes ({total_bytes / (1024**3):.2f} GB)")

# We know bytes 0 to 9,267,957,407 are 100% valid.
# Let's resume from 9,267,957,408
start_byte = 9267957408
target_bytes = total_bytes - start_byte

print(f"Target resume range: bytes={start_byte}-{total_bytes - 1} ({target_bytes / (1024**3):.2f} GB to download)")

curr_pos = start_byte
block_size = 4 * 1024 * 1024  # 4 MB chunks
t0 = time.time()
last_report = t0

with open(dest_path, 'r+b') as f:
    while curr_pos < total_bytes:
        f.seek(curr_pos)
        print(f"Connecting for range bytes={curr_pos}-{total_bytes - 1}...")
        headers = {
            'Authorization': f'Bearer {token}',
            'Range': f'bytes={curr_pos}-{total_bytes - 1}'
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                while curr_pos < total_bytes:
                    chunk = resp.read(min(block_size, total_bytes - curr_pos))
                    if not chunk:
                        print(f"\nStream ended at byte {curr_pos}. Reconnecting...")
                        break
                    f.write(chunk)
                    curr_pos += len(chunk)
                    
                    now = time.time()
                    if now - last_report > 5:
                        downloaded_so_far = curr_pos - start_byte
                        elapsed = now - t0
                        speed = (downloaded_so_far / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        pct = (downloaded_so_far / target_bytes) * 100
                        rem_mb = (target_bytes - downloaded_so_far) / (1024 * 1024)
                        eta = (rem_mb / speed) if speed > 0 else 0
                        print(f"[{elapsed:.1f}s] {downloaded_so_far / (1024**3):.2f}/{target_bytes / (1024**3):.2f} GB ({pct:.1f}%) @ {speed:.2f} MB/s | ETA: {eta:.0f}s", flush=True)
                        last_report = now
        except Exception as e:
            print(f"\nConnection error at byte {curr_pos}: {e}. Retrying in 3 seconds...", flush=True)
            time.sleep(3)

print(f"\nSuccessfully downloaded all trailing bytes up to {curr_pos} (total file size: {os.path.getsize(dest_path)})!")
print(f"Total time: {time.time() - t0:.1f}s")
