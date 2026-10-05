"""
Resilient single-stream downloader for HEADS-UP rgb_unconstrained.tar.gz.
Downloads a clean, continuous GZIP DEFLATE stream without byte-range splitting,
with automatic resume if connection drops.
"""

import os
import sys
import time
import urllib.request
import winreg

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        token, _ = winreg.QueryValueEx(key, 'HF_TOKEN')
        os.environ['HF_TOKEN'] = token
    except Exception:
        pass

if not token:
    raise ValueError("HF_TOKEN environment variable is required.")

url = 'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/rgb_unconstrained.tar.gz'
dest_path = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\rgb_unconstrained.tar.gz'

# 1. Check total size
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Range': 'bytes=0-10'})
with urllib.request.urlopen(req) as resp:
    cr = resp.headers.get('Content-Range', '')
    total_bytes = int(cr.split('/')[-1])

print(f"Verified remote total file size: {total_bytes} bytes ({total_bytes / (1024**3):.2f} GB)")

# Check if file needs clean re-download
# If existing file is corrupted by multi-worker chunk boundary writing, re-download from 0
print("Starting clean single-stream download to prevent GZIP block corruption...")
block_size = 8 * 1024 * 1024  # 8 MB chunks
curr_pos = 0
t0 = time.time()
last_report = t0

# Download to temp file first
temp_path = dest_path + ".tmp"
with open(temp_path, 'wb') as f:
    while curr_pos < total_bytes:
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
                        print(f"\nStream cut at byte {curr_pos}. Reconnecting...", flush=True)
                        break
                    f.write(chunk)
                    curr_pos += len(chunk)

                    now = time.time()
                    if now - last_report > 5 or curr_pos == total_bytes:
                        elapsed = now - t0
                        speed = (curr_pos / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        pct = (curr_pos / total_bytes) * 100
                        rem_mb = (total_bytes - curr_pos) / (1024 * 1024)
                        eta = (rem_mb / speed) if speed > 0 else 0
                        print(f"[{elapsed:.1f}s] {curr_pos / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB ({pct:.1f}%) @ {speed:.2f} MB/s | ETA: {eta:.0f}s", flush=True)
                        last_report = now
        except Exception as e:
            print(f"\nConnection retry at byte {curr_pos}: {e}. Retrying in 3 seconds...", flush=True)
            time.sleep(3)

print(f"\nClean single-stream download complete! Replacing {dest_path}...")
if os.path.exists(dest_path):
    os.remove(dest_path)
os.rename(temp_path, dest_path)
print(f"File updated successfully ({os.path.getsize(dest_path)} bytes). Total time: {time.time() - t0:.1f}s")
