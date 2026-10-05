"""
Multi-threaded, resilient downloader for HEADS-UP rgb_unconstrained.tar.gz.
Features automatic reconnects for dropped sockets, progress reporting, and byte verification.
"""

import os
import urllib.request
import concurrent.futures
import time
import sys

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        token, _ = winreg.QueryValueEx(key, 'HF_TOKEN')
    except Exception:
        pass

if not token:
    raise ValueError(
        "HF_TOKEN environment variable is required to access the gated HEADS-UP dataset. "
        "Please set HF_TOKEN in your environment (e.g., $env:HF_TOKEN='your_token')."
    )

url = 'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/rgb_unconstrained.tar.gz'
dest_path = r'c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\datasets\heads_up\rgb_unconstrained.tar.gz'

os.makedirs(os.path.dirname(dest_path), exist_ok=True)

# 1. Get total file size
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Range': 'bytes=0-10'})
with urllib.request.urlopen(req) as resp:
    cr = resp.headers.get('Content-Range', '')
    total_bytes = int(cr.split('/')[-1])

print(f"Total file size: {total_bytes / (1024**3):.2f} GB ({total_bytes} bytes)")

# If file already exists and matches size, check if valid gzip
if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_bytes:
    print("File already downloaded and matches full size. Skipping download.")
    sys.exit(0)

# Pre-allocate file
with open(dest_path, 'wb') as f:
    f.seek(total_bytes - 1)
    f.write(b'\0')

num_workers = 8
chunk_size = total_bytes // num_workers
ranges = []
for i in range(num_workers):
    start = i * chunk_size
    end = (i + 1) * chunk_size - 1 if i < num_workers - 1 else total_bytes - 1
    ranges.append((i, start, end))

print(f"Starting {num_workers}-thread parallel download...")
t0 = time.time()
bytes_downloaded = [0] * num_workers


def download_chunk(worker_id, start_byte, end_byte):
    curr_pos = start_byte
    target_pos = end_byte
    block_size = 1024 * 1024

    with open(dest_path, 'r+b') as f:
        while curr_pos <= target_pos:
            headers = {
                'Authorization': f'Bearer {token}',
                'Range': f'bytes={curr_pos}-{target_pos}'
            }
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    f.seek(curr_pos)
                    while curr_pos <= target_pos:
                        to_read = min(block_size, target_pos - curr_pos + 1)
                        buf = resp.read(to_read)
                        if not buf:
                            # Stream cut off early, reconnect
                            break
                        f.write(buf)
                        curr_pos += len(buf)
                        bytes_downloaded[worker_id] = curr_pos - start_byte
            except Exception as e:
                time.sleep(2)

    return curr_pos - start_byte


with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(download_chunk, wid, s, e) for wid, s, e in ranges]

    # Progress reporter
    while not all(f.done() for f in futures):
        time.sleep(5)
        curr_total = sum(bytes_downloaded)
        elapsed = time.time() - t0
        speed = (curr_total / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        pct = (curr_total / total_bytes) * 100
        rem_mb = (total_bytes - curr_total) / (1024 * 1024)
        eta = (rem_mb / speed) if speed > 0 else 0
        print(f"[{elapsed:.1f}s] Downloaded {curr_total/(1024**3):.2f}/{total_bytes/(1024**3):.2f} GB ({pct:.1f}%) @ {speed:.2f} MB/s | ETA: {eta:.0f}s", flush=True)

    for f in futures:
        f.result()

total_time = time.time() - t0
final_size = os.path.getsize(dest_path)
print(f"Download complete in {total_time:.1f}s ({final_size / (1024**3):.2f} GB) @ {final_size / (1024**2) / total_time:.2f} MB/s.")
