import urllib.request
import zlib
import tarfile
import io
import os
import re
import winreg
import time

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        token, _ = winreg.QueryValueEx(key, 'HF_TOKEN')
    except Exception:
        pass

def test_stream_scan(split_name, target_frames_set, max_stream_mb=500):
    url = f'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/{split_name}.tar.gz'
    headers = {'Authorization': f'Bearer {token}'}
    req = urllib.request.Request(url, headers=headers)
    
    print(f"\n==================== STREAMING {split_name} ====================")
    print(f"Looking for {len(target_frames_set)} target frames...")
    
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    t0 = time.time()
    downloaded_bytes = 0
    found_frames = set()
    
    # We buffer decompressed bytes into an in-memory or pipe stream
    decomp_buf = bytearray()
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        while True:
            chunk = resp.read(1024 * 1024) # 1 MB chunks
            if not chunk:
                break
            downloaded_bytes += len(chunk)
            
            try:
                decomp = dobj.decompress(chunk)
                decomp_buf.extend(decomp)
            except Exception as e:
                print("Decomp error:", e)
                break
                
            # Scan tar headers in decomp_buf
            # Tar header is 512 bytes
            while len(decomp_buf) >= 512:
                header = decomp_buf[:512]
                if header == b'\x00' * 512:
                    # Empty block
                    decomp_buf = decomp_buf[512:]
                    continue
                name = header[:100].split(b'\x00')[0].decode('utf-8', errors='ignore')
                size_str = header[124:136].split(b'\x00')[0].decode('utf-8', errors='ignore').strip()
                try:
                    size = int(size_str, 8) if size_str else 0
                except Exception:
                    size = 0
                
                # Total block size including padding
                pad = (512 - (size % 512)) % 512
                total_entry = 512 + size + pad
                
                if len(decomp_buf) < total_entry:
                    # Not enough data yet to consume this entry
                    break
                    
                mat = re.search(r'left_(\d+)\.png', name)
                if mat:
                    fid = int(mat.group(1))
                    if fid in target_frames_set:
                        found_frames.add(fid)
                        print(f"  [{downloaded_bytes/1e6:6.1f} MB dl] Found target frame {fid:5d}! ({len(found_frames)}/{len(target_frames_set)})")
                
                # Consume this entry
                decomp_buf = decomp_buf[total_entry:]
                
            if len(found_frames) >= len(target_frames_set):
                print(f"All {len(target_frames_set)} target frames found! Stopping stream.")
                break
                
            if downloaded_bytes > max_stream_mb * 1024 * 1024:
                print(f"Reached stream test limit ({max_stream_mb} MB). Found {len(found_frames)} frames.")
                break
                
    elapsed = time.time() - t0
    print(f"Stream scan finished in {elapsed:.1f}s ({downloaded_bytes/1e6:.1f} MB downloaded). Found {len(found_frames)} target frames.")
    return found_frames

# Test with a candidate set from rgb_easy: frames 4320-4350 (since we saw 4326 in first 20 MB!)
test_easy_fids = set(range(4300, 4350))
test_stream_scan('rgb_easy', test_easy_fids, max_stream_mb=50)
