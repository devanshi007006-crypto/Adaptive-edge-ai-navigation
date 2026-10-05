import urllib.request
import zlib
import re
import os
import winreg
from collections import defaultdict

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        token, _ = winreg.QueryValueEx(key, 'HF_TOKEN')
    except Exception:
        pass

def inspect_member_distribution(split_name, scan_mb=100):
    url = f'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/{split_name}.tar.gz'
    headers = {'Authorization': f'Bearer {token}', 'Range': f'bytes=0-{scan_mb*1024*1024}'}
    req = urllib.request.Request(url, headers=headers)
    
    print(f"\n==================== SCANNING {split_name} (first {scan_mb} MB) ====================")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        print("Error:", e)
        return
        
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decomp = dobj.decompress(data)
    except Exception as e:
        decomp = b""
        
    print(f"Downloaded {len(data)/1e6:.1f} MB, decompressed {len(decomp)/1e6:.1f} MB")
    
    # Parse members
    buf = decomp
    members = []
    while len(buf) >= 512:
        h = buf[:512]
        if h == b'\x00' * 512:
            buf = buf[512:]
            continue
        name = h[:100].split(b'\x00')[0].decode('utf-8', errors='ignore')
        size_str = h[124:136].split(b'\x00')[0].decode('utf-8', errors='ignore').strip()
        try:
            sz = int(size_str, 8) if size_str else 0
        except Exception:
            sz = 0
        pad = (512 - (sz % 512)) % 512
        tot = 512 + sz + pad
        if len(buf) < tot:
            break
        mat = re.search(r'left_(\d+)\.png', name)
        if mat:
            members.append(int(mat.group(1)))
        buf = buf[tot:]
        
    print(f"Total frame members found in slice: {len(members)}")
    if members:
        members.sort()
        print(f"Min FID: {min(members)}, Max FID: {max(members)}")
        # Look for contiguous streaks
        streaks = []
        curr_streak = [members[0]]
        for fid in members[1:]:
            if fid == curr_streak[-1] + 1:
                curr_streak.append(fid)
            else:
                if len(curr_streak) >= 3:
                    streaks.append((curr_streak[0], curr_streak[-1], len(curr_streak)))
                curr_streak = [fid]
        if len(curr_streak) >= 3:
            streaks.append((curr_streak[0], curr_streak[-1], len(curr_streak)))
            
        print(f"Found {len(streaks)} contiguous streaks of length >= 3:")
        for s in streaks[:15]:
            print(f"  Frames {s[0]} to {s[1]} (len={s[2]})")

inspect_member_distribution('rgb_easy', scan_mb=100)
inspect_member_distribution('rgb_hard', scan_mb=100)
