import urllib.request
import tarfile
import gzip
import io
import os
import winreg

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        token, _ = winreg.QueryValueEx(key, 'HF_TOKEN')
    except Exception:
        pass

headers = {'Authorization': f'Bearer {token}', 'Range': 'bytes=0-20971520'}

for split in ['rgb_easy', 'rgb_hard']:
    url = f'https://huggingface.co/datasets/Yassaman/HEADS-UP/resolve/main/{split}.tar.gz'
    req = urllib.request.Request(url, headers=headers)
    print(f"\n--- Checking {split} (first 20 MB) ---")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            print(f"Downloaded {len(data)/1e6:.2f} MB")
    except Exception as e:
        print("Download error:", e)
        continue

    import zlib
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decomp = dobj.decompress(data)
        print(f"Decompressed {len(decomp)/1e6:.2f} MB")
        tar_stream = io.BytesIO(decomp)
        tar = tarfile.TarFile(fileobj=tar_stream)
        members = []
        for m in tar:
            members.append(m.name)
            if len(members) >= 10:
                break
        print(f"Found {len(members)} members in first slice:")
        for m in members[:10]:
            print(" ", m)
    except Exception as err:
        print("Decompress error:", err)
