"""Download a release artifact in verified HTTP ranges, then verify its SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time

import requests

p = argparse.ArgumentParser()
p.add_argument('url')
p.add_argument('destination', type=Path)
p.add_argument('sha256')
a = p.parse_args()
response = requests.get(a.url, headers={'Range':'bytes=0-0'}, timeout=60)
response.raise_for_status()
if response.status_code != 206 or len(response.content) != 1:
    raise RuntimeError('Server does not honor byte ranges')
url = response.url
size = int(response.headers['Content-Range'].split('/')[-1])
temp = a.destination.with_suffix(a.destination.suffix+'.download')
temp.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(temp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
os.ftruncate(fd, size)
chunk = 4*1024*1024

def fetch(start):
    end = min(size, start+chunk)-1
    for attempt in range(4):
        try:
            r = requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=90)
            r.raise_for_status()
            assert r.status_code == 206 and r.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
            assert len(r.content) == end-start+1
            written = 0
            while written < len(r.content):
                written += os.pwrite(fd, r.content[written:], start+written)
            return len(r.content)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)

try:
    done = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(fetch, s) for s in range(0,size,chunk)]):
            done += f.result()
            print(f'{done}/{size} bytes', flush=True)
finally:
    os.close(fd)
h = hashlib.sha256()
with temp.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024), b''):
        h.update(block)
if h.hexdigest() != a.sha256:
    raise RuntimeError('SHA256 verification failed')
temp.replace(a.destination)
print('SHA256 verified:', h.hexdigest(), flush=True)
