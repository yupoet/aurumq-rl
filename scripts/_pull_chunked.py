"""Chunked resumable single-file download from OSS.

Resumes from existing .part file by sending Range header.
Crashes won't lose progress.

Usage:
    python scripts/_pull_chunked.py <key-suffix>

Example:
    python scripts/_pull_chunked.py gtja_panel_year=2022
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
LOCAL_ROOT = ROOT / "panels_v3_src" / "panels"
PREFIX = "aurumq-rl/handoffs/2026-05-08-panels-main-board-v3/panels/"
CHUNK = 32 * 1024 * 1024  # 32 MB

env = {}
raw = (ROOT / ".env").read_bytes()
for line in raw.decode("utf-8", errors="ignore").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])

def make_bucket():
    return oss2.Bucket(auth, "oss-cn-shenzhen.aliyuncs.com", "ledashi-oss",
                       connect_timeout=30)

def main():
    if len(sys.argv) != 2:
        print("usage: _pull_chunked.py <name-without-.parquet>")
        return 2
    name = sys.argv[1]
    key = PREFIX + name + ".parquet"
    local = LOCAL_ROOT / (name + ".parquet")
    # Use timestamped .part file to avoid conflicts with zombie holders.
    suffix = sys.argv[2] if len(sys.argv) > 2 else f".part{int(time.time())}"
    part = LOCAL_ROOT / (name + ".parquet" + suffix)

    bucket = make_bucket()
    head = bucket.head_object(key)
    total = int(head.content_length)
    print(f"target: {key}", flush=True)
    print(f"total size: {total:,} bytes", flush=True)

    start = part.stat().st_size if part.exists() else 0
    print(f"resume from: {start:,} ({100*start/total:.1f}%)", flush=True)

    if start >= total:
        print("already complete on .part — skipping download")
    else:
        t0 = time.time()
        with open(part, "ab") as f:
            cur = start
            while cur < total:
                end = min(cur + CHUNK - 1, total - 1)
                for attempt in range(3):
                    try:
                        rsp = bucket.get_object(key, byte_range=(cur, end))
                        data = rsp.read()
                        if len(data) != end - cur + 1:
                            raise IOError(f"chunk {cur}-{end}: got {len(data)} bytes")
                        f.write(data)
                        cur = end + 1
                        elapsed = time.time() - t0
                        rate = (cur - start) / 1024 / 1024 / max(elapsed, 0.001)
                        print(f"  {cur:,}/{total:,} ({100*cur/total:.1f}%) {rate:.1f} MB/s", flush=True)
                        break
                    except Exception as e:
                        print(f"    chunk {cur}-{end} attempt {attempt+1} fail: {e}", flush=True)
                        time.sleep(2)
                        # Reopen bucket on retry
                        bucket = make_bucket()
                else:
                    print(f"GIVE UP on chunk {cur}-{end}")
                    return 1

    actual = part.stat().st_size
    if actual != total:
        print(f"size mismatch: {actual} vs {total}")
        return 1

    print("verifying sha256...", flush=True)
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    got = h.hexdigest()

    m = json.loads((LOCAL_ROOT.parent / "MANIFEST.json").read_text())
    expected = next((e["sha256"] for e in m["files"] if e["path"] == name + ".parquet"), None)
    if got != expected:
        print(f"SHA MISMATCH: got={got[:12]}.. expected={expected[:12]}..")
        print("deleting bad .part — next run will redownload from scratch")
        part.unlink()
        return 1
    part.replace(local)
    print(f"OK: {local}  sha={got[:12]}..")
    return 0

if __name__ == "__main__":
    sys.exit(main())
