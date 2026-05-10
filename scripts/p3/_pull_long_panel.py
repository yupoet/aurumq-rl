"""Pull paris's long-panel-extension bundle from OSS to data/p3_4070_long/.

Mirrors scripts/_pull_p3_bundle.py pattern with resumable byte-range download.
Source: oss://ledashi-oss/aurumq-rl/handoffs/2026-05-10-long-panel-extension/
Dest:   data/p3_4070_long/
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET = "ledashi-oss"
PREFIX = "aurumq-rl/handoffs/2026-05-10-long-panel-extension/"
OUT_ROOT = ROOT / "data" / "p3_4070_long"
RETRIES_PER_FILE = 6


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    text = (ROOT / ".env").read_bytes().decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resumable_get(bucket: oss2.Bucket, key: str, dest: Path, expected_size: int) -> None:
    os.environ["NO_PROXY"] = "*"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == expected_size:
        return
    have = part.stat().st_size if part.exists() else 0
    last_err = None
    for attempt in range(RETRIES_PER_FILE):
        try:
            headers = {}
            if have > 0:
                headers["Range"] = f"bytes={have}-"
            obj = bucket.get_object(key, headers=headers)
            mode = "ab" if have > 0 else "wb"
            with part.open(mode) as out_f:
                while True:
                    block = obj.read(1 << 20)
                    if not block:
                        break
                    out_f.write(block)
                    have += len(block)
            part.rename(dest)
            return
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 30)
            print(f"  [retry {attempt + 1}/{RETRIES_PER_FILE}] {key}: {type(e).__name__}: {e}; sleep {sleep_s}s")
            time.sleep(sleep_s)
            have = part.stat().st_size if part.exists() else 0
    raise RuntimeError(f"download exhausted retries for {key}: {last_err}")


def _fmt(size: int) -> str:
    if size > 1 << 30:
        return f"{size / (1 << 30):6.2f} GiB"
    if size > 1 << 20:
        return f"{size / (1 << 20):6.2f} MiB"
    return f"{size / 1024:6.2f} KiB"


def main() -> int:
    env = _read_env()
    auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET, connect_timeout=30, app_name="long-panel-pull")

    print(f"[long-panel-pull] listing {PREFIX}")
    objs = []
    for o in oss2.ObjectIterator(bucket, prefix=PREFIX):
        if o.key.endswith("/"):
            continue
        objs.append((o.key, o.size))
    objs.sort(key=lambda x: x[1])
    total = sum(s for _, s in objs)
    print(f"[long-panel-pull] {len(objs)} files, {_fmt(total)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    bytes_pulled = 0
    t0 = time.time()
    for key, size in objs:
        rel = key[len(PREFIX):]
        dest = OUT_ROOT / rel
        if dest.exists() and dest.stat().st_size == size:
            n_skip += 1
            continue
        t_file = time.time()
        _resumable_get(bucket, key, dest, size)
        elapsed = time.time() - t_file
        speed = size / elapsed / (1 << 20) if elapsed > 0 else 0
        bytes_pulled += size
        n_done += 1
        print(f"  [{n_done + n_skip}/{len(objs)}] {_fmt(size)}  {speed:6.1f} MB/s  {rel}")

    elapsed = time.time() - t0
    print(f"\n[long-panel-pull] DONE. pulled {n_done} files ({_fmt(bytes_pulled)}), "
          f"skipped {n_skip}, in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
