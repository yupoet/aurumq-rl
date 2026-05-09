"""Sequential resumable pull from OSS with per-file retry + heartbeat.

Resume logic verifies SHA256 against MANIFEST.json (panels_v3_src/MANIFEST.json),
not just file size. Earlier parallel pulls left files at correct size but
corrupt bytes (probably overlapping writes from multiple threads).
"""
from __future__ import annotations
import hashlib, json, os, sys, time, traceback
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "panels_v3_src"
PREFIX = "aurumq-rl/handoffs/2026-05-08-panels-main-board-v3/"

raw = (ROOT / ".env").read_bytes()
try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    text = raw.decode("gbk", errors="ignore")

env = {}
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    env[k.strip()] = v.strip().strip('"').strip("'")

auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])

def make_bucket():
    return oss2.Bucket(
        auth, "oss-cn-shenzhen.aliyuncs.com", "ledashi-oss",
        connect_timeout=120,
    )

def get_with_retry(key: str, local: Path, expected_size: int, max_retries: int = 5) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            b = make_bucket()
            tmp = local.with_suffix(local.suffix + ".part")
            b.get_object_to_file(key, str(tmp))
            actual = tmp.stat().st_size
            if actual != expected_size:
                print(f"    size mismatch {actual} vs {expected_size} on attempt {attempt}; retry", flush=True)
                tmp.unlink(missing_ok=True)
                time.sleep(2 * attempt)
                continue
            tmp.replace(local)
            return True
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}", flush=True)
            time.sleep(2 * attempt)
    return False

def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Load manifest to enable sha-based resume verification.
    manifest_path = LOCAL / "MANIFEST.json"
    expected_sha: dict[str, str] = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        # manifest entries have "path" relative to panels/ subfolder
        for e in m.get("files", []):
            rel_in_panels = "panels/" + e["path"]
            expected_sha[rel_in_panels] = e.get("sha256", "")

    bucket = make_bucket()
    work = []
    for obj in oss2.ObjectIterator(bucket, prefix=PREFIX):
        rel = obj.key[len(PREFIX):]
        if not rel or rel.endswith("/") or obj.size == 0:
            continue
        local = LOCAL / rel
        if local.exists() and local.stat().st_size == obj.size:
            # SHA verify if we have it in manifest
            want = expected_sha.get(rel, "")
            if want:
                got = sha256_of(local)
                if got == want:
                    continue
                print(f"[resume] {rel}: size match but sha mismatch — REDOWNLOAD", flush=True)
                local.unlink(missing_ok=True)
            else:
                continue  # no sha known, assume size is enough
        work.append((obj.key, rel, local, obj.size))

    # Process smallest first — large files often time out and prevent
    # smaller files behind them from being attempted at all.
    work.sort(key=lambda w: w[3])

    print(f"[pull] {len(work)} files queued (sorted by size asc)", flush=True)
    for k, r, l, sz in work[:5]:
        print(f"  queued: {r} ({sz/1e6:.1f} MB)", flush=True)
    if not work:
        return 0

    t0 = time.time()
    done = 0
    fails = 0
    total_b = 0
    for i, (key, rel, local, size) in enumerate(work, start=1):
        local.parent.mkdir(parents=True, exist_ok=True)
        elapsed = int(time.time() - t0)
        print(f"  [{i}/{len(work)}] (t+{elapsed}s) GET {rel} ({size/1e6:.1f} MiB)...", flush=True)
        ok = get_with_retry(key, local, size)
        if ok:
            done += 1
            total_b += size
            print(f"    OK  total {total_b/1e9:.2f} GB / {elapsed}s elapsed", flush=True)
        else:
            fails += 1
            print(f"    GIVE UP after retries", flush=True)

    print(f"[pull] complete: {done} ok, {fails} failed, {total_b/1e9:.2f} GB", flush=True)
    return 1 if fails > 0 else 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
