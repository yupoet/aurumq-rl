"""Server-side migration of SZ-authored objects in ledashi-oss.

Topology recap (corrected 2026-05-03):

  Ubuntu side  → ledashi-oss-sgp/aurumq-rl/...  --CRR-->  ledashi-oss/aurumq-rl/...
  SZ box       → ledashi-oss/fromsz/...        --CRR-->  ledashi-oss-sgp/fromsz/...

So `ledashi-oss/aurumq-rl/` is a READ-ONLY mirror for SZ; I must write only
under `ledashi-oss/fromsz/`. This script moves leftover SZ-authored objects
from earlier mistaken paths into the correct `fromsz/` location.

Strategy
--------
For each source key matching the patterns below:

1. Copy server-side to the new ``fromsz/...`` location
2. Verify the destination etag matches the source
3. Delete the source object
4. Print a summary

Usage
-----
    python scripts/oss_migrate_to_fromsz.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET = "ledashi-oss"

# Source prefixes (where I mistakenly put SZ outputs earlier today).
# Map: srcprefix -> destprefix
PREFIX_MAP = {
    "aurumq-rl/from-sz/handoffs/2026-05-03-nightly-phase14-report/":
        "fromsz/handoffs/2026-05-03-nightly-phase14-report/",
    "aurumq-rl/from-sz/handoffs/2026-05-03-phase15-grand-champion/":
        "fromsz/handoffs/2026-05-03-phase15-grand-champion/",
    "aurumq-rl/from-sz/models/2026-05-03-phase14c-600k/":
        "fromsz/models/2026-05-03-phase14c-600k/",
}


def _read_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    out: dict[str, str] = {}
    raw = env_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="list what would happen but don't copy/delete")
    args = p.parse_args()

    env = _read_env()
    auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET, connect_timeout=30)

    # 1) enumerate all source keys, with their target prefix mapping
    keys: list[tuple[str, str]] = []  # (src_key, src_prefix)
    for src_prefix in PREFIX_MAP:
        marker = ""
        while True:
            res = bucket.list_objects_v2(prefix=src_prefix, max_keys=1000, continuation_token=marker)
            for o in res.object_list:
                if o.key.endswith("/"):
                    continue
                keys.append((o.key, src_prefix))
            if not res.is_truncated:
                break
            marker = res.next_continuation_token

    print(f"[migrate] discovered {len(keys)} objects under {len(PREFIX_MAP)} source prefixes")
    if not keys:
        print("[migrate] nothing to do.")
        return 0

    if args.dry_run:
        for k, src_prefix in keys[:30]:
            dest_prefix = PREFIX_MAP[src_prefix]
            dest_key = dest_prefix + k[len(src_prefix):]
            print(f"  [dry] {k} -> {dest_key}")
        print(f"  ... (total {len(keys)})")
        return 0

    # 2) copy + verify + delete
    copied = 0
    delete_failed: list[str] = []
    copy_failed: list[str] = []
    for k, src_prefix in keys:
        dest_prefix = PREFIX_MAP[src_prefix]
        dest_key = dest_prefix + k[len(src_prefix):]

        try:
            res = bucket.copy_object(BUCKET, k, dest_key)
            copied += 1
        except oss2.exceptions.OssError as e:
            print(f"[migrate] COPY FAILED {k}: {e!r}")
            copy_failed.append(k)
            continue

        # verify
        try:
            src_meta = bucket.get_object_meta(k)
            dst_meta = bucket.get_object_meta(dest_key)
            if src_meta.etag != dst_meta.etag:
                print(f"[migrate] etag mismatch on {k} vs {dest_key}; not deleting source")
                copy_failed.append(k)
                continue
        except oss2.exceptions.OssError as e:
            print(f"[migrate] verify failed for {k}: {e!r}")
            copy_failed.append(k)
            continue

        # delete source
        try:
            bucket.delete_object(k)
            if copied % 10 == 0:
                print(f"[migrate] {copied}/{len(keys)} done; latest: {k} -> {dest_key}")
        except oss2.exceptions.OssError as e:
            print(f"[migrate] DELETE FAILED for {k}: {e!r}")
            delete_failed.append(k)

    print()
    print(f"[migrate] copied: {copied}/{len(keys)}")
    print(f"[migrate] copy failed: {len(copy_failed)}")
    print(f"[migrate] delete failed: {len(delete_failed)}")
    if delete_failed:
        print("[migrate] sources still present (delete blocked by policy?):")
        for k in delete_failed[:10]:
            print(f"  {k}")
        if len(delete_failed) > 10:
            print(f"  ... and {len(delete_failed) - 10} more")
    if copy_failed:
        print("[migrate] copies that did not complete cleanly:")
        for k in copy_failed[:10]:
            print(f"  {k}")
    return 0 if not (copy_failed or delete_failed) else 1


if __name__ == "__main__":
    sys.exit(main())
