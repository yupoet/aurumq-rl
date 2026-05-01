"""Resumable OSS downloader for the SGP bucket.

Why this exists
---------------
Plain ``oss2.Bucket.get_object_to_file`` does the whole transfer in one
streaming HTTP response. On flaky international links to ap-southeast-3
the connection gets reset partway through GB-scale objects, raising
``ConnectionError: Read timed out`` or ``ChunkedEncodingError:
IncompleteRead``. The whole download then has to start over.

This script:

1. ``HEAD`` the object to learn its content-length
2. Compare with the local file size
3. Issue ``GET`` with ``Range: bytes=<resume>-`` for the missing tail
4. Append the bytes to the local file with a small chunked write loop
5. On any ``ConnectionError`` / ``IncompleteRead`` / timeout, sleep and
   retry from the new local size — bounded retries, exponential backoff

It also bypasses any system proxy (``NO_PROXY=*``) since the
international link is the bottleneck, not a corporate proxy.

Usage
-----
    python scripts/oss_download_resumable.py \\
        --key   aurumq-rl/panels/factor_panel_combined_short_2023_2026.parquet \\
        --out   data/factor_panel_combined_short_2023_2026.parquet

Credentials are read from D:/dev/aurumq-handoffs/.secrets/credentials.env
(``OSS_ACCESS_KEY_ID`` / ``OSS_ACCESS_KEY_SECRET``) or from the
``OSS_ACCESS_KEY_ID`` / ``OSS_ACCESS_KEY_SECRET`` env vars.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import oss2
from oss2.exceptions import OssError

# Default to Shenzhen (`ledashi-oss`) — much faster from mainland China.
# A CRR rule (rule_id a4138ec9-2339-49b2-b1be-3d2457fd8ebb) mirrors
# `ledashi-oss-sgp/aurumq-rl/*` -> `ledashi-oss/aurumq-rl/*`, so the
# same keys are addressable from either bucket. Override with --bucket /
# --endpoint if you specifically need the SGP origin.
DEFAULT_BUCKET = "ledashi-oss"
DEFAULT_ENDPOINT = "https://oss-cn-shenzhen.aliyuncs.com"
CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks
MAX_RETRIES = 50
INITIAL_BACKOFF_S = 3
MAX_BACKOFF_S = 60


def _load_creds() -> tuple[str, str]:
    """Read AK from env or from D:/dev/aurumq-handoffs/.secrets/credentials.env."""
    ak = os.environ.get("OSS_ACCESS_KEY_ID")
    sk = os.environ.get("OSS_ACCESS_KEY_SECRET")
    if ak and sk:
        return ak, sk

    secrets_path = Path("D:/dev/aurumq-handoffs/.secrets/credentials.env")
    if secrets_path.exists():
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OSS_ACCESS_KEY_ID":
                ak = v.strip()
            elif k.strip() == "OSS_ACCESS_KEY_SECRET":
                sk = v.strip()
    if not ak or not sk:
        raise SystemExit(
            "OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET not found "
            "(neither in env nor in .secrets/credentials.env)"
        )
    return ak, sk


def _format_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _format_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1e6:
        return f"{bytes_per_sec / 1e6:.2f} MB/s"
    if bytes_per_sec >= 1e3:
        return f"{bytes_per_sec / 1e3:.0f} KB/s"
    return f"{bytes_per_sec:.0f} B/s"


def download_resumable(
    bucket: oss2.Bucket,
    key: str,
    out_path: Path,
    chunk_size: int = CHUNK_SIZE,
    max_retries: int = MAX_RETRIES,
) -> int:
    """Download `key` into `out_path` with resume + retry. Returns total bytes."""

    head = bucket.head_object(key)
    total = head.content_length
    print(f"[oss] target  {key}", flush=True)
    print(f"[oss] size    {_format_bytes(total)} ({total} bytes)", flush=True)
    print(f"[oss] etag    {head.etag}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = out_path.stat().st_size
        if existing == total:
            print(f"[oss] already complete: {out_path}", flush=True)
            return existing
        if existing > total:
            print(
                f"[oss] local file ({existing}) larger than remote ({total}) — "
                f"truncating",
                flush=True,
            )
            out_path.unlink()
            existing = 0
    else:
        existing = 0

    print(f"[oss] resume  from {_format_bytes(existing)} ({existing} bytes)", flush=True)

    attempt = 0
    backoff = INITIAL_BACKOFF_S
    last_print_time = time.monotonic()
    last_print_bytes = existing

    while existing < total:
        attempt += 1
        try:
            byte_range = (existing, total - 1)
            obj = bucket.get_object(key, byte_range=byte_range)
            with open(out_path, "ab") as fh:
                while True:
                    chunk = obj.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    existing += len(chunk)

                    now = time.monotonic()
                    if now - last_print_time >= 5:
                        rate = (existing - last_print_bytes) / (now - last_print_time)
                        pct = existing * 100 / total
                        eta = (total - existing) / rate if rate > 0 else float("inf")
                        print(
                            f"[oss] {pct:5.1f}%  "
                            f"{_format_bytes(existing)}/{_format_bytes(total)}  "
                            f"{_format_rate(rate)}  "
                            f"ETA {eta:5.0f}s",
                            flush=True,
                        )
                        last_print_time = now
                        last_print_bytes = existing
            backoff = INITIAL_BACKOFF_S
        except (
            ConnectionError,
            TimeoutError,
            OssError,
            Exception,
        ) as exc:
            if attempt >= max_retries:
                raise SystemExit(
                    f"[oss] gave up after {attempt} attempts at "
                    f"{_format_bytes(existing)}: {exc}"
                ) from exc
            existing = out_path.stat().st_size if out_path.exists() else 0
            print(
                f"[oss] {type(exc).__name__}: {exc}\n"
                f"[oss] retrying from {_format_bytes(existing)} "
                f"in {backoff}s (attempt {attempt + 1}/{max_retries})",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF_S, int(backoff * 1.7))
            continue

    final_size = out_path.stat().st_size
    if final_size != total:
        raise SystemExit(
            f"[oss] final size mismatch: expected {total}, got {final_size}"
        )
    print(f"[oss] DONE    {out_path} ({_format_bytes(final_size)})", flush=True)
    return final_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--key", required=True, help="OSS object key")
    parser.add_argument("--out", required=True, type=Path, help="local output path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"read chunk size in bytes (default {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"max retry attempts (default {MAX_RETRIES})",
    )
    args = parser.parse_args(argv)

    # Bypass any inherited system proxy — international link, no proxy benefits.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"

    ak, sk = _load_creds()
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, args.endpoint, args.bucket, connect_timeout=30)

    download_resumable(
        bucket,
        key=args.key,
        out_path=args.out,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
