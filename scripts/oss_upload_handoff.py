"""Upload Phase 15 handoff bundle to ledashi-oss (Shenzhen).

Forces Shenzhen endpoint regardless of .env defaults. Reads AK/SECRET from
.env. Skips uploads when remote ETag matches local content.

Usage
-----
    python scripts/oss_upload_handoff.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-03-phase15-grand-champion/"


def _read_env() -> dict[str, str]:
    """Read .env (gbk-tolerant) and return a dict."""
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


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    import base64
    return base64.b64encode(h.digest()).decode("ascii")


def _upload_one(bucket: oss2.Bucket, local: Path, remote_key: str) -> tuple[str, str]:
    size = local.stat().st_size
    label = f"{size/1e6:7.2f} MiB" if size > 1e6 else f"{size/1e3:7.2f} KiB"

    # Skip if etag matches.
    try:
        meta = bucket.get_object_meta(remote_key)
        local_md5 = _md5_b64(local)
        local_md5_hex = hashlib.md5(local.read_bytes()).hexdigest().lower()
        remote_etag = meta.etag.strip('"').lower()
        if remote_etag == local_md5_hex:
            print(f"  [skip] {remote_key} ({label}) — etag matches")
            return "skip", remote_key
    except oss2.exceptions.NoSuchKey:
        pass

    print(f"  [put]  {remote_key} ({label})")
    headers = {"Content-MD5": _md5_b64(local)}
    bucket.put_object_from_file(remote_key, str(local), headers=headers)
    return "uploaded", remote_key


def main() -> int:
    env = _read_env()
    ak = env.get("OSS_ACCESS_KEY_ID")
    sk = env.get("OSS_ACCESS_KEY_SECRET")
    if not ak or not sk:
        print("[error] OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET missing from .env",
              file=sys.stderr)
        return 1
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")

    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME, connect_timeout=30)

    # File manifest: (local_path, remote_subpath_under_PREFIX)
    manifest: list[tuple[Path, str]] = []

    # 1) Top-level handoff doc
    manifest.append((
        ROOT / "runs" / "phase15_8h_exploration" / "HANDOFF_2026-05-03_phase15.md",
        "HANDOFF_2026-05-03_phase15.md",
    ))

    # 2) Production model bundle
    prod = ROOT / "models" / "production"
    for fname in (
        "phase15e_150k_GRAND_CHAMPION.zip",
        "phase15e_150k_metadata.json",
        "phase15e2_225k_continuation_peak.zip",
        "phase15e_100k_alt_peak.zip",
        "phase15a_674k_new_champion.zip",
        "phase15a_674k_metadata.json",
        "phase15a_125k_alt_peak.zip",
        "phase15a_200k_alt_peak.zip",
        "phase14c_600k_best_oos.zip",
        "phase14c_600k_metadata.json",
        "phase14c_600k_factor_importance_n10.json",
        "phase14c_200k_peak.zip",
    ):
        local = prod / fname
        if local.exists():
            manifest.append((local, f"models/{fname}"))

    # 3) Phase 15 reports
    p15 = ROOT / "runs" / "phase15_8h_exploration"
    for fname in (
        "decision_log.md",
        "final_report.md",
        "README.md",
        "experiment_queue.md",
        "mkt_factors_handoff.md",
        "phase15g_audit.md",
        "industry_oos_600k.md",
        "industry_oos_600k.json",
        "orchestrator.sh",
        "orchestrator.log",
        "after_orchestrator.sh",
        "launcher_commands.sh",
    ):
        local = p15 / fname
        if local.exists():
            manifest.append((local, f"reports/{fname}"))

    # 4) Per-run OOS sweep markdown + json (compact)
    for run_name in (
        "phase15a_14c_fine_ckpt_700k",
        "phase15b1_resume600k_lr3e5_300k",
        "phase15d1_cosine_lr_600k",
        "phase15e1_drop_mkt_300k",
        "phase15f_seed1_300k",
        "phase15e2_resume150k_lr3e5_400k",
        "phase15e_150k_champion_eval",  # holds factor_importance.json on the 150k champion
    ):
        run_dir = ROOT / "runs" / run_name
        for fname in ("oos_sweep.md", "oos_sweep.json", "training_summary.json", "metadata.json", "factor_importance.json"):
            local = run_dir / fname
            if local.exists():
                manifest.append((local, f"run_sweeps/{run_name}/{fname}"))

    print(f"[oss-upload] manifest: {len(manifest)} files")
    for local, sub in manifest:
        if not local.exists():
            print(f"  [missing] {local}")
            continue
        remote_key = PREFIX + sub
        _upload_one(bucket, local, remote_key)

    print(f"[oss-upload] DONE. Browse via:")
    print(f"  https://oss.console.aliyun.com/bucket/oss-cn-shenzhen/{BUCKET_NAME}/object?path={PREFIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
