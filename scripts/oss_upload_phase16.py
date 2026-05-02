"""Upload Phase 16 handoff bundle to ledashi-oss/fromsz/.

Same skeleton as ``oss_upload_handoff.py`` (Phase 15) but pointed at the
Phase 16 artifacts.

Usage
-----
    python scripts/oss_upload_phase16.py
"""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-03-phase16-corrected-eval/"


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


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _upload_one(bucket: oss2.Bucket, local: Path, remote_key: str) -> tuple[str, str]:
    size = local.stat().st_size
    label = f"{size/1e6:7.2f} MiB" if size > 1e6 else f"{size/1e3:7.2f} KiB"
    try:
        meta = bucket.get_object_meta(remote_key)
        local_md5_hex = hashlib.md5(local.read_bytes()).hexdigest().lower()
        remote_etag = meta.etag.strip('"').lower()
        if remote_etag == local_md5_hex:
            print(f"  [skip] {remote_key} ({label}) -- etag matches")
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
        print("[error] OSS AK/SK missing from .env", file=sys.stderr)
        return 1
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME, connect_timeout=30)

    manifest: list[tuple[Path, str]] = []

    # 1) HANDOFF + decision_log
    p16 = ROOT / "runs" / "phase16_4h"
    for fname in (
        "HANDOFF_2026-05-03_phase16.md",
        "decision_log.md",
        "orchestrator.py",
        "orchestrator_stdout.log",
    ):
        local = p16 / fname
        if local.exists():
            manifest.append((local, fname if fname.startswith("HANDOFF") else f"reports/{fname}"))

    # 2) Per-run training + eval logs (small)
    for sub in (
        "phase16a_fixed_drop_mkt_300k.train.log",
        "phase16a_fixed_drop_mkt_300k.eval.log",
        "phase16b_fixed_drop_mkt_gtja_300k.train.log",
        "phase16b_fixed_drop_mkt_gtja_300k.eval.log",
        "phase16c_drop_mkt_gtja_450k.train.log",
        "phase16c_drop_mkt_gtja_450k.eval.log",
    ):
        local = p16 / sub
        if local.exists():
            manifest.append((local, f"logs/{sub}"))

    # 3) Production model bundle (Phase 16 candidates)
    prod = ROOT / "models" / "production"
    for fname in (
        "phase16_16a_drop_mkt_best.zip",
        "phase16_16a_drop_mkt_best_metadata.json",
        "phase16_16b_drop_mkt_gtja_best.zip",
        "phase16_16b_drop_mkt_gtja_best_metadata.json",
        "phase16_phase16c_drop_mkt_gtja_450k_best.zip",
        "phase16_phase16c_drop_mkt_gtja_450k_best_metadata.json",
    ):
        local = prod / fname
        if local.exists():
            manifest.append((local, f"models/{fname}"))

    # 4) Per-run sweep + metadata + factor importance
    runs = (
        "phase16a_fixed_drop_mkt_300k",
        "phase16b_fixed_drop_mkt_gtja_300k",
        "phase16c_drop_mkt_gtja_450k",
    )
    for run_name in runs:
        run_dir = ROOT / "runs" / run_name
        for fname in (
            "oos_sweep.md",
            "oos_sweep.json",
            "training_summary.json",
            "metadata.json",
        ):
            local = run_dir / fname
            if local.exists():
                manifest.append((local, f"run_sweeps/{run_name}/{fname}"))

    # 5) Factor importance staging dirs
    for sub in p16.glob("*__importance__*"):
        if sub.is_dir():
            for child in ("factor_importance.json", "metadata.json"):
                local = sub / child
                if local.exists():
                    manifest.append((local, f"factor_importance/{sub.name}/{child}"))

    print(f"[oss-upload] manifest: {len(manifest)} files")
    for local, sub in manifest:
        if not local.exists():
            print(f"  [missing] {local}")
            continue
        _upload_one(bucket, local, PREFIX + sub)

    print("[oss-upload] DONE.")
    print(
        f"  https://oss.console.aliyun.com/bucket/oss-cn-shenzhen/{BUCKET_NAME}/object?path={PREFIX}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
