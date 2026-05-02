"""Upload Phase 17 handoff bundle to ledashi-oss/fromsz/.

Same skeleton as oss_upload_phase16.py, pointed at Phase 17.
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
PREFIX = "fromsz/handoffs/2026-05-03-phase17-7h/"


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
        print("[error] OSS AK/SK missing", file=sys.stderr)
        return 1
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")
    bucket = oss2.Bucket(oss2.Auth(ak, sk), ENDPOINT, BUCKET_NAME, connect_timeout=30)

    manifest: list[tuple[Path, str]] = []

    # Top-level handoff
    handoff_dir = ROOT / "handoffs" / "2026-05-03-phase17-7h"
    for fname in (
        "HANDOFF_2026-05-03_phase17.md",
    ):
        local = handoff_dir / fname
        if local.exists():
            manifest.append((local, fname))

    # Phase 17 reports + orchestrator
    p17 = ROOT / "runs" / "phase17_7h"
    for fname in (
        "decision_log.md",
        "orchestrator.py",
        "orchestrator_stdout.log",
    ):
        local = p17 / fname
        if local.exists():
            manifest.append((local, f"reports/{fname}"))

    # Per-feature drilldown outputs
    for f in p17.glob("per_feature_drilldown__*"):
        manifest.append((f, f"reports/{f.name}"))

    # Per-run train + eval logs
    for fname in (
        "phase17_17a_drop_mkt_cyq_inst_seed42.train.log",
        "phase17_17a_drop_mkt_cyq_inst_seed42.eval.log",
        "phase17_17b_drop_mkt_seed1.train.log",
        "phase17_17b_drop_mkt_seed1.eval.log",
        "phase17_17c_drop_mkt_seed2.train.log",
        "phase17_17c_drop_mkt_seed2.eval.log",
        "phase17_17d_drop_mkt_seed3.train.log",
        "phase17_17d_drop_mkt_seed3.eval.log",
        "phase17_17e_drop_mkt_seed42_450k.train.log",
        "phase17_17e_drop_mkt_seed42_450k.eval.log",
    ):
        local = p17 / fname
        if local.exists():
            manifest.append((local, f"logs/{fname}"))

    # Production model bundle (Phase 17 candidates)
    prod = ROOT / "models" / "production"
    for fname in (
        "phase17_17a_drop_mkt_cyq_inst_seed42_best.zip",
        "phase17_17a_drop_mkt_cyq_inst_seed42_best_metadata.json",
        "phase17_17b_drop_mkt_seed1_best.zip",
        "phase17_17b_drop_mkt_seed1_best_metadata.json",
        "phase17_17c_drop_mkt_seed2_best.zip",
        "phase17_17c_drop_mkt_seed2_best_metadata.json",
        "phase17_17d_drop_mkt_seed3_best.zip",
        "phase17_17d_drop_mkt_seed3_best_metadata.json",
        "phase17_17e_drop_mkt_seed42_450k_best.zip",
        "phase17_17e_drop_mkt_seed42_450k_best_metadata.json",
    ):
        local = prod / fname
        if local.exists():
            manifest.append((local, f"models/{fname}"))

    # Per-run sweep + metadata
    runs = (
        "phase17_17a_drop_mkt_cyq_inst_seed42",
        "phase17_17b_drop_mkt_seed1",
        "phase17_17c_drop_mkt_seed2",
        "phase17_17d_drop_mkt_seed3",
        "phase17_17e_drop_mkt_seed42_450k",
    )
    for run_name in runs:
        run_dir = ROOT / "runs" / run_name
        for fname in ("oos_sweep.md", "oos_sweep.json", "training_summary.json", "metadata.json"):
            local = run_dir / fname
            if local.exists():
                manifest.append((local, f"run_sweeps/{run_name}/{fname}"))

    # Factor importance staging dirs
    for sub in p17.glob("*__importance__*"):
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
